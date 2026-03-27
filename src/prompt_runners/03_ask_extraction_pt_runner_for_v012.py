import os
import sys
import json
import time
import glob
import yaml
import shutil
import hashlib
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from dotenv import load_dotenv

# ----------------------------
# Optional provider SDK imports
# ----------------------------
# Install:
#   pip install openai google-genai
try:
    from openai import OpenAI  # type: ignore
except Exception:
    OpenAI = None  # type: ignore

try:
    from google import genai  # type: ignore
    from google.genai import types  # type: ignore
except Exception:
    genai = None  # type: ignore


# ============================
# Utilities
# ============================

def now_utc_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_text(path: str, content: str) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(path: str, obj: Any) -> None:
    safe_mkdir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def sha1_short(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:10]

_INVALID_FILENAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1F]')

def safe_filename_component(s: str) -> str:
    # Make a filesystem-safe component (Windows-safe).
    if not s:
        return "empty"
    cleaned = _INVALID_FILENAME_CHARS.sub("_", s)
    cleaned = cleaned.strip(" .")
    return cleaned or "empty"

def normalize_section_heading(heading: Optional[str]) -> str:
    if heading is None:
        return ""
    return str(heading)

def sort_key_int(obj: Dict[str, Any], key: str, default: int = 10**9) -> int:
    v = obj.get(key, None)
    try:
        return int(v)
    except Exception:
        return default

def render_template(template: str, values: Dict[str, str]) -> str:
    # Very small safe templater: replaces {{key}} occurrences.
    out = template
    for k, v in values.items():
        out = out.replace("{{" + k + "}}", v)
    return out


# ============================
# Prompt spec
# ============================

@dataclass
class RetryPolicy:
    max_retries: int = 1
    retry_on_invalid_json: bool = True

@dataclass
class TracePolicy:
    save_requests: bool = True
    save_responses: bool = True
    save_errors: bool = True

@dataclass
class PromptConfig:
    provider: str
    model: str
    temperature: Optional[float]
    max_output_tokens: int
    response_format: str
    timeout_s: int
    max_askr_items: int
    retry: RetryPolicy
    trace: TracePolicy
    provider_overrides: Dict[str, Any]

@dataclass
class PromptSpec:
    prompt_type: str
    revision: str
    config: PromptConfig
    system_command: str
    user_command: str


def load_prompt_yaml(prompt_path: str) -> PromptSpec:
    raw = yaml.safe_load(read_text(prompt_path))

    # Required
    prompt_type = raw.get("prompt_type")
    revision = raw.get("revision")
    system_command = raw.get("system_command")
    user_command = raw.get("user_command")
    cfg = raw.get("config", {})

    missing = [k for k, v in [
        ("prompt_type", prompt_type),
        ("revision", revision),
        ("system_command", system_command),
        ("user_command", user_command),
        ("config.provider", cfg.get("provider")),
        ("config.model", cfg.get("model")),
    ] if not v]
    if missing:
        raise ValueError(f"Prompt YAML missing required fields: {missing}")

    provider = str(cfg.get("provider")).strip().lower()
    model = str(cfg.get("model")).strip()

    # Common knobs
    temperature = cfg.get("temperature", None)
    if temperature is not None:
        try:
            temperature = float(temperature)
        except Exception:
            temperature = None

    max_output_tokens = int(cfg.get("max_output_tokens", 1500))
    response_format = str(cfg.get("response_format", "json")).strip().lower()
    timeout_s = int(cfg.get("timeout_s", 60))
    max_askr_items = int(cfg.get("max_askr_items", 3))
    if max_askr_items < 1:
        max_askr_items = 1

    # Retry/trace
    retry_cfg = cfg.get("retry", {}) or {}
    trace_cfg = cfg.get("trace", {}) or {}

    retry = RetryPolicy(
        max_retries=int(retry_cfg.get("max_retries", 1)),
        retry_on_invalid_json=bool(retry_cfg.get("retry_on_invalid_json", True)),
    )
    trace = TracePolicy(
        save_requests=bool(trace_cfg.get("save_requests", True)),
        save_responses=bool(trace_cfg.get("save_responses", True)),
        save_errors=bool(trace_cfg.get("save_errors", True)),
    )

    # Provider-specific metadata (do not assume API support)
    provider_overrides = {
        "google": cfg.get("google", {}) or {},
        "openai": cfg.get("openai", {}) or {},
    }

    return PromptSpec(
        prompt_type=prompt_type,
        revision=str(revision),
        config=PromptConfig(
            provider=provider,
            model=model,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_format=response_format,
            timeout_s=timeout_s,
            max_askr_items=max_askr_items,
            retry=retry,
            trace=trace,
            provider_overrides=provider_overrides,
        ),
        system_command=system_command,
        user_command=user_command,
    )


# ============================
# LLM Adapters
# ============================

class LLMClient:
    provider: str
    model: str

    def generate(
        self,
        system_text: str,
        user_text: str,
        *,
        temperature: Optional[float],
        max_output_tokens: int,
        timeout_s: int,
        provider_overrides: Optional[Dict[str, Any]] = None,
    ) -> "LLMResponse":
        raise NotImplementedError


@dataclass
class LLMResponse:
    text: str
    raw_response: Optional[Dict[str, Any]]
    meta: Dict[str, Any]


def _safe_model_dump(obj: Any) -> Optional[Dict[str, Any]]:
    # Best-effort conversion of SDK response object to a serializable dict.
    try:
        if hasattr(obj, "model_dump"):
            return obj.model_dump()
        if hasattr(obj, "to_dict"):
            return obj.to_dict()
    except Exception:
        return None
    def _make_json_safe(x: Any) -> Any:
        if isinstance(x, (bytes, bytearray)):
            # Convert bytes to a safe text representation for logging.
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        if isinstance(x, dict):
            return {str(k): _make_json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_make_json_safe(v) for v in x]
        return x
    try:
        safe_obj = _make_json_safe(obj)
        return json.loads(json.dumps(safe_obj, default=str))
    except Exception:
        return None

def _safe_json_dumps(obj: Any) -> str:
    # Use the same bytes-safe conversion for logging payloads.
    def _make_json_safe(x: Any) -> Any:
        if isinstance(x, (bytes, bytearray)):
            try:
                return x.decode("utf-8", errors="replace")
            except Exception:
                return str(x)
        if isinstance(x, dict):
            return {str(k): _make_json_safe(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [_make_json_safe(v) for v in x]
        return x
    safe_obj = _make_json_safe(obj)
    return json.dumps(safe_obj, ensure_ascii=False, indent=2, default=str)


def _extract_finish_reason(resp: Any) -> Optional[str]:
    # Best-effort finish_reason extraction across SDK variants.
    try:
        output = getattr(resp, "output", None)
        if isinstance(output, list):
            for item in output:
                if isinstance(item, dict):
                    fr = item.get("finish_reason") or item.get("status")
                else:
                    fr = getattr(item, "finish_reason", None) or getattr(item, "status", None)
                if fr:
                    return str(fr)
    except Exception:
        return None
    return None


def _askr_json_schema(max_askr_items: int) -> Dict[str, Any]:
    # JSON Schema used for OpenAI Structured Outputs (strict).
    wl_enum = sorted(list(ALLOWED_WORKLOADS))
    askr_enum = sorted(list(ALLOWED_ASKR))

    return {
        "name": "askr_extraction_schema",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["extractions"],
            "properties": {
                "extractions": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "workload_type",
                            "workload_summary",
                            "workload_quote",
                            "related_askr"
                        ],
                        "properties": {
                            "workload_type": {
                                "anyOf": [
                                    {"type": "string", "enum": wl_enum},
                                    {"type": "null"}
                                ]
                            },
                            "workload_summary": {"type": "string"},
                            "workload_quote": {"type": "string"},
                            "related_askr": {
                                "type": "array",
                                "maxItems": max_askr_items,
                                "items": {
                                    "type": "object",
                                    "additionalProperties": False,
                                    "required": [
                                        "askr_type",
                                        "askr_summary",
                                        "verbatim_quote",
                                        "gap_signal"
                                    ],
                                    "properties": {
                                        "askr_type": {"type": "string", "enum": askr_enum},
                                        "askr_summary": {"type": "string"},
                                        "verbatim_quote": {"type": "string"},
                                        "gap_signal": {"type": "boolean"}
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }


class OpenAIAdapter(LLMClient):
    def __init__(self, model: str, api_key: str):
        if OpenAI is None:
            raise RuntimeError("OpenAI SDK not installed. pip install openai")
        self.provider = "openai"
        self.model = model
        self.client = OpenAI(api_key=api_key)

    def generate(
        self,
        system_text: str,
        user_text: str,
        *,
        temperature: Optional[float],
        max_output_tokens: int,
        timeout_s: int,
        provider_overrides: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        # Note: GPT-5 does not support temperature.
        kwargs: Dict[str, Any] = {}
        is_gpt5 = self.model.startswith("gpt-5")
        if temperature is not None and not is_gpt5:
            kwargs["temperature"] = temperature
        overrides = (provider_overrides or {}).get("openai", {}) or {}
        if "reasoning" in overrides:
            kwargs["reasoning"] = overrides["reasoning"]

        meta: Dict[str, Any] = {
            "provider": "openai",
            "model": self.model,
            "max_output_tokens": max_output_tokens,
        }

        use_structured = bool(overrides.get("structured_output", True))
        structured_info = {"attempted": False, "used": False, "error": None}
        resp = None

        # Attempt schema-enforced structured output when possible.
        if use_structured:
            structured_info["attempted"] = True
            try:
                response_format = {
                    "type": "json_schema",
                    "json_schema": _askr_json_schema(int(overrides.get("max_askr_items", 3)))
                }
                resp = self.client.responses.create(
                    model=self.model,
                    input=[
                        {"role": "system", "content": system_text},
                        {"role": "user", "content": user_text},
                    ],
                    response_format=response_format,
                    max_output_tokens=max_output_tokens,
                    timeout=timeout_s,
                    **kwargs
                )
                structured_info["used"] = True
            except Exception as e:
                # Clean fallback for older SDKs/models that don't support response_format.
                structured_info["error"] = f"{type(e).__name__}: {e}"
                resp = None

        if resp is None:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                max_output_tokens=max_output_tokens,
                timeout=timeout_s,
                **kwargs
            )

        meta["structured_output"] = structured_info
        meta["response_status"] = getattr(resp, "status", None)
        meta["incomplete_details"] = getattr(resp, "incomplete_details", None)
        meta["finish_reason"] = _extract_finish_reason(resp)

        raw = _safe_model_dump(resp)
        text = getattr(resp, "output_text", None) or ""
        return LLMResponse(text=text, raw_response=raw, meta=meta)


class GeminiAdapter(LLMClient):
    def __init__(self, model: str, api_key: str):
        if genai is None:
            raise RuntimeError("Google GenAI SDK not installed. pip install google-genai")
        self.provider = "google"
        self.model = model
        self.client = genai.Client(api_key=api_key)

    def generate(
        self,
        system_text: str,
        user_text: str,
        *,
        temperature: Optional[float],
        max_output_tokens: int,
        timeout_s: int,
        provider_overrides: Optional[Dict[str, Any]] = None,
    ) -> LLMResponse:
        # google-genai uses GenerateContentConfig; keep minimal and stable
        cfg_kwargs: Dict[str, Any] = {
            "max_output_tokens": max_output_tokens,
            "system_instruction": system_text,
        }
        if temperature is not None:
            cfg_kwargs["temperature"] = temperature

        overrides = (provider_overrides or {}).get("google", {}) or {}
        if "thinking_config" in overrides:
            tc = overrides["thinking_config"]
            if isinstance(tc, dict):
                cfg_kwargs["thinking_config"] = types.ThinkingConfig(**tc)
            else:
                cfg_kwargs["thinking_config"] = tc

        config = types.GenerateContentConfig(**cfg_kwargs)

        # Some SDKs support timeout differently; we keep it at request level if available.
        # If not supported, the call will run and your OS/python can be used for outer timeout.
        resp = self.client.models.generate_content(
            model=self.model,
            contents=user_text,
            config=config,
        )
        # Pull text
        text = resp.text or ""
        return LLMResponse(
            text=text,
            raw_response=_safe_model_dump(resp),
            meta={
                "provider": "google",
                "model": self.model,
                "max_output_tokens": max_output_tokens,
            }
        )


def build_llm_client(spec: PromptSpec) -> LLMClient:
    provider = spec.config.provider
    model = spec.config.model

    load_dotenv()

    if provider == "openai":
        key = os.getenv("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing OPENAI_API_KEY in environment/.env")
        return OpenAIAdapter(model=model, api_key=key)

    if provider == "google":
        key = os.getenv("GEMINI_API_KEY", "").strip()
        if not key:
            raise RuntimeError("Missing GEMINI_API_KEY in environment/.env")
        return GeminiAdapter(model=model, api_key=key)

    raise ValueError(f"Unsupported provider: {provider}")


# ============================
# Semantic JSON validation
# ============================

ALLOWED_WORKLOADS = {
    "W5.2", "W5.3", "W5.4", "W5.5", "W5.6", "W5.7",
    "W7.2", "W7.3", "W7.4", "W7.5"
}
ALLOWED_ASKR = {"A", "S", "K", "R"}
ALLOWED_CONF = {"High", "Medium", "Low"}

def fallback_semantic() -> Dict[str, Any]:
    return {
        "extractions": []
    }

def parse_semantic_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(text)
        return obj, None
    except Exception as e:
        return None, str(e)

def validate_semantic_schema(obj: Dict[str, Any]) -> Tuple[bool, str]:
    # Must contain extractions array
    if "extractions" not in obj:
        return False, "Missing key: extractions"

    extractions = obj.get("extractions")
    if not isinstance(extractions, list):
        return False, "extractions must be an array"

    for ex in extractions:
        if not isinstance(ex, dict):
            return False, "each extraction must be an object"
        required_ex_keys = {"workload_type", "workload_summary", "workload_quote", "related_askr"}
        if not required_ex_keys.issubset(set(ex.keys())):
            return False, f"extraction missing keys: {required_ex_keys - set(ex.keys())}"
        wt = ex.get("workload_type")
        if wt is not None and wt not in ALLOWED_WORKLOADS:
            return False, f"Invalid workload_type: {wt}"

        askr = ex.get("related_askr")
        if not isinstance(askr, list):
            return False, "related_askr must be an array"
        for item in askr:
            if not isinstance(item, dict):
                return False, "related_askr items must be objects"
            for k in ["askr_type", "askr_summary", "verbatim_quote", "gap_signal"]:
                if k not in item:
                    return False, f"related_askr item missing key: {k}"
            if item["askr_type"] not in ALLOWED_ASKR:
                return False, f"Invalid askr_type: {item['askr_type']}"
            if not isinstance(item["gap_signal"], bool):
                return False, f"Invalid gap_signal (must be boolean): {item['gap_signal']}"

    return True, "ok"


# ============================
# JSON repair + parse diagnostics
# ============================

_CODE_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*([\s\S]*?)\s*```\s*$", re.IGNORECASE)

def _strip_code_fences(text: str) -> Tuple[str, bool]:
    m = _CODE_FENCE_RE.match(text or "")
    if not m:
        return text, False
    return m.group(1), True

def _normalize_smart_quotes(text: str) -> Tuple[str, bool]:
    # Smart quotes are valid Unicode inside JSON strings.
    # Replacing them with raw " can break JSON (unescaped quotes).
    # Keep them as-is to avoid invalidating otherwise valid JSON.
    return text, False

def _trim_to_json_object(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    first = text.find("{")
    last = text.rfind("}")
    if first != -1 and last != -1 and last > first:
        trimmed = text[first:last + 1]
        return trimmed, trimmed != text
    return text, False

def _remove_trailing_commas(text: str) -> Tuple[str, bool]:
    if not text:
        return text, False
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    return cleaned, cleaned != text

def repair_json_text(raw_text: str) -> Tuple[str, Dict[str, Any]]:
    steps: List[str] = []
    text = raw_text or ""

    text, changed = _strip_code_fences(text)
    if changed:
        steps.append("strip_code_fences")

    text, changed = _trim_to_json_object(text)
    if changed:
        steps.append("trim_to_json_object")

    # Intentionally skip smart-quote normalization (see _normalize_smart_quotes).

    text, changed = _remove_trailing_commas(text)
    if changed:
        steps.append("remove_trailing_commas")

    return text, {
        "applied": len(steps) > 0,
        "steps": steps,
        "raw_len": len(raw_text or ""),
        "repaired_len": len(text or ""),
    }

def _count_unescaped_quotes(text: str) -> int:
    count = 0
    escaped = False
    for ch in text:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == '"':
            count += 1
    return count

def is_probably_truncated(text: str) -> bool:
    if not text or not text.strip():
        return False
    t = text.strip()
    if "{" in t and t.count("{") > t.count("}"):
        return True
    if "[" in t and t.count("[") > t.count("]"):
        return True
    if _count_unescaped_quotes(t) % 2 == 1:
        return True
    if t.endswith("\\"):
        return True
    return False

def classify_parse_failure(raw_text: str, repaired_text: str, parse_err: Optional[str]) -> str:
    if not raw_text or not raw_text.strip():
        return "empty_response"
    if is_probably_truncated(raw_text) or is_probably_truncated(repaired_text):
        return "truncated_response"
    if parse_err:
        return "invalid_json"
    return "parse_failed"


# ============================
# Tracing / logging
# ============================

def trace_paths(base_logs_dir: str, paper_key: str) -> Dict[str, str]:
    base = os.path.join(base_logs_dir, paper_key)
    return {
        "base": base,
        "requests": os.path.join(base, "requests"),
        "responses": os.path.join(base, "responses"),
        "errors": os.path.join(base, "errors"),
    }

def _attempted_fname(unit_id: str, attempt: int, suffix: str) -> str:
    safe_unit = safe_filename_component(unit_id)
    return f"{safe_unit}__{sha1_short(unit_id)}__attempt{attempt}_{suffix}"

def save_trace_request(paths: Dict[str, str], unit_id: str, attempt: int, content: str) -> None:
    safe_mkdir(paths["requests"])
    fname = _attempted_fname(unit_id, attempt, "request.txt")
    write_text(os.path.join(paths["requests"], fname), content)

def save_trace_response(paths: Dict[str, str], unit_id: str, attempt: int, content: str) -> None:
    safe_mkdir(paths["responses"])
    fname = _attempted_fname(unit_id, attempt, "response.json")
    write_text(os.path.join(paths["responses"], fname), content)

def save_trace_error(paths: Dict[str, str], unit_id: str, attempt: int, content: str) -> None:
    safe_mkdir(paths["errors"])
    fname = _attempted_fname(unit_id, attempt, "error.txt")
    write_text(os.path.join(paths["errors"], fname), content)


# ============================
# Paper processing
# ============================

def process_one_unit(
    *,
    llm: LLMClient,
    prompt: PromptSpec,
    paper_key: str,
    section: Dict[str, Any],
    unit: Dict[str, Any],
    log_paths: Dict[str, str],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns:
      - record (final wrapped output record)
      - unit_summary_delta (counts for paper summary)
    """
    section_heading = normalize_section_heading(section.get("section_heading"))
    raw_text = (unit.get("raw_text") or "").strip()

    # skip empty
    if not raw_text:
        record = None
        return {}, {"skipped_empty": 1}

    # Build LLM input
    user_text = render_template(prompt.user_command, {
        "section_heading": section_heading,
        "raw_text": raw_text,
    })
    # Reduce failure risk by capping ASK+R items per paragraph.
    user_text = user_text + f"\n\nIMPORTANT: Return at most {prompt.config.max_askr_items} items in the askr array."
    system_text = prompt.system_command

    # For trace: store rendered request payload (system + user)
    request_dump = f"=== SYSTEM ===\n{system_text}\n\n=== USER ===\n{user_text}\n"

    # LLM call with retry on invalid JSON
    attempts = 0
    invalid_json_events = 0
    last_raw = ""
    last_repaired = ""
    parsed_obj: Optional[Dict[str, Any]] = None
    status = "ok"
    response_valid_json = False
    last_response_meta: Dict[str, Any] = {}
    askr_trimmed_meta: Optional[List[Dict[str, Any]]] = None

    while True:
        attempts += 1

        if prompt.config.trace.save_requests:
            save_trace_request(log_paths, unit["unit_id"], attempts, request_dump)

        try:
            provider_overrides = prompt.config.provider_overrides or {}
            # Ensure max_askr_items is available for providers that support schema enforcement.
            provider_overrides = {
                **provider_overrides,
                "openai": {
                    **(provider_overrides.get("openai", {}) or {}),
                    "max_askr_items": prompt.config.max_askr_items,
                }
            }
            resp = llm.generate(
                system_text=system_text,
                user_text=user_text,
                temperature=prompt.config.temperature if llm.provider == "google" else prompt.config.temperature,
                max_output_tokens=prompt.config.max_output_tokens,
                timeout_s=prompt.config.timeout_s,
                provider_overrides=provider_overrides,
            )
            last_raw = resp.text or ""
            last_response_meta = resp.meta or {}
        except Exception as e:
            status = "api_error"
            if prompt.config.trace.save_errors:
                save_trace_error(log_paths, unit["unit_id"], attempts, f"API ERROR: {e}\n\nREQUEST:\n{request_dump}")
            # Return fallback semantic
            parsed_obj = fallback_semantic()
            response_valid_json = False
            break

        # Repair before parsing
        last_repaired, repair_info = repair_json_text(last_raw)

        if prompt.config.trace.save_responses:
            raw_empty = len((last_raw or "").strip()) == 0
            incomplete_details_safe = _safe_model_dump(last_response_meta.get("incomplete_details"))
            if incomplete_details_safe is None and last_response_meta.get("incomplete_details") is not None:
                incomplete_details_safe = str(last_response_meta.get("incomplete_details"))
            response_log = {
                "attempt": attempts,
                "provider": llm.provider,
                "model": prompt.config.model,
                "max_output_tokens": prompt.config.max_output_tokens,
                "text_len": len(last_raw or ""),
                "text_empty": raw_empty,
                "repaired_len": len(last_repaired or ""),
                "repair_applied": repair_info.get("applied"),
                "repair_steps": repair_info.get("steps"),
                "response_status": last_response_meta.get("response_status"),
                "incomplete_details": incomplete_details_safe,
                "finish_reason": last_response_meta.get("finish_reason"),
                "structured_output": last_response_meta.get("structured_output"),
                "raw_text": last_raw,
                "repaired_text": last_repaired,
            }
            save_trace_response(log_paths, unit["unit_id"], attempts, _safe_json_dumps(response_log))
        obj, parse_err = parse_semantic_json(last_repaired)
        if obj is None:
            invalid_json_events += 1
            if attempts <= prompt.config.retry.max_retries and prompt.config.retry.retry_on_invalid_json:
                # Append strict reminder
                user_text = user_text + "\n\nIMPORTANT: Return JSON only. No extra text. Ensure valid JSON."
                continue
            status = classify_parse_failure(last_raw, last_repaired, parse_err)
            parsed_obj = fallback_semantic()
            response_valid_json = False
            if prompt.config.trace.save_errors:
                raw_excerpt_head = (last_raw or "")[:300]
                raw_excerpt_tail = (last_raw or "")[-300:]
                rep_excerpt_head = (last_repaired or "")[:300]
                rep_excerpt_tail = (last_repaired or "")[-300:]
                save_trace_error(
                    log_paths,
                    unit["unit_id"],
                    attempts,
                    (
                        f"PARSE FAILURE after {attempts} attempts.\n"
                        f"Status: {status}\n"
                        f"Parse error: {parse_err}\n"
                        f"Raw len: {len(last_raw or '')} | Repaired len: {len(last_repaired or '')}\n"
                        f"Repair applied: {repair_info.get('applied')} | Steps: {repair_info.get('steps')}\n"
                        f"RAW HEAD:\n{raw_excerpt_head}\n\nRAW TAIL:\n{raw_excerpt_tail}\n\n"
                        f"REPAIRED HEAD:\n{rep_excerpt_head}\n\nREPAIRED TAIL:\n{rep_excerpt_tail}\n\n"
                        f"RAW FULL:\n{last_raw}\n\nREPAIRED FULL:\n{last_repaired}\n\nREQUEST:\n{request_dump}\n"
                        f"RAW_RESPONSE_OBJ:\n{_safe_json_dumps(resp.raw_response) if resp.raw_response else 'null'}"
                    )
                )
            break

        ok, why = validate_semantic_schema(obj)
        if not ok:
            invalid_json_events += 1
            if attempts <= prompt.config.retry.max_retries and prompt.config.retry.retry_on_invalid_json:
                user_text = user_text + "\n\nIMPORTANT: Output must match the required JSON schema exactly."
                continue
            status = "invalid_schema"
            parsed_obj = fallback_semantic()
            response_valid_json = False
            if prompt.config.trace.save_errors:
                save_trace_error(
                    log_paths,
                    unit["unit_id"],
                    attempts,
                    f"INVALID SCHEMA after {attempts} attempts.\nReason: {why}\n\nRAW:\n{last_raw}\n\nREQUEST:\n{request_dump}\nRAW_RESPONSE_OBJ:\n{_safe_json_dumps(resp.raw_response) if resp.raw_response else 'null'}"
                )
            break

        # Success (optionally trim related_askr per extraction if over cap)
        parsed_obj = obj
        trim_events: List[Dict[str, Any]] = []
        extractions = parsed_obj.get("extractions", []) if isinstance(parsed_obj, dict) else []
        if isinstance(extractions, list):
            for idx, ex in enumerate(extractions):
                if not isinstance(ex, dict):
                    continue
                related = ex.get("related_askr", [])
                if isinstance(related, list) and len(related) > prompt.config.max_askr_items:
                    ex["related_askr"] = related[:prompt.config.max_askr_items]
                    trim_events.append({
                        "extraction_index": idx,
                        "original_count": len(related),
                        "kept_count": prompt.config.max_askr_items,
                    })
        if trim_events:
            askr_trimmed_meta = trim_events
            if prompt.config.trace.save_errors:
                save_trace_error(
                    log_paths,
                    unit["unit_id"],
                    attempts,
                    f"ASKR TRIMMED: {json.dumps(trim_events, ensure_ascii=False)}"
                )
        response_valid_json = True
        break

    # Wrap deterministic metadata
    record = {
        "section": {
            "section_id": section.get("section_id"),
            "section_heading": section.get("section_heading"),
            "heading_level": section.get("heading_level"),
            "order_in_document": section.get("order_in_document"),
        },
        "unit": {
            "unit_id": unit.get("unit_id"),
            "unit_type": unit.get("unit_type"),
            "order_in_section": unit.get("order_in_section"),
            "raw_text": unit.get("raw_text"),
        },
        "extraction": {
            "extractions": parsed_obj["extractions"],
        },
        "run_meta": {
            "status": status,
            "attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "response_valid_json": response_valid_json,
            "timestamp_utc": now_utc_iso(),
            "askr_trimmed": askr_trimmed_meta,
        }
    }

    # summary deltas
    extractions = parsed_obj.get("extractions", []) if parsed_obj else []
    if not isinstance(extractions, list):
        extractions = []
    askr_items = 0
    askr_empty = 0
    workload_null = 0
    for ex in extractions:
        if isinstance(ex, dict):
            if ex.get("workload_type") is None:
                workload_null += 1
            related = ex.get("related_askr", [])
            if isinstance(related, list):
                askr_items += len(related)
                if len(related) == 0:
                    askr_empty += 1

    return record, {
        "processed": 1,
        "invalid_json_events": invalid_json_events,
        "failed": 1 if status in {"api_error", "invalid_json", "invalid_schema", "empty_response", "truncated_response", "parse_failed"} else 0,
        "askr_items_total": askr_items,
        "askr_empty": askr_empty,
        "workload_primary_null": workload_null,
    }


def process_one_paper(
    *,
    llm: LLMClient,
    prompt: PromptSpec,
    paper_path: str,
    output_dir: str,
    logs_dir: str,
) -> None:
    paper_json = read_json(paper_path)

    paper_key = paper_json.get("paper_key")
    if not paper_key:
        raise ValueError(f"Missing paper_key in {paper_path}")

    log_paths = trace_paths(logs_dir, paper_key)
    safe_mkdir(log_paths["base"])

    # Prepare paper output container (LOCKED schema)
    sections = paper_json.get("sections", []) or []
    input_units_count = 0
    for s in sections:
        input_units_count += len((s.get("units") or []))

    output_obj = {
        "schema_version": "askr_extraction_output_v0.1",
        "prompt_type": prompt.prompt_type,
        "prompt_revision": prompt.revision,

        "run_config": {
            "provider": prompt.config.provider,
            "model": prompt.config.model,
            "temperature": prompt.config.temperature,
            "max_output_tokens": prompt.config.max_output_tokens,
            "response_format": prompt.config.response_format,
            "timeout_s": prompt.config.timeout_s,
            "max_askr_items": prompt.config.max_askr_items,
            "retry": {
                "max_retries": prompt.config.retry.max_retries,
                "retry_on_invalid_json": prompt.config.retry.retry_on_invalid_json,
            },
            "trace": {
                "save_requests": prompt.config.trace.save_requests,
                "save_responses": prompt.config.trace.save_responses,
                "save_errors": prompt.config.trace.save_errors,
            },
            "provider_overrides": {
                "google": prompt.config.provider_overrides.get("google", {}),
                "openai": prompt.config.provider_overrides.get("openai", {}),
            }
        },

        "paper": {
            "paper_key": paper_json.get("paper_key"),
            "source_filename": paper_json.get("source_filename"),
            "paper_title": paper_json.get("paper_title"),
            "paper_authors": paper_json.get("paper_authors"),
            "input_sections_count": len(sections),
            "input_units_count": input_units_count,
        },

        "records": [],

        "summary": {
            "processed_units": 0,
            "skipped_units": {
                "non_paragraph": 0,
                "empty_text": 0,
            },
            "extraction_counts": {
                "workload_primary_null": 0,
                "askr_empty": 0,
                "askr_items_total": 0
            },
            "invalid_json_events": 0,
            "failed_units": 0
        }
    }

    # Preserve order
    sections_sorted = sorted(sections, key=lambda s: sort_key_int(s, "order_in_document"))
    for section in sections_sorted:
        units = section.get("units", []) or []
        units_sorted = sorted(units, key=lambda u: sort_key_int(u, "order_in_section"))

        for unit in units_sorted:
            unit_type = unit.get("unit_type")
            if unit_type != "paragraph":
                output_obj["summary"]["skipped_units"]["non_paragraph"] += 1
                continue

            raw_text = (unit.get("raw_text") or "").strip()
            if not raw_text:
                output_obj["summary"]["skipped_units"]["empty_text"] += 1
                continue

            record, delta = process_one_unit(
                llm=llm,
                prompt=prompt,
                paper_key=paper_key,
                section=section,
                unit=unit,
                log_paths=log_paths,
            )

            if record:
                output_obj["records"].append(record)

            output_obj["summary"]["processed_units"] += delta.get("processed", 0)
            output_obj["summary"]["invalid_json_events"] += delta.get("invalid_json_events", 0)
            output_obj["summary"]["failed_units"] += delta.get("failed", 0)
            output_obj["summary"]["extraction_counts"]["askr_items_total"] += delta.get("askr_items_total", 0)
            output_obj["summary"]["extraction_counts"]["askr_empty"] += delta.get("askr_empty", 0)
            output_obj["summary"]["extraction_counts"]["workload_primary_null"] += delta.get("workload_primary_null", 0)

    # Save per-paper output
    out_path = os.path.join(output_dir, f"{paper_key}.askr.json")
    write_json(out_path, output_obj)


# ============================
# CLI (interactive prompts)
# ============================

def prompt_input(msg: str) -> str:
    try:
        return input(msg).strip()
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(1)

def main():
    print("ASK+R Prompt Runner (paper JSON -> per-unit extraction -> per-paper output)")

    input_dir = prompt_input("Please enter the input directory: ")
    output_dir = prompt_input("Please enter the output directory: ")
    prompt_path = prompt_input("Please enter the prompt file path: ")
    logs_dir = os.path.join(output_dir, "_logs")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Input directory not found: {input_dir}")
    safe_mkdir(output_dir)
    safe_mkdir(logs_dir)

    prompt_spec = load_prompt_yaml(prompt_path)
    llm = build_llm_client(prompt_spec)

    # Discover paper json files
    paper_files = sorted(glob.glob(os.path.join(input_dir, "*.json")))
    if not paper_files:
        raise ValueError(f"No .json files found in input directory: {input_dir}")

    print(f"Provider: {prompt_spec.config.provider} | Model: {prompt_spec.config.model}")
    print(f"Found {len(paper_files)} paper JSON files.")
    print(f"Outputs -> {output_dir}")
    print(f"Logs    -> {logs_dir}")
    print("Starting...\n")

    for idx, paper_path in enumerate(paper_files, start=1):
        try:
            paper_json = read_json(paper_path)
            paper_key = paper_json.get("paper_key", os.path.basename(paper_path))
            print(f"[{idx}/{len(paper_files)}] Processing: {paper_key}")

            # Skip already-processed papers if output exists.
            out_path = os.path.join(output_dir, f"{paper_key}.askr.json")
            if os.path.isfile(out_path):
                print(f"  SKIP (already processed): {os.path.basename(out_path)}")
                continue

            process_one_paper(
                llm=llm,
                prompt=prompt_spec,
                paper_path=paper_path,
                output_dir=output_dir,
                logs_dir=logs_dir
            )

        except Exception as e:
            # Paper-level failure: log and continue
            safe_mkdir(logs_dir)
            err_path = os.path.join(logs_dir, "_paper_failures.txt")
            with open(err_path, "a", encoding="utf-8") as f:
                f.write(f"{now_utc_iso()} | {paper_path} | {e}\n")
            print(f"  ERROR (paper skipped): {e}")

    print("\nDone.")


if __name__ == "__main__":
    main()
