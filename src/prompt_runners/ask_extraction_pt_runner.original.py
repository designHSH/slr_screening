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
    ) -> str:
        raise NotImplementedError


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
    ) -> str:
        # Note: GPT-5 does not support temperature.
        kwargs: Dict[str, Any] = {}
        is_gpt5 = self.model.startswith("gpt-5")
        if temperature is not None and not is_gpt5:
            kwargs["temperature"] = temperature
        overrides = (provider_overrides or {}).get("openai", {}) or {}
        if "reasoning" in overrides:
            kwargs["reasoning"] = overrides["reasoning"]

        # Prefer JSON-only behavior by prompt; if you later use strict response formats,
        # you can add response_format mapping here.
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
        # responses API returns structured output; pull text:
        return resp.output_text


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
    ) -> str:
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
        return resp.text or ""


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
        "workload": {
            "primary_workload": None,
            "secondary_workloads": [],
            "workload_summary": None,
            "workload_quote": None
        },
        "askr": []
    }

def parse_semantic_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    try:
        obj = json.loads(text)
        return obj, None
    except Exception as e:
        return None, str(e)

def validate_semantic_schema(obj: Dict[str, Any]) -> Tuple[bool, str]:
    # Must contain workload and askr
    if "workload" not in obj or "askr" not in obj:
        return False, "Missing keys: workload/askr"

    wl = obj.get("workload")
    if not isinstance(wl, dict):
        return False, "workload must be an object"

    required_wl_keys = {"primary_workload", "secondary_workloads", "workload_summary", "workload_quote"}
    if not required_wl_keys.issubset(set(wl.keys())):
        return False, f"workload missing keys: {required_wl_keys - set(wl.keys())}"

    # primary_workload can be null or allowed code
    pw = wl.get("primary_workload")
    if pw is not None and pw not in ALLOWED_WORKLOADS:
        return False, f"Invalid primary_workload: {pw}"

    sw = wl.get("secondary_workloads")
    if not isinstance(sw, list):
        return False, "secondary_workloads must be an array"
    for x in sw:
        if x not in ALLOWED_WORKLOADS:
            return False, f"Invalid secondary_workload: {x}"

    askr = obj.get("askr")
    if not isinstance(askr, list):
        return False, "askr must be an array"

    for item in askr:
        if not isinstance(item, dict):
            return False, "askr items must be objects"
        for k in ["askr_type", "linked_workload", "askr_summary", "verbatim_quote", "gap_signal", "short_reasoning", "confidence_level"]:
            if k not in item:
                return False, f"askr item missing key: {k}"
        if item["askr_type"] not in ALLOWED_ASKR:
            return False, f"Invalid askr_type: {item['askr_type']}"
        if item["linked_workload"] not in ALLOWED_WORKLOADS:
            return False, f"Invalid linked_workload: {item['linked_workload']}"
        if not isinstance(item["gap_signal"], bool):
            return False, f"Invalid gap_signal (must be boolean): {item['gap_signal']}"
        if item["confidence_level"] not in ALLOWED_CONF:
            return False, f"Invalid confidence_level: {item['confidence_level']}"

    return True, "ok"


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

def save_trace_request(paths: Dict[str, str], unit_id: str, content: str) -> None:
    safe_mkdir(paths["requests"])
    safe_unit = safe_filename_component(unit_id)
    fname = f"{safe_unit}__{sha1_short(unit_id)}.txt"
    write_text(os.path.join(paths["requests"], fname), content)

def save_trace_response(paths: Dict[str, str], unit_id: str, content: str) -> None:
    safe_mkdir(paths["responses"])
    safe_unit = safe_filename_component(unit_id)
    fname = f"{safe_unit}__{sha1_short(unit_id)}.txt"
    write_text(os.path.join(paths["responses"], fname), content)

def save_trace_error(paths: Dict[str, str], unit_id: str, content: str) -> None:
    safe_mkdir(paths["errors"])
    safe_unit = safe_filename_component(unit_id)
    fname = f"{safe_unit}__{sha1_short(unit_id)}.txt"
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
    system_text = prompt.system_command

    # For trace: store rendered request payload (system + user)
    request_dump = f"=== SYSTEM ===\n{system_text}\n\n=== USER ===\n{user_text}\n"

    # LLM call with retry on invalid JSON
    attempts = 0
    invalid_json_events = 0
    last_raw = ""
    parsed_obj: Optional[Dict[str, Any]] = None
    status = "ok"
    response_valid_json = False

    while True:
        attempts += 1

        if prompt.config.trace.save_requests:
            save_trace_request(log_paths, unit["unit_id"], request_dump)

        try:
            last_raw = llm.generate(
                system_text=system_text,
                user_text=user_text,
                temperature=prompt.config.temperature if llm.provider == "google" else prompt.config.temperature,
                max_output_tokens=prompt.config.max_output_tokens,
                timeout_s=prompt.config.timeout_s,
                provider_overrides=prompt.config.provider_overrides,
            )
        except Exception as e:
            status = "api_error"
            if prompt.config.trace.save_errors:
                save_trace_error(log_paths, unit["unit_id"], f"API ERROR: {e}\n\nREQUEST:\n{request_dump}")
            # Return fallback semantic
            parsed_obj = fallback_semantic()
            response_valid_json = True
            break

        if prompt.config.trace.save_responses:
            save_trace_response(log_paths, unit["unit_id"], last_raw)

        obj, parse_err = parse_semantic_json(last_raw)
        if obj is None:
            invalid_json_events += 1
            if attempts <= prompt.config.retry.max_retries and prompt.config.retry.retry_on_invalid_json:
                # Append strict reminder
                user_text = user_text + "\n\nIMPORTANT: Return JSON only. No extra text. Ensure valid JSON."
                continue
            status = "invalid_json"
            parsed_obj = fallback_semantic()
            response_valid_json = False
            if prompt.config.trace.save_errors:
                save_trace_error(
                    log_paths,
                    unit["unit_id"],
                    f"INVALID JSON after {attempts} attempts.\nParse error: {parse_err}\n\nRAW:\n{last_raw}\n\nREQUEST:\n{request_dump}"
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
                    f"INVALID SCHEMA after {attempts} attempts.\nReason: {why}\n\nRAW:\n{last_raw}\n\nREQUEST:\n{request_dump}"
                )
            break

        # Success
        parsed_obj = obj
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
            "workload": parsed_obj["workload"],
            "askr": parsed_obj["askr"],
        },
        "run_meta": {
            "status": status,
            "attempts": attempts,
            "retry_count": max(0, attempts - 1),
            "response_valid_json": response_valid_json,
            "timestamp_utc": now_utc_iso(),
        }
    }

    # summary deltas
    askr_items = len(parsed_obj.get("askr", [])) if parsed_obj else 0
    workload_null = 1 if parsed_obj and parsed_obj.get("workload", {}).get("primary_workload") is None else 0
    askr_empty = 1 if askr_items == 0 else 0

    return record, {
        "processed": 1,
        "invalid_json_events": invalid_json_events,
        "failed": 1 if status in {"api_error", "invalid_json", "invalid_schema"} else 0,
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
