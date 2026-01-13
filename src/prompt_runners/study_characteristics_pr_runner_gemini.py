"""
Robust Gemini PDF Extraction Runner (Thesis-ready)

Key features:
- Loads YAML prompt with system_command + user_command
- Uploads PDF and calls Gemini with JSON response
- Strict JSON parsing + schema validation
- Normalizes evidence:
    - Bibliography evidence -> location only
    - Study characteristics evidence -> quote + location (when present)
- Flattens to stable CSV columns with consistent headers
- Handles variable-length participant activities by (a) keeping JSON column and (b) optional fixed slots
- Resume-safe: skips already-processed files
- Retries with exponential backoff on 429 / transient errors
- Writes one row per PDF, appends safely

Requirements:
pip install google-genai python-dotenv pyyaml
"""

import os
import re
import json
import csv
import time
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types


# -----------------------------
# Utilities
# -----------------------------
def safe_json_loads(text: str) -> Dict[str, Any]:
    """
    More robust than json.loads(response.text) because models sometimes
    return extra whitespace or accidental wrappers.
    Strategy:
    - Try direct loads
    - If fails, extract first {...} JSON object block and try again
    """
    try:
        return json.loads(text)
    except Exception:
        pass

    # Attempt to extract the first JSON object from the text
    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError("Model response is not valid JSON and no JSON object was found.")
    return json.loads(match.group(0))


def ensure_keys(obj: Dict[str, Any], required: List[str], context: str = "") -> None:
    missing = [k for k in required if k not in obj]
    if missing:
        raise ValueError(f"Missing required key(s) {missing} in {context or 'JSON'}.")


def as_list(x: Any) -> Optional[List[Any]]:
    if x is None:
        return None
    return x if isinstance(x, list) else [x]


def jdump(x: Any) -> str:
    """Compact JSON string for storing lists/dicts safely in CSV."""
    return json.dumps(x, ensure_ascii=False, separators=(",", ":"))


# -----------------------------
# Schema-aware normalization
# -----------------------------
BIB_FIELDS = ["paper_title", "authors", "publication_date", "author_affiliations"]

STUDY_FIELDS = [
    "objective_of_paper",
    "designed_solution_or_intervention",
    "domain_of_solution",
    "design_team_composition",
    "main_users_or_target_group",
    "participants_in_design_process",
    "number_of_participants_in_design_process",
    "design_methodology",
    "specific_HCD_UCD_framework_version",
    "design_process_steps_or_phases",
    "other_stakeholders_involved_besides_users",
]


def normalize_field_node(node: Any, evidence_mode: str) -> Dict[str, Any]:
    """
    Normalizes a field node that should look like:
      { "value": ..., "evidence": ... } or null (but prompt enforces object)
    evidence_mode:
      - "location_only": evidence expected to be {"location": "..."} or null
      - "quote_and_location": evidence expected to be {"quote": "...", "location": "..."} or null
    """
    if not isinstance(node, dict) or "value" not in node or "evidence" not in node:
        # If model violated schema, coerce to null-safe structure
        return {"value": None, "evidence_location": None, "evidence_quote": None, "raw_evidence": None}

    value = node.get("value", None)
    evidence = node.get("evidence", None)

    # Evidence may be null
    if evidence is None:
        return {"value": value, "evidence_location": None, "evidence_quote": None, "raw_evidence": None}

    # Evidence should be dict
    if not isinstance(evidence, dict):
        # Preserve as raw evidence for debugging
        return {"value": value, "evidence_location": None, "evidence_quote": None, "raw_evidence": jdump(evidence)}

    loc = evidence.get("location", None)

    if evidence_mode == "location_only":
        return {"value": value, "evidence_location": loc, "evidence_quote": None, "raw_evidence": None}

    # quote_and_location
    quote = evidence.get("quote", None)
    return {"value": value, "evidence_location": loc, "evidence_quote": quote, "raw_evidence": None}


def normalize_response_schema(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validates and normalizes the model output to a stable internal representation.
    Produces a dict with:
      bibliography.<field>.value
      bibliography.<field>.evidence_location
      study_characteristics.<field>.value
      study_characteristics.<field>.evidence_location
      study_characteristics.<field>.evidence_quote
    """
    ensure_keys(raw, ["bibliography", "study_characteristics"], context="top-level JSON")

    bib = raw["bibliography"]
    study = raw["study_characteristics"]

    if not isinstance(bib, dict) or not isinstance(study, dict):
        raise ValueError("bibliography and study_characteristics must be JSON objects.")

    normalized: Dict[str, Any] = {"bibliography": {}, "study_characteristics": {}}

    # Bibliography: location only
    for f in BIB_FIELDS:
        node = bib.get(f, {"value": None, "evidence": None})
        normalized["bibliography"][f] = normalize_field_node(node, evidence_mode="location_only")

    # Study: quote + location
    for f in STUDY_FIELDS:
        node = study.get(f, {"value": None, "evidence": None})
        normalized["study_characteristics"][f] = normalize_field_node(node, evidence_mode="quote_and_location")

    return normalized


# -----------------------------
# Flattening (CSV-stable)
# -----------------------------
def flatten_normalized(normalized: Dict[str, Any], max_activity_slots: int = 10) -> Dict[str, Any]:
    """
    Flattens normalized dict into stable CSV columns.

    Special handling:
    - number_of_participants_in_design_process:
        - keep raw JSON in one column: ..._value_json
        - additionally expand into fixed slots activity_i, participants_i (up to max_activity_slots)
    """
    out: Dict[str, Any] = {}

    # Bibliography fields
    for f in BIB_FIELDS:
        node = normalized["bibliography"][f]
        out[f"bibliography_{f}_value"] = jdump(node["value"]) if isinstance(node["value"], (list, dict)) else node["value"]
        out[f"bibliography_{f}_evidence_location"] = node["evidence_location"]
        # No quote for bibliographic fields
        out[f"bibliography_{f}_evidence_quote"] = None
        out[f"bibliography_{f}_raw_evidence"] = node["raw_evidence"]

    # Study fields
    for f in STUDY_FIELDS:
        node = normalized["study_characteristics"][f]

        # Special: activity list
        if f == "number_of_participants_in_design_process":
            val = node["value"]
            out[f"study_characteristics_{f}_value_json"] = None if val is None else jdump(val)

            # Expand fixed slots if list-of-objects
            for i in range(1, max_activity_slots + 1):
                out[f"study_characteristics_{f}_activity_{i}"] = None
                out[f"study_characteristics_{f}_participants_{i}"] = None

            if isinstance(val, list):
                for idx, entry in enumerate(val[:max_activity_slots], start=1):
                    if isinstance(entry, dict):
                        out[f"study_characteristics_{f}_activity_{idx}"] = entry.get("activity")
                        out[f"study_characteristics_{f}_participants_{idx}"] = entry.get("participants")
            else:
                # If model returns "<n> (total)" or number etc.
                out[f"study_characteristics_{f}_value"] = val
        else:
            out[f"study_characteristics_{f}_value"] = jdump(node["value"]) if isinstance(node["value"], (list, dict)) else node["value"]

        out[f"study_characteristics_{f}_evidence_location"] = node["evidence_location"]
        out[f"study_characteristics_{f}_evidence_quote"] = node["evidence_quote"]
        out[f"study_characteristics_{f}_raw_evidence"] = node["raw_evidence"]

    return out


# -----------------------------
# Runner
# -----------------------------
class GeminiExtractionRunner:
    def __init__(
        self,
        prompt_yaml_path: str,
        model: str = "gemini-2.0-flash-001",
        temperature: float = 0.0,
        response_mime_type: str = "application/json",
        max_retries: int = 6,
        base_backoff_sec: float = 5.0,
        max_activity_slots: int = 10,
    ):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment (.env).")

        self.client = genai.Client(api_key=api_key)
        self.model_id = model
        self.temperature = temperature
        self.response_mime_type = response_mime_type
        self.max_retries = max_retries
        self.base_backoff_sec = base_backoff_sec
        self.max_activity_slots = max_activity_slots

        with open(prompt_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        if not isinstance(cfg, dict) or "system_command" not in cfg or "user_command" not in cfg:
            raise ValueError("Prompt YAML must contain system_command and user_command keys.")

        self.system_command = cfg["system_command"]
        self.user_command = cfg["user_command"]

    def _already_processed(self, output_csv: Path) -> set:
        if not output_csv.exists():
            return set()
        with open(output_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            return {row.get("file_name") for row in reader if row.get("file_name")}

    def _call_gemini_with_retry(self, uploaded_file) -> Dict[str, Any]:
        last_err = None
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[self.user_command, uploaded_file],
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_command,
                        response_mime_type=self.response_mime_type,
                        temperature=self.temperature,
                    ),
                )
                return safe_json_loads(resp.text)
            except Exception as e:
                last_err = e
                s = str(e)
                is_rate = ("429" in s) or ("RESOURCE_EXHAUSTED" in s) or ("Rate limit" in s)
                is_transient = is_rate or ("503" in s) or ("500" in s) or ("timeout" in s.lower())

                if attempt == self.max_retries or not is_transient:
                    break

                # Exponential backoff + jitter
                sleep_s = min(120.0, self.base_backoff_sec * (2 ** (attempt - 1)))
                sleep_s = sleep_s * (0.8 + 0.4 * random.random())
                print(f"⏳ Transient error (attempt {attempt}/{self.max_retries}). Sleeping {sleep_s:.1f}s. Error: {e}")
                time.sleep(sleep_s)

        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries. Last error: {last_err}")

    def process_folder(self, input_pdf_folder: str, output_csv_path: str) -> None:
        input_dir = Path(input_pdf_folder)
        output_file = Path(output_csv_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        processed = self._already_processed(output_file)
        pdf_files = [p for p in sorted(input_dir.glob("*.pdf")) if p.name not in processed]

        print(f"📋 Found {len(pdf_files)} new PDFs. (Skipping {len(processed)} already done)")

        # Determine stable headers:
        # - If CSV exists, keep its headers
        existing_headers: Optional[List[str]] = None
        if output_file.exists():
            with open(output_file, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                existing_headers = next(reader, None)

        for pdf_path in pdf_files:
            print(f"🚀 Extracting: {pdf_path.name}")

            try:
                uploaded_file = self.client.files.upload(file=str(pdf_path))

                raw = self._call_gemini_with_retry(uploaded_file)
                normalized = normalize_response_schema(raw)
                row = flatten_normalized(normalized, max_activity_slots=self.max_activity_slots)

                # Add file_name & optional raw json for audit
                row["file_name"] = pdf_path.name
                row["raw_json"] = jdump(raw)

                # Freeze headers:
                if existing_headers is None:
                    # Create stable headers from first row
                    existing_headers = sorted(row.keys())

                # Fill missing columns with None
                for h in existing_headers:
                    if h not in row:
                        row[h] = None

                # If new columns appear later, DO NOT silently change headers.
                # Instead, store extras in raw_json and warn.
                extras = [k for k in row.keys() if k not in existing_headers]
                if extras:
                    print(f"⚠️ New unexpected columns ignored (kept in raw_json): {extras}")
                    for k in extras:
                        row.pop(k, None)

                file_exists = output_file.exists()
                with open(output_file, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=existing_headers)
                    if not file_exists:
                        writer.writeheader()
                    writer.writerow(row)

                print(f"✅ Saved: {pdf_path.name}")

                # Conservative pacing (adjust as needed)
                time.sleep(12)

            except Exception as e:
                print(f"❌ Failed: {pdf_path.name}: {e}")


if __name__ == "__main__":
    INPUT_PDF_FOLDER = r"data\test_data\gemini\test_ch_input"
    OUTPUT_CSV_FILE = r"data\test_data\gemini\study_characteristics_extraction_result\study_characteristics_extraction_results.csv"
    PROMPT_YAML_FILE = r"prompt\study_characteristics_full_text_pr.yaml"

    runner = GeminiExtractionRunner(
        prompt_yaml_path=PROMPT_YAML_FILE,
        model="gemini-2.0-flash-001",
        temperature=0.0,
        max_retries=6,
        base_backoff_sec=5.0,
        max_activity_slots=10,
    )
    runner.process_folder(INPUT_PDF_FOLDER, OUTPUT_CSV_FILE)
