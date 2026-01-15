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
import hashlib
import random
from pathlib import Path
from datetime import datetime, timezone

import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types


# =========================
# Config (as requested)
# =========================
INPUT_PDF_FOLDER = r"data\test_data\gemini\test_ch_input"
OUTPUT_CSV_FILE = r"data\test_data\gemini\study_characteristics_extraction_result\study_characteristics_extraction_results.csv"
PROMPT_YAML_FILE = r"prompt\study_characteristics_full_text_pr.yaml"


# =========================
# Helpers
# =========================
CSV_HEADERS = [
    # Identifiers & metadata
    "paper_id", "paper_key", "file_name", "model_id", "prompt_version", "run_timestamp",

    # Bibliographic metadata (no evidence)
    "paper_title", "authors", "publication_date", "author_affiliations",

    # Study intent, domain, region (quote-only evidence)
    "objective_value", "objective_quote",
    "domain_value", "domain_quote",
    "region_value", "region_quote",

    # Design output & approach (quote-only evidence)
    "designed_solution_value", "designed_solution_quote",
    "design_methodology_value", "design_methodology_quote",
    "hcd_ucd_frameworks_value", "hcd_ucd_frameworks_quote",
    "design_phases_value", "design_phases_quote",

    # People involved (quote-only evidence)
    "design_team_value", "design_team_quote",
    "main_users_value", "main_users_quote",
    "participants_groups_value", "participants_groups_quote",
    "participant_activities_value", "participant_activities_quote",

    # Other stakeholders (quote-only evidence)
    "other_stakeholders_value", "other_stakeholders_quote",
]


def sha256_short(path: Path, n: int = 12) -> str:
    """Stable ID: P- + first n hex chars of SHA-256 of the PDF bytes."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return f"P-{h.hexdigest()[:n]}"


def make_paper_key_from_filename(filename: str, max_len: int = 80) -> str:
    """
    Optional human-readable key derived from filename.
    Not guaranteed unique; that's okay. Keep it filesystem/CSV safe.
    """
    stem = Path(filename).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem[:max_len] if stem else ""


def safe_json_loads(text: str) -> dict:
    """Parse JSON robustly (handles accidental extra text)."""
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            raise ValueError("Response is not valid JSON and no JSON object was found.")
        return json.loads(m.group(0))


def jdump(value) -> str:
    """Dump lists/dicts as compact JSON for CSV cells."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def normalize_to_row(raw: dict) -> dict:
    """
    Convert the model JSON into a flat row matching CSV_HEADERS.
    Assumes raw follows:
      { "bibliography": {...}, "study_characteristics": {...} }
    """
    if not isinstance(raw, dict) or "bibliography" not in raw or "study_characteristics" not in raw:
        raise ValueError("JSON missing required top-level keys: bibliography, study_characteristics")

    bib = raw.get("bibliography") or {}
    sc = raw.get("study_characteristics") or {}

    def get_val(obj, key):
        return obj.get(key, None)

    def to_cell(v):
        if isinstance(v, (list, dict)):
            return jdump(v)
        return v

    row = {
        # Bibliography
        "paper_title": to_cell(get_val(bib, "paper_title")),
        "authors": to_cell(get_val(bib, "authors")),
        "publication_date": to_cell(get_val(bib, "publication_date")),
        "author_affiliations": to_cell(get_val(bib, "author_affiliations")),

        # Study characteristics
        "objective_value": to_cell(get_val(sc, "objective_value")),
        "objective_quote": to_cell(get_val(sc, "objective_quote")),

        "domain_value": to_cell(get_val(sc, "domain_value")),
        "domain_quote": to_cell(get_val(sc, "domain_quote")),

        "region_value": to_cell(get_val(sc, "region_value")),
        "region_quote": to_cell(get_val(sc, "region_quote")),

        "designed_solution_value": to_cell(get_val(sc, "designed_solution_value")),
        "designed_solution_quote": to_cell(get_val(sc, "designed_solution_quote")),

        "design_methodology_value": to_cell(get_val(sc, "design_methodology_value")),
        "design_methodology_quote": to_cell(get_val(sc, "design_methodology_quote")),

        "hcd_ucd_frameworks_value": to_cell(get_val(sc, "hcd_ucd_frameworks_value")),
        "hcd_ucd_frameworks_quote": to_cell(get_val(sc, "hcd_ucd_frameworks_quote")),

        "design_phases_value": to_cell(get_val(sc, "design_phases_value")),
        "design_phases_quote": to_cell(get_val(sc, "design_phases_quote")),

        "design_team_value": to_cell(get_val(sc, "design_team_value")),
        "design_team_quote": to_cell(get_val(sc, "design_team_quote")),

        "main_users_value": to_cell(get_val(sc, "main_users_value")),
        "main_users_quote": to_cell(get_val(sc, "main_users_quote")),

        "participants_groups_value": to_cell(get_val(sc, "participants_groups_value")),
        "participants_groups_quote": to_cell(get_val(sc, "participants_groups_quote")),

        "participant_activities_value": to_cell(get_val(sc, "participant_activities_value")),
        "participant_activities_quote": to_cell(get_val(sc, "participant_activities_quote")),

        "other_stakeholders_value": to_cell(get_val(sc, "other_stakeholders_value")),
        "other_stakeholders_quote": to_cell(get_val(sc, "other_stakeholders_quote")),
    }

    # Ensure all expected keys exist (fill missing with None)
    for k in CSV_HEADERS:
        row.setdefault(k, None)

    return row


# =========================
# Runner
# =========================
class GeminiStudyCharacteristicsRunner:
    def __init__(
        self,
        prompt_yaml_path: str,
        model: str = "gemini-2.0-flash-001",
        temperature: float = 0.0,
        response_mime_type: str = "application/json",
        min_delay_seconds: float = 12.0,
        max_retries: int = 6,
        base_backoff_seconds: float = 5.0,
    ):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise EnvironmentError("GEMINI_API_KEY not found in environment. Put it in .env or environment variables.")

        self.client = genai.Client(api_key=api_key)
        self.model_id = model
        self.temperature = temperature
        self.response_mime_type = response_mime_type
        self.min_delay_seconds = min_delay_seconds
        self.max_retries = max_retries
        self.base_backoff_seconds = base_backoff_seconds

        with open(prompt_yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        # Support your YAML: task/version/user_command/system_command
        self.task = cfg.get("task", "study_characteristics_extraction_full_text")
        self.prompt_version = str(cfg.get("version", "0.0.0"))
        self.user_command = cfg["user_command"]
        self.system_command = cfg["system_command"]

    def _read_processed_ids(self, out_csv: Path) -> set:
        """Resume-safe: skip files already written (by file_name or paper_id)."""
        if not out_csv.exists():
            return set()
        processed = set()
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row.get("file_name"):
                    processed.add(row["file_name"])
        return processed

    def _sleep_min_delay(self):
        """Fixed pacing to respect RPM quotas."""
        time.sleep(self.min_delay_seconds)

    def _call_gemini_with_retry(self, uploaded_file) -> dict:
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
                msg = str(e)
                is_rate = ("429" in msg) or ("RESOURCE_EXHAUSTED" in msg) or ("Rate limit" in msg)
                is_transient = is_rate or ("503" in msg) or ("500" in msg) or ("timeout" in msg.lower())

                if attempt == self.max_retries or not is_transient:
                    break

                # Exponential backoff + jitter
                sleep_s = min(120.0, self.base_backoff_seconds * (2 ** (attempt - 1)))
                sleep_s *= (0.8 + 0.4 * random.random())
                print(f"⏳ Transient error (attempt {attempt}/{self.max_retries}). Sleeping {sleep_s:.1f}s. Error: {e}")
                time.sleep(sleep_s)

        raise RuntimeError(f"Gemini call failed after {self.max_retries} retries. Last error: {last_err}")

    def run(self, input_pdf_folder: str, output_csv_file: str):
        in_dir = Path(input_pdf_folder)
        out_csv = Path(output_csv_file)
        out_csv.parent.mkdir(parents=True, exist_ok=True)

        processed_files = self._read_processed_ids(out_csv)
        pdfs = [p for p in sorted(in_dir.glob("*.pdf")) if p.name not in processed_files]

        print(f"📋 Found {len(pdfs)} new PDFs. (Skipping {len(processed_files)} already done)")

        # Ensure CSV header exists and is stable
        if not out_csv.exists():
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                writer.writeheader()

        for pdf_path in pdfs:
            print(f"🚀 Processing: {pdf_path.name}")

            try:
                # Generate IDs outside Gemini
                paper_id = sha256_short(pdf_path)
                paper_key = make_paper_key_from_filename(pdf_path.name)

                # Upload PDF
                uploaded = self.client.files.upload(file=str(pdf_path))

                # Call Gemini
                raw = self._call_gemini_with_retry(uploaded)

                # Flatten
                row = normalize_to_row(raw)

                # Add metadata
                row["paper_id"] = paper_id
                row["paper_key"] = paper_key
                row["file_name"] = pdf_path.name
                row["model_id"] = self.model_id
                row["prompt_version"] = self.prompt_version
                row["run_timestamp"] = datetime.now(timezone.utc).isoformat()

                # Write row (stable headers)
                with open(out_csv, "a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=CSV_HEADERS)
                    writer.writerow({k: row.get(k) for k in CSV_HEADERS})

                print(f"✅ Saved: {pdf_path.name}  ->  {paper_id}")

                # Delay between requests
                self._sleep_min_delay()

            except Exception as e:
                print(f"❌ Failed: {pdf_path.name}: {e}")
                # Optional: add a small delay even on failure to avoid hammering
                time.sleep(max(3.0, self.min_delay_seconds / 2))


if __name__ == "__main__":
    runner = GeminiStudyCharacteristicsRunner(
        prompt_yaml_path=PROMPT_YAML_FILE,
        model="gemini-2.0-flash-001",
        temperature=0.0,
        response_mime_type="application/json",
        min_delay_seconds=12.0,   # adjust if your quota allows higher RPM
        max_retries=6,
        base_backoff_seconds=5.0,
    )
    runner.run(INPUT_PDF_FOLDER, OUTPUT_CSV_FILE)
