#!/usr/bin/env python3
"""
Quote Extraction Prompt Runner (Step 2)
======================================

What it does
------------
- Loads a YAML prompt file (your Step-2 quote extraction prompt).
- Loops through .txt papers in an input folder.
- For each paper:
  - Extracts first-author surname + publication year from the filename (pattern: "<Author> et al. - <YEAR> - ... .txt" or "<Author> - <YEAR> - ... .txt").
  - Calls OpenAI (gpt-5-mini by default from YAML config) to extract quotes (JSON array).
  - Generates deterministic Quote IDs + quote anchors (script-side).
  - Saves ONE JSON file per paper to the output folder (same stem as .txt).

Output JSON (per paper)
-----------------------
A JSON array of quote records, each with:
  paper_id, quote_id, quote_anchor, section, section_code, quote, quote_with_context

Requirements
------------
pip install openai pyyaml

Env
---
export OPENAI_API_KEY="..."

Usage
-----
python quote_extraction_runner.py \
  --prompt-yaml prompts/hcd_barrier_quote_extraction.yaml \
  --input-dir  data/papers_txt \
  --output-dir output/quotes_step2 \
  --log-file   output/quotes_step2/run.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI


# -------------------------
# Filename parsing
# -------------------------

FILENAME_RE = re.compile(
    r"^\s*(?P<author_part>.+?)\s*-\s*(?P<year>\d{4})\s*-",
    re.IGNORECASE,
)

def slugify_keep_hyphen(s: str) -> str:
    """Lowercase; keep letters/digits/hyphen; convert whitespace/underscores to hyphen; drop other punctuation."""
    s = s.strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    s = re.sub(r"[^a-z0-9\-]+", "", s)
    s = re.sub(r"-{2,}", "-", s).strip("-")
    return s

def extract_first_author_and_year(filename: str) -> Tuple[str, str]:
    """
    Extract first author's surname (slugified) and publication year from filename.

    Expected patterns:
      "<Author stuff> et al. - 2024 - Some title.txt"
      "<Author stuff> - 2024 - Some title.txt"

    Rules:
    - author_part is everything before " - <YEAR> - "
    - if "et al." is present, trim it
    - if " and " is present, use only the first author chunk (before "and")
    - remove trailing initials (tokens that are 1-2 letters after stripping '.')
    - remaining tokens are treated as the surname (supports multiword surnames like 'van der Waals')
    - slugify to lowercase hyphenated string
    """
    m = FILENAME_RE.match(filename)
    if not m:
        raise ValueError(
            f"Filename does not match required pattern '<Author> - <YEAR> - ...': {filename}"
        )

    author_part = m.group("author_part").strip()
    author_part = re.sub(r"\s+et al\.\s*$", "", author_part, flags=re.IGNORECASE).strip()
    author_part = re.split(r"\s+and\s+", author_part, maxsplit=1, flags=re.IGNORECASE)[0].strip()
    year = m.group("year").strip()

    # Tokenize while preserving hyphenated last names.
    tokens = author_part.split()
    # Remove trailing initials like "J." "M." "JR." etc., but keep if it's the only token.
    while len(tokens) > 1:
        t = tokens[-1].replace(".", "")
        if t.isalpha() and (1 <= len(t) <= 2 or (len(t) <= 3 and t.isupper())):
            tokens.pop()
        else:
            break
    if not tokens:
        raise ValueError(f"Could not extract author surname from: {filename}")

    surname = "-".join(tokens)
    surname_slug = slugify_keep_hyphen(surname)
    if not surname_slug:
        raise ValueError(f"Could not slugify surname from: {surname} (filename: {filename})")

    return surname_slug, year

def compute_paper_id(txt_path: Path) -> str:
    """
    Stable-ish paper_id derived from filename stem + short hash to avoid collisions.
    """
    stem = txt_path.stem.strip()
    stem_slug = re.sub(r"[^A-Za-z0-9\-_.]+", "_", stem).strip("_")
    h = hashlib.sha1(stem.encode("utf-8")).hexdigest()[:8]
    return f"{stem_slug}__{h}"


# -------------------------
# Quote ID + Anchor
# -------------------------

VALID_SECTION_CODES = {"M", "R", "D", "C", "L", "O"}

def normalize_quote_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text

def quote_anchor(text: str) -> str:
    norm = normalize_quote_text(text)
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()[:6]

def generate_quote_records(
    paper_id: str,
    first_author_lastname: str,
    publication_year: str,
    extracted_items: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Deterministic Quote ID rules:
      quote_id = "{first_author_lastname}-{publication_year}-{section_code}-{counter:03d}"
      counter is per paper and increments in the accepted extracted_items order.
    """
    output: List[Dict[str, Any]] = []
    counter = 1

    for item in extracted_items:
        quote = (item.get("quote") or "").strip()
        quote_with_context = (item.get("quote_with_context") or "").strip()
        section = (item.get("section") or "").strip()
        section_code = (item.get("section_code") or "").strip()

        # Validation checks
        if not quote:
            raise ValueError("Model returned an item with empty 'quote'.")
        # quote_with_context must exist; if empty, set equal to quote (per prompt contract)
        if not quote_with_context:
            quote_with_context = quote
        if section_code not in VALID_SECTION_CODES:
            raise ValueError(f"Invalid section_code '{section_code}'. Expected one of {sorted(VALID_SECTION_CODES)}.")

        q_anchor = quote_anchor(quote)
        q_id = f"{first_author_lastname}-{publication_year}-{section_code}-{counter:03d}"

        output.append({
            "paper_id": paper_id,
            "quote_id": q_id,
            "quote_anchor": q_anchor,
            "section": section,
            "section_code": section_code,
            "quote": quote,
            "quote_with_context": quote_with_context,
        })

        counter += 1

    return output


# -------------------------
# Prompt loading
# -------------------------

@dataclass
class PromptSpec:
    task: str
    version: str
    model: str
    top_p: float
    system_command: str
    user_command: str

def load_prompt_yaml(path: Path) -> PromptSpec:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    task = data.get("task", "")
    version = data.get("version", "")
    config = data.get("config", {}) or {}
    model = config.get("model", "gpt-5-mini")
    top_p = float(config.get("top_p", 1))

    system_command = data.get("system_command", "")
    user_command = data.get("user_command", "")

    if not system_command or not user_command:
        raise ValueError("YAML must include 'system_command' and 'user_command'.")
    if not task or not version:
        raise ValueError("YAML must include 'task' and 'version'.")

    return PromptSpec(
        task=task,
        version=version,
        model=model,
        top_p=top_p,
        system_command=system_command,
        user_command=user_command,
    )


# -------------------------
# OpenAI call
# -------------------------

def call_extractor(
    client: OpenAI,
    model: str,
    top_p: float,
    system_command: str,
    user_command: str,
    input_text: str,
) -> List[Dict[str, Any]]:
    user_text = user_command.replace("{input_text}", input_text)

    resp = client.responses.create(
        model=model,
        top_p=top_p,
        input=[
            {"role": "system", "content": system_command},
            {"role": "user", "content": user_text},
        ],
    )

    # The Responses API returns content in output_text for most plain-text models.
    raw = resp.output_text.strip()
    if not raw:
        raise ValueError("Empty response from model.")

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON. Error: {e}. Raw output (first 500 chars): {raw[:500]}")

    if not isinstance(parsed, list):
        raise ValueError("Model output must be a JSON array.")

    # Validate each item has exactly the keys expected by the prompt
    expected_keys = {"quote", "quote_with_context", "section", "section_code"}
    for i, item in enumerate(parsed, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"Item #{i} is not an object/dict.")
        if set(item.keys()) != expected_keys:
            raise ValueError(
                f"Item #{i} has wrong keys. Expected exactly {sorted(expected_keys)}, got {sorted(item.keys())}."
            )

    return parsed


# -------------------------
# Runner
# -------------------------

def setup_logger(log_file: Path) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

def main() -> int:
    ap = argparse.ArgumentParser(description="Run Step-2 quote extraction over a folder of .txt papers.")
    ap.add_argument("--prompt-yaml", required=True, type=Path, help="Path to YAML prompt file.")
    ap.add_argument("--input-dir", required=True, type=Path, help="Folder containing paper .txt files.")
    ap.add_argument("--output-dir", required=True, type=Path, help="Output folder for per-paper JSON files.")
    ap.add_argument("--log-file", required=True, type=Path, help="Path to log file.")
    ap.add_argument("--glob", default="*.txt", help="Glob pattern for input files (default: *.txt).")
    ap.add_argument("--max-chars", type=int, default=180000, help="Max chars of input text sent to model (truncate from end).")
    args = ap.parse_args()

    setup_logger(args.log_file)

    prompt = load_prompt_yaml(args.prompt_yaml)
    load_dotenv()
    client = OpenAI()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    txt_files = sorted(args.input_dir.glob(args.glob))
    if not txt_files:
        logging.error("No input files found in %s with glob '%s'", args.input_dir, args.glob)
        return 2

    logging.info("Task=%s Version=%s Model=%s Files=%d", prompt.task, prompt.version, prompt.model, len(txt_files))

    for idx, txt_path in enumerate(txt_files, start=1):
        file_name = txt_path.name
        logging.info("(%d/%d) Processing: %s", idx, len(txt_files), file_name)

        try:
            first_author, year = extract_first_author_and_year(file_name)
            pid = compute_paper_id(txt_path)

            text = txt_path.read_text(encoding="utf-8", errors="replace")
            if len(text) > args.max_chars:
                # keep start of doc (often contains headings); truncate tail
                text = text[:args.max_chars]
                logging.warning("Truncated input text to %d chars for %s", args.max_chars, file_name)

            extracted = call_extractor(
                client=client,
                model=prompt.model,
                top_p=prompt.top_p,
                system_command=prompt.system_command,
                user_command=prompt.user_command,
                input_text=text,
            )

            enriched = generate_quote_records(
                paper_id=pid,
                first_author_lastname=first_author,
                publication_year=year,
                extracted_items=extracted,
            )

            # Optional: flag duplicates within paper (same anchor) in log (does not change output)
            anchors = [q["quote_anchor"] for q in enriched]
            dup = {a for a in anchors if anchors.count(a) > 1}
            if dup:
                logging.warning("Duplicate quote_anchor(s) detected in %s: %s", file_name, sorted(dup))

            out_path = args.output_dir / f"{txt_path.stem}.json"
            out_path.write_text(json.dumps(enriched, ensure_ascii=False, indent=2), encoding="utf-8")

            logging.info("Saved %d quotes -> %s", len(enriched), out_path)

        except Exception as e:
            logging.exception("FAILED processing %s: %s", file_name, e)
            # Continue to next file
            continue

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
