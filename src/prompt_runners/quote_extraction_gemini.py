#!/usr/bin/env python3
"""
HCD Barrier Quote Extractor (Gemini 3.0 Flash Edition)
=====================================================

DESCRIPTION:
This script identifies and extracts verbatim HCD barrier quotes from academic PDFs.
It uses Gemini 3.0 Flash with 'High' thinking and a strict JSON Response Schema.

PREREQUISITES:
- pip install google-genai pyyaml python-dotenv pydantic

DIRECTORY STRUCTURE:
├── .env (Contains GEMINI_API_KEY)
├── prompt/barrier_quote_extraction_from_pdfv011_gemini.yaml
├── data/papers_pdf/ (Input PDFs)
└── output/quotes_step2/ (JSON results)
"""

import os
import re
import json
import yaml
import hashlib
import logging
import time
from pathlib import Path
from typing import List, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from multiprocessing import Process, Queue

# Google GenAI & Validation imports
from google import genai
from google.genai import types
from pydantic import BaseModel, Field

THINKING_LEVEL = "high"

# -------------------------
# 1. STRICT SCHEMA DEFINITION
# -------------------------

class BarrierQuote(BaseModel):
    """Schema for a single HCD barrier extraction."""
    section_code: str = Field(description="The single letter code (M, R, D, C, L, O).")
    description: str = Field(description="Brief description of the barrier (1-2 sentences).")
    justification: str = Field(description="Explains which HCD workload(s) are hindered (1-2 sentences).")
    reference_excerpt: str = Field(description="Verbatim short quote (1-3 sentences); empty if none fits.")
    certainty: str = Field(description="One of: certain, uncertain.")

class ExtractionResponse(BaseModel):
    """Wrapper to ensure API compatibility for list-based returns."""
    barriers: List[BarrierQuote]

# -------------------------
# 2. UTILITY LOGIC
# -------------------------

@dataclass
class PromptSpec:
    task: str
    version: str
    model: str
    top_p: float
    system_command: str
    user_command: str

def setup_logger(output_dir: Path, mode: str = "w"):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log", mode=mode, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )

def extract_metadata(filename: str) -> Tuple[str, str]:
    """Extracts author surname and year for deterministic IDs."""
    base = Path(filename).stem
    parts = [p.strip() for p in base.split("_") if p.strip()]
    if len(parts) < 2:
        return "unknown", "0000"

    author_part = parts[0]
    author_part = re.sub(r"\s+et al\.\s*$", "", author_part, flags=re.IGNORECASE).strip()
    author_part = re.split(r"\s+and\s+", author_part, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    # Supports both "..._2023_..." and "..._2023-..." styles.
    year = "0000"
    remainder = "_".join(parts[1:])
    year_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", remainder)
    if year_match:
        year = year_match.group(1)

    # Preserve compound surnames/particles in source name; normalize for ID safety.
    surname = re.sub(r"[^a-z0-9\-]+", "", author_part.lower().replace(" ", "-"))
    return surname or "unknown", year

# -------------------------
# 3. API CALL WRAPPERS
# -------------------------

def _generate_with_timeout(callable_fn, timeout_s: int):
    """Run a blocking call with a timeout in a worker thread."""
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(callable_fn)
        try:
            return future.result(timeout=timeout_s)
        except FuturesTimeoutError:
            future.cancel()
            raise

def generate_content_with_retry(client, prompt, uploaded_file, max_retries: int, timeout_s: int, backoff_base: float):
    """Retry generate_content with timeout and exponential backoff."""
    last_error = None
    for attempt in range(1, max_retries + 1):
        logging.info(f"Calling generate_content (attempt {attempt}/{max_retries}, timeout={timeout_s}s)")
        start = time.monotonic()
        try:
            response = _generate_with_timeout(
                lambda: client.models.generate_content(
                    model=prompt.model,
                    contents=[uploaded_file, prompt.user_command],
                    config=types.GenerateContentConfig(
                        system_instruction=prompt.system_command,
                        temperature=0.0,
                        top_p=prompt.top_p,
                        response_mime_type="application/json",
                        response_schema=ExtractionResponse,
                        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                    )
                ),
                timeout_s=timeout_s
            )
            elapsed = time.monotonic() - start
            logging.info(f"generate_content completed in {elapsed:.1f}s")
            return response
        except FuturesTimeoutError as e:
            elapsed = time.monotonic() - start
            last_error = e
            logging.warning(f"generate_content timed out after {elapsed:.1f}s")
        except Exception as e:
            elapsed = time.monotonic() - start
            last_error = e
            logging.warning(f"generate_content failed after {elapsed:.1f}s: {e}")

        if attempt < max_retries:
            sleep_s = backoff_base ** (attempt - 1)
            logging.info(f"Retrying after {sleep_s:.1f}s...")
            time.sleep(sleep_s)

    raise RuntimeError(f"generate_content failed after {max_retries} attempts: {last_error}")

# -------------------------
# 4. CORE RUNNER
# -------------------------

def _process_pdf_worker(pdf_path_str: str, prompt_dict: dict, api_key: str, output_dir_str: str,
                        max_retries: int, timeout_s: int, backoff_base: float, result_q: Queue):
    try:
        output_dir = Path(output_dir_str)
        setup_logger(output_dir, mode="a")

        pdf_path = Path(pdf_path_str)
        prompt = PromptSpec(**prompt_dict)
        client = genai.Client(api_key=api_key)

        author, year = extract_metadata(pdf_path.name)
        paper_id = f"{pdf_path.stem}__{hashlib.sha1(pdf_path.stem.encode()).hexdigest()[:8]}"

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        logging.info(f"Uploading PDF ({file_size_mb:.1f} MB)")
        uploaded_file = client.files.upload(file=pdf_path)
        logging.info("Upload complete")

        response = generate_content_with_retry(
            client=client,
            prompt=prompt,
            uploaded_file=uploaded_file,
            max_retries=max_retries,
            timeout_s=timeout_s,
            backoff_base=backoff_base,
        )

        if not response.parsed or not hasattr(response.parsed, 'barriers'):
            logging.warning(f"No barriers found or empty response for {pdf_path.name}")
            barrier_list = []
        else:
            barrier_list = response.parsed.barriers

        enriched_results = []
        for i, item in enumerate(barrier_list, 1):
            enriched_results.append({
                "paper_id": paper_id,
                "quote_id": f"{author}-{year}-{item.section_code}-{i:03d}",
                "section_code": item.section_code,
                "description": item.description,
                "justification": item.justification,
                "reference_excerpt": item.reference_excerpt,
                "certainty": item.certainty
            })

        out_file = output_dir / f"{pdf_path.stem}.json"
        with open(out_file, 'w', encoding='utf-8') as f:
            json.dump(enriched_results, f, indent=2, ensure_ascii=False)

        logging.info(f"Successfully saved {len(enriched_results)} quotes to {out_file.name}")
        result_q.put({"ok": True, "count": len(enriched_results)})
    except Exception as e:
        logging.error(f"Failed processing {Path(pdf_path_str).name}: {str(e)}")
        result_q.put({"ok": False, "error": str(e)})

def run_extraction():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        return

    prompt_path = Path("prompt/barrier_quote_extraction_pr_pdfv0.1.3.yaml")
    input_dir = Path("data/test_data/gpt_test/barrier_identification_test/Quote_extraction/pdf_files/first_study")
    output_dir = Path("data/test_data/gpt_test/barrier_identification_test/Quote_extraction/gemini/5_first_study_paper_v013")
    max_retries = 3
    timeout_s = 600
    backoff_base = 2.0
    hard_timeout_s = timeout_s * max_retries + 60

    setup_logger(output_dir)

    # Load YAML Prompt
    try:
        with open(prompt_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
            prompt = PromptSpec(
                task=data['task'],
                version=data['version'],
                model=data['config']['model'],
                top_p=data['config']['top_p'],
                system_command=data['system_command'],
                user_command=data['user_command']
            )
    except Exception as e:
        logging.error(f"Failed to load YAML prompt: {e}")
        return

    runner_rel_path = Path(os.path.relpath(Path(__file__).resolve(), Path.cwd()))
    prompt_rel_path = Path(os.path.relpath(prompt_path.resolve(), Path.cwd()))
    input_rel_path = Path(os.path.relpath(input_dir.resolve(), Path.cwd()))
    logging.info("----- Run Config -----")
    logging.info(f"Gemini Model: {prompt.model}")
    logging.info(f"Thinking level: {THINKING_LEVEL}")
    logging.info(f"Prompt version: {prompt.version}")
    logging.info(f"Prompt relative path: {prompt_rel_path}")
    logging.info(f"Prompt runner relative path: {runner_rel_path}")
    logging.info(f"Input relative path: {input_rel_path}")
    logging.info("----------------------")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    logging.info(f"Task: {prompt.task} | Found {len(pdf_files)} PDFs")

    for pdf_path in pdf_files:
        out_file = output_dir / f"{pdf_path.stem}.json"
        if out_file.exists():
            logging.info(f"Skipping (already processed): {pdf_path.name}")
            continue
        logging.info(f"Processing: {pdf_path.name}")
        prompt_dict = {
            "task": prompt.task,
            "version": prompt.version,
            "model": prompt.model,
            "top_p": prompt.top_p,
            "system_command": prompt.system_command,
            "user_command": prompt.user_command,
        }

        completed = False
        for attempt in range(1, max_retries + 1):
            result_q = Queue()
            proc = Process(
                target=_process_pdf_worker,
                args=(
                    str(pdf_path),
                    prompt_dict,
                    api_key,
                    str(output_dir),
                    max_retries,
                    timeout_s,
                    backoff_base,
                    result_q,
                ),
            )
            proc.start()
            proc.join(hard_timeout_s)
            if proc.is_alive():
                proc.terminate()
                proc.join()
                logging.warning(
                    f"Hard timeout after {hard_timeout_s}s for {pdf_path.name} (attempt {attempt}/{max_retries})"
                )
                continue

            result = result_q.get() if not result_q.empty() else {"ok": False, "error": "No result returned"}
            if result.get("ok"):
                completed = True
                break

            logging.warning(
                f"Attempt {attempt}/{max_retries} failed for {pdf_path.name}: {result.get('error')}"
            )

        if not completed:
            logging.error(f"Exhausted retries for {pdf_path.name}; moving to next file.")

    logging.info("Batch processing complete.")

if __name__ == "__main__":
    run_extraction()
