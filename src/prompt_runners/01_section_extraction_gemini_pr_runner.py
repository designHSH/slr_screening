#!/usr/bin/env python3
"""
Gemini PDF Section Extraction Prompt Runner
==========================================

Goal:
- Batch process PDFs in a folder
- For each PDF, call Gemini (gemini-3-flash-preview) using a YAML prompt spec
- Save ONLY the model output as a .md file (Controlled Markdown), same stem as PDF

Prerequisites:
- pip install google-genai pyyaml python-dotenv

Environment:
- .env must contain: GEMINI_API_KEY=...

Example:
python section_extraction_gemini_runner.py \
  --input_dir data/papers_pdf \
  --output_dir output/sections_step1 \
  --prompt_yaml prompt/section_extraction_gemini.yaml

Optional:
  --force        overwrite existing outputs
  --max_retries  default 3
  --timeout_s    per-attempt timeout (seconds), default 600
  --backoff_base exponential backoff base, default 2.0
"""

import os
import sys
import yaml
import time
import logging
import argparse
from pathlib import Path
from dataclasses import dataclass
from multiprocessing import Process, Queue
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

from dotenv import load_dotenv

# Official Google GenAI SDK
from google import genai
from google.genai import types

THINKING_LEVEL = "medium"  # gemini-3-flash-preview supports: low/medium/high


# -------------------------
# 1) Data Structures
# -------------------------

@dataclass
class PromptSpec:
    prompt_type: str
    revision: str
    model: str
    system_command: str
    user_command: str


# -------------------------
# 2) Logging (matches your sample style)
# -------------------------

def setup_logger(output_dir: Path, mode: str = "w") -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "run.log", mode=mode, encoding="utf-8"),
            logging.StreamHandler()
        ],
    )


# -------------------------
# 3) Prompt Loading
# -------------------------

def load_prompt_yaml(prompt_yaml: Path) -> PromptSpec:
    """Load YAML prompt and map fields exactly as requested."""
    with open(prompt_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    # Expected YAML keys (based on your uploaded file):
    # prompt_type, revision, config.model, user_command, system_command
    try:
        prompt_type = data["prompt_type"]
        revision = str(data["revision"])
        model = data["config"]["model"]
        system_command = data["system_command"]
        user_command = data["user_command"]
    except Exception as e:
        raise ValueError(f"Invalid prompt YAML structure: {e}")

    return PromptSpec(
        prompt_type=prompt_type,
        revision=revision,
        model=model,
        system_command=system_command,
        user_command=user_command,
    )


# -------------------------
# 4) Timeout + Retry wrappers (matches your sample logic)
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


def generate_content_with_retry(
    client: genai.Client,
    prompt: PromptSpec,
    uploaded_file,
    max_retries: int,
    timeout_s: int,
    backoff_base: float,
):
    """
    Retry generate_content with timeout + exponential backoff on transient failures.
    Preserves model output (no post-processing).
    """
    last_error = None
    for attempt in range(1, max_retries + 1):
        logging.info(f"Calling generate_content (attempt {attempt}/{max_retries}, timeout={timeout_s}s)")
        start = time.monotonic()
        try:
            response = _generate_with_timeout(
                lambda: client.models.generate_content(
                    model=prompt.model,  # will be forced to gemini-3-flash-preview in main()
                    contents=[uploaded_file, prompt.user_command],
                    config=types.GenerateContentConfig(
                        system_instruction=prompt.system_command,
                        temperature=0.0,   # determinism
                        top_p=1.0,         # determinism
                        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                        thinking_config=types.ThinkingConfig(thinking_level=THINKING_LEVEL),
                    ),
                ),
                timeout_s=timeout_s,
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
# 5) Worker (separate process per PDF, like your sample)
# -------------------------

def _process_pdf_worker(
    pdf_path_str: str,
    prompt_dict: dict,
    api_key: str,
    output_dir_str: str,
    max_retries: int,
    timeout_s: int,
    backoff_base: float,
    force: bool,
    result_q: Queue,
):
    """
    Worker process:
    - attaches PDF (Gemini accepts whole PDF)
    - applies system + user commands from YAML
    - saves ONLY the raw model output to .md
    """
    try:
        output_dir = Path(output_dir_str)
        setup_logger(output_dir, mode="a")  # append in child processes

        pdf_path = Path(pdf_path_str)
        prompt = PromptSpec(**prompt_dict)

        out_file = output_dir / f"{pdf_path.stem}.md"
        if out_file.exists() and not force:
            logging.info(f"Skipping (already processed): {pdf_path.name}")
            result_q.put({"ok": True, "skipped": True})
            return

        client = genai.Client(api_key=api_key)

        file_size_mb = pdf_path.stat().st_size / (1024 * 1024)
        logging.info(f"Uploading PDF ({file_size_mb:.1f} MB): {pdf_path.name}")
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

        # IMPORTANT: preserve output exactly; do NOT strip/normalize
        # google-genai response generally exposes response.text
        raw_text = response.text
        if raw_text is None or raw_text.strip() == "":
            raise RuntimeError(f"Model returned no text output for {pdf_path.name}")

        with open(out_file, "w", encoding="utf-8") as f:
            f.write(raw_text)

        logging.info(f"Saved Controlled Markdown to {out_file.name}")
        result_q.put({"ok": True, "skipped": False})

    except Exception as e:
        logging.error(f"Failed processing {Path(pdf_path_str).name}: {str(e)}")
        result_q.put({"ok": False, "error": str(e)})


# -------------------------
# 6) Main Runner (CLI + summary)
# -------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch run a Gemini YAML prompt over PDFs (Controlled Markdown output).")
    p.add_argument("--input_dir", default=None, help="Directory containing PDF files.")
    p.add_argument("--output_dir", default=None, help="Directory to write .md outputs and run.log.")
    p.add_argument("--prompt_yaml", default=None, help="Path to YAML prompt file.")
    p.add_argument("--force", action="store_true", help="Overwrite existing .md outputs.")
    p.add_argument("--max_retries", type=int, default=3, help="Retries per PDF (default: 3).")
    p.add_argument("--timeout_s", type=int, default=300, help="Per-attempt timeout in seconds (default: 300).")
    p.add_argument("--backoff_base", type=float, default=2.0, help="Exponential backoff base (default: 2.0).")
    return p.parse_args()


def _prompt_path(label: str, default: str | None = None) -> str:
    while True:
        suffix = f" [{default}]" if default else ""
        raw = input(f"{label}{suffix}: ").strip().strip("\"'")
        if raw:
            return os.path.expanduser(raw)
        if default:
            return os.path.expanduser(default)
        print("Please provide a path.")


def main() -> int:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY not found in .env file.")
        return 2

    args = parse_args()
    # Always prompt interactively; allow Enter to accept provided args (if any)
    args.input_dir = _prompt_path("Input PDF directory", args.input_dir)
    args.output_dir = _prompt_path("Output directory", args.output_dir)
    args.prompt_yaml = _prompt_path("Prompt YAML path", args.prompt_yaml)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    prompt_yaml = Path(args.prompt_yaml)

    setup_logger(output_dir, mode="w")

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"Input directory not found or not a directory: {input_dir}")
        return 2
    if not prompt_yaml.exists():
        logging.error(f"Prompt YAML not found: {prompt_yaml}")
        return 2

    # Load prompt
    try:
        prompt = load_prompt_yaml(prompt_yaml)
    except Exception as e:
        logging.error(f"Failed to load YAML prompt: {e}")
        return 2

    runner_rel_path = Path(os.path.relpath(Path(__file__).resolve(), Path.cwd()))
    prompt_rel_path = Path(os.path.relpath(prompt_yaml.resolve(), Path.cwd()))
    input_rel_path = Path(os.path.relpath(input_dir.resolve(), Path.cwd()))
    output_rel_path = Path(os.path.relpath(output_dir.resolve(), Path.cwd()))

    logging.info("----- Run Config -----")
    logging.info(f"Gemini Model: {prompt.model}")
    logging.info(f"Thinking level: {THINKING_LEVEL}")
    logging.info(f"Prompt type: {prompt.prompt_type}")
    logging.info(f"Prompt revision: {prompt.revision}")
    logging.info(f"Prompt relative path: {prompt_rel_path}")
    logging.info(f"Prompt runner relative path: {runner_rel_path}")
    logging.info(f"Input relative path: {input_rel_path}")
    logging.info(f"Output relative path: {output_rel_path}")
    logging.info(f"Force overwrite: {args.force}")
    logging.info("----------------------")

    pdf_files = sorted(input_dir.glob("*.pdf"))
    logging.info(f"Found {len(pdf_files)} PDFs")

    success = 0
    failed = 0
    skipped = 0

    # hard timeout like your sample: worst-case attempt-time * retries + buffer
    hard_timeout_s = args.timeout_s * args.max_retries + 60

    for pdf_path in pdf_files:
        out_file = output_dir / f"{pdf_path.stem}.md"
        if out_file.exists() and not args.force:
            logging.info(f"Skipping (already processed): {pdf_path.name}")
            skipped += 1
            continue

        logging.info(f"Processing: {pdf_path.name}")

        prompt_dict = {
            "prompt_type": prompt.prompt_type,
            "revision": prompt.revision,
            "model": prompt.model,
            "system_command": prompt.system_command,
            "user_command": prompt.user_command,
        }

        # Outer attempts: restart whole process if hard-timeout kills it
        completed = False
        last_err = None

        for attempt in range(1, args.max_retries + 1):
            result_q = Queue()
            proc = Process(
                target=_process_pdf_worker,
                args=(
                    str(pdf_path),
                    prompt_dict,
                    api_key,
                    str(output_dir),
                    args.max_retries,   # inner retry for transient errors
                    args.timeout_s,
                    args.backoff_base,
                    args.force,
                    result_q,
                ),
            )

            proc.start()
            proc.join(hard_timeout_s)

            if proc.is_alive():
                proc.terminate()
                proc.join()
                last_err = f"Hard timeout after {hard_timeout_s}s"
                logging.warning(f"{last_err} for {pdf_path.name} (attempt {attempt}/{args.max_retries})")
                continue

            result = result_q.get() if not result_q.empty() else {"ok": False, "error": "No result returned"}
            if result.get("ok"):
                if result.get("skipped"):
                    skipped += 1
                else:
                    success += 1
                completed = True
                break

            last_err = result.get("error", "Unknown error")
            logging.warning(f"Attempt {attempt}/{args.max_retries} failed for {pdf_path.name}: {last_err}")

        if not completed:
            failed += 1
            logging.error(f"Exhausted retries for {pdf_path.name}; moving to next file. Last error: {last_err}")

    logging.info("----- Run Summary -----")
    logging.info(f"Total PDFs: {len(pdf_files)}")
    logging.info(f"Success:    {success}")
    logging.info(f"Skipped:    {skipped}")
    logging.info(f"Failed:     {failed}")
    logging.info("------------------------")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
