from __future__ import annotations

import csv
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
import multiprocessing as mp


# -----------------------------
# Utilities
# -----------------------------
def utc_now_str() -> str:
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")


def slugify_ascii(text: str) -> str:
    """
    Make filename safe:
    - convert to ASCII-friendly
    - replace weird chars with closest basic char where possible
    - collapse spaces/underscores into hyphens
    - keep [a-z0-9-]
    """
    # minimal transliteration without external deps:
    # replace common diacritics (extend if you want)
    repl = {
        "ö": "o", "Ö": "o",
        "ä": "a", "Ä": "a",
        "ü": "u", "Ü": "u",
        "é": "e", "è": "e", "ê": "e", "É": "e", "È": "e", "Ê": "e",
        "á": "a", "à": "a", "â": "a", "Á": "a", "À": "a", "Â": "a",
        "í": "i", "ì": "i", "î": "i", "Í": "i", "Ì": "i", "Î": "i",
        "ó": "o", "ò": "o", "ô": "o", "Ó": "o", "Ò": "o", "Ô": "o",
        "ú": "u", "ù": "u", "û": "u", "Ú": "u", "Ù": "u", "Û": "u",
        "ç": "c", "Ç": "c",
        "ñ": "n", "Ñ": "n",
    }
    for k, v in repl.items():
        text = text.replace(k, v)

    text = text.lower().strip()
    text = re.sub(r"[ \t\r\n_]+", "-", text)
    text = re.sub(r"[^a-z0-9\-]+", "", text)
    text = re.sub(r"-{2,}", "-", text).strip("-")
    return text or "paper"


def short_hash_from_path(path: Path, n: int = 8) -> str:
    """
    Deterministic short id from filename + size + modified time.
    (Stable unless the file changes.)
    """
    import hashlib
    stat = path.stat()
    payload = f"{path.name}|{stat.st_size}|{int(stat.st_mtime)}".encode("utf-8", errors="ignore")
    return hashlib.sha256(payload).hexdigest()[:n]


# -----------------------------
# Logging structures
# -----------------------------
@dataclass
class ExtractionLogRow:
    timestamp_utc: str
    original_pdf_name: str
    doc_id: str
    input_pdf_path: str
    structured_json_path: str
    markdown_path: str
    status: str  # PASS / FAIL / TIMEOUT / ERROR
    attempt: int
    duration_sec: float
    extracted_chars: int
    text_blocks: int
    table_blocks: int
    figure_blocks: int
    error_type: Optional[str]
    error_message: Optional[str]


# -----------------------------
# Worker process (Docling call)
# -----------------------------
def _docling_worker(
    input_pdf: str,
    structured_json_out: str,
    markdown_out: str,
    allow_tables: bool,
    allow_figures: bool,
    result_queue: mp.Queue
) -> None:
    """
    Runs in a child process.
    Must NEVER print huge stuff; returns a small dict via result_queue.
    """
    try:
        # Import inside worker so parent can kill the entire process safely
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        t0 = time.time()
        doc = converter.convert(input_pdf).document  # DoclingDocument

        # 1) Export structured JSON-like
        structured = doc.export_to_dict()  # stable representation for your raw archive
        Path(structured_json_out).write_text(json.dumps(structured, ensure_ascii=False, indent=2), encoding="utf-8")

        # 2) Export Markdown
        # NOTE: this is a "preview" renderer. You'll build a stricter custom markdown later.
        md = doc.export_to_markdown()
        Path(markdown_out).write_text(md, encoding="utf-8")

        # 3) Quick stats for deterministic gates + logs
        extracted_chars = len(md)
        text_blocks = len(structured.get("texts", [])) if isinstance(structured, dict) else 0
        table_blocks = len(structured.get("tables", [])) if isinstance(structured, dict) else 0
        figure_blocks = len(structured.get("pictures", [])) if isinstance(structured, dict) else 0

        if not allow_tables:
            table_blocks = 0
        if not allow_figures:
            figure_blocks = 0

        result_queue.put({
            "ok": True,
            "duration_sec": time.time() - t0,
            "extracted_chars": extracted_chars,
            "text_blocks": text_blocks,
            "table_blocks": table_blocks,
            "figure_blocks": figure_blocks,
        })

    except Exception as e:
        result_queue.put({
            "ok": False,
            "error_type": type(e).__name__,
            "error_message": str(e),
            "traceback": traceback.format_exc(limit=10),
        })


# -----------------------------
# Main batch runner
# -----------------------------
class DoclingBatchExtractor:
    def __init__(
        self,
        input_dir: Path,
        out_structured_dir: Path,
        out_markdown_dir: Path,
        log_csv_path: Path,
        timeout_seconds: int = 600,
        max_attempts: int = 2,
        allow_tables: bool = True,
        allow_figures: bool = True,
    ) -> None:
        self.input_dir = input_dir
        self.out_structured_dir = out_structured_dir
        self.out_markdown_dir = out_markdown_dir
        self.log_csv_path = log_csv_path
        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.allow_tables = allow_tables
        self.allow_figures = allow_figures

        self.out_structured_dir.mkdir(parents=True, exist_ok=True)
        self.out_markdown_dir.mkdir(parents=True, exist_ok=True)
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)

        self._ensure_log_header()

    def _ensure_log_header(self) -> None:
        if not self.log_csv_path.exists():
            with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(asdict(ExtractionLogRow(
                    timestamp_utc="",
                    original_pdf_name="",
                    doc_id="",
                    input_pdf_path="",
                    structured_json_path="",
                    markdown_path="",
                    status="",
                    attempt=0,
                    duration_sec=0.0,
                    extracted_chars=0,
                    text_blocks=0,
                    table_blocks=0,
                    figure_blocks=0,
                    error_type=None,
                    error_message=None,
                )).keys()))
                writer.writeheader()

    def _append_log(self, row: ExtractionLogRow) -> None:
        with self.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(asdict(row).keys()))
            writer.writerow(asdict(row))

    def _write_reject(self, reject_path: Path, payload: Dict[str, Any]) -> None:
        reject_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def run(self) -> None:
        pdfs = sorted(self.input_dir.glob("*.pdf"))
        total = len(pdfs)
        print(f"Found {total} PDFs in: {self.input_dir}")

        for i, pdf_path in enumerate(pdfs, start=1):
            print(f"\n[{utc_now_str()}] ({i}/{total}) Processing: {pdf_path.name}")

            # Build doc_id = safe slug + hash
            base = pdf_path.stem
            safe = slugify_ascii(base)
            doc_id = f"{safe}__{short_hash_from_path(pdf_path)}"

            structured_json = self.out_structured_dir / f"{doc_id}.structured.json"
            markdown_file = self.out_markdown_dir / f"{doc_id}.raw.md"
            reject_file = self.out_structured_dir / f"{doc_id}.reject.json"

            # Attempts loop
            final_status = "FAIL"
            last_error_type = None
            last_error_message = None
            duration = 0.0
            extracted_chars = 0
            text_blocks = 0
            table_blocks = 0
            figure_blocks = 0

            for attempt in range(1, self.max_attempts + 1):
                t0 = time.time()
                print(f"  -> doc_id: {doc_id}")
                print(f"  -> attempt {attempt}/{self.max_attempts} (timeout={self.timeout_seconds}s)")

                q: mp.Queue = mp.Queue()
                p = mp.Process(
                    target=_docling_worker,
                    args=(
                        str(pdf_path),
                        str(structured_json),
                        str(markdown_file),
                        self.allow_tables,
                        self.allow_figures,
                        q
                    ),
                    daemon=True
                )
                p.start()
                p.join(timeout=self.timeout_seconds)

                if p.is_alive():
                    # TIMEOUT: kill and record
                    p.terminate()
                    p.join(timeout=5)
                    duration = time.time() - t0
                    final_status = "TIMEOUT"
                    last_error_type = "TimeoutError"
                    last_error_message = f"Docling conversion exceeded {self.timeout_seconds} seconds (likely OCR hang)."
                    print(f"  !! TIMEOUT: killed worker process after {duration:.1f}s")
                else:
                    # Read worker result
                    if not q.empty():
                        result = q.get()
                    else:
                        result = {"ok": False, "error_type": "EmptyResult", "error_message": "Worker returned no result."}

                    duration = time.time() - t0

                    if result.get("ok"):
                        extracted_chars = int(result.get("extracted_chars", 0))
                        text_blocks = int(result.get("text_blocks", 0))
                        table_blocks = int(result.get("table_blocks", 0))
                        figure_blocks = int(result.get("figure_blocks", 0))

                        # Minimal deterministic gate: if extraction is extremely low, treat as FAIL
                        # (Adjust thresholds as you learn your dataset)
                        if extracted_chars < 500 and pdf_path.stat().st_size > 200_000:
                            final_status = "FAIL"
                            last_error_type = "TextTooLow"
                            last_error_message = f"Extracted markdown too small ({extracted_chars} chars). Likely OCR empty or PDF is image-only."
                            print(f"  !! FAIL: {last_error_message}")
                        else:
                            final_status = "PASS"
                            last_error_type = None
                            last_error_message = None
                            print(f"  -> PASS in {duration:.1f}s | chars={extracted_chars} texts={text_blocks} tables={table_blocks} figs={figure_blocks}")
                            break
                    else:
                        final_status = "ERROR"
                        last_error_type = result.get("error_type", "DoclingError")
                        last_error_message = result.get("error_message", "Unknown error")
                        print(f"  !! ERROR: {last_error_type}: {last_error_message}")

                # If not last attempt, retry
                if attempt < self.max_attempts and final_status != "PASS":
                    print("  -> retrying...")
                    time.sleep(1.0)

            # Write reject file if not PASS
            if final_status != "PASS":
                reject_payload = {
                    "timestamp_utc": utc_now_str(),
                    "doc_id": doc_id,
                    "input_pdf": str(pdf_path),
                    "status": final_status,
                    "error_type": last_error_type,
                    "error_message": last_error_message,
                    "notes": "This paper was not processed into safe output. See logs for details."
                }
                self._write_reject(reject_file, reject_payload)
                print(f"  -> wrote reject: {reject_file.name}")

            # Always append CSV log row
            log_row = ExtractionLogRow(
                timestamp_utc=utc_now_str(),
                original_pdf_name=pdf_path.name,
                doc_id=doc_id,
                input_pdf_path=str(pdf_path),
                structured_json_path=str(structured_json),
                markdown_path=str(markdown_file),
                status=final_status,
                attempt=attempt,
                duration_sec=float(duration),
                extracted_chars=int(extracted_chars),
                text_blocks=int(text_blocks),
                table_blocks=int(table_blocks),
                figure_blocks=int(figure_blocks),
                error_type=last_error_type,
                error_message=last_error_message,
            )
            self._append_log(log_row)


if __name__ == "__main__":
    # IMPORTANT for Windows multiprocessing
    mp.freeze_support()
    mp.set_start_method("spawn", force=True)

    # Update these paths to your folders:
    INPUT_DIR = Path(r"data\leveraging_docling\1_raw_pdfs")
    OUT_STRUCTURED = Path(r"data\leveraging_docling\2_docling_output")
    OUT_MD = Path(r"data\leveraging_docling\3_docling_markdown")
    LOG_CSV = OUT_STRUCTURED / "extraction_log.csv"

    extractor = DoclingBatchExtractor(
        input_dir=INPUT_DIR,
        out_structured_dir=OUT_STRUCTURED,
        out_markdown_dir=OUT_MD,
        log_csv_path=LOG_CSV,
        timeout_seconds=600,   # e.g., 10 minutes per PDF
        max_attempts=2,        # retry once
        allow_tables=True,
        allow_figures=True,
    )
    extractor.run()
