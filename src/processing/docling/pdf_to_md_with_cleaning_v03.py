"""
Docling-based PDF → (Structured JSON + Raw Markdown + Header/Footer-cleaned Markdown) batch extractor
===============================================================================================

What this script does (deterministic, rule-based):
1) Loops over PDFs in:     data\leveraging_docling\1_raw_pdfs
2) For each PDF:
   - Runs Docling extraction in a timeout-safe subprocess (Windows-safe)
   - Saves structured JSON (Docling export) to:  data\leveraging_docling\2_docling_output
   - Creates a "raw" Markdown for quick human review (NOT trusted for structure)
   - Creates a "clean" Markdown by removing repetitive headers/footers deterministically
   - Writes an audit.json per PDF describing what was removed
   - On failure, writes reject.json per PDF with error + evidence

3) Writes a CSV run log to: data\leveraging_docling\2_docling_output\extraction_log.csv

Notes:
- This stage only removes repetitive header/footer (and obvious running headers).
- It DOES NOT remove funding/acknowledgements/etc. (you can add later).
- OCR edge cases are handled by retry + timeout so the batch won't get stuck.
"""

from __future__ import annotations

import csv
import json
import multiprocessing as mp
import os
import re
import sys
import time
import traceback
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ------------------------------
# Utilities
# ------------------------------

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def read_text_safe(p: Path, max_chars: int = 8000) -> str:
    try:
        s = p.read_text(encoding="utf-8", errors="replace")
        return s[:max_chars]
    except Exception:
        return ""


def strip_diacritics(s: str) -> str:
    """
    Converts e.g., 'björling' -> 'bjorling' deterministically.
    """
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in nfkd if not unicodedata.combining(ch))


def slugify_filename(base: str, max_len: int = 120) -> str:
    """
    Deterministic filename "slug":
      - remove diacritics
      - lowercase
      - replace non-alnum with hyphens
      - collapse hyphens
      - trim
    """
    base = strip_diacritics(base)
    base = base.lower()
    base = re.sub(r"[^a-z0-9]+", "-", base)
    base = re.sub(r"-{2,}", "-", base).strip("-")
    if not base:
        base = "document"
    return base[:max_len]


def short_hash(s: str, n: int = 8) -> str:
    """
    Deterministic short hash for stable IDs.
    Uses stdlib only (sha256), returns n hex chars.
    """
    import hashlib
    h = hashlib.sha256(s.encode("utf-8", errors="ignore")).hexdigest()
    return h[:n]


def compute_doc_id(pdf_path: Path) -> str:
    """
    doc_id = slugified filename + '__' + short hash of (absolute path + size + mtime)
    This stays stable unless the file changes.
    """
    base = slugify_filename(pdf_path.stem)
    try:
        st = pdf_path.stat()
        sig = f"{pdf_path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    except Exception:
        sig = str(pdf_path.resolve())
    return f"{base}__{short_hash(sig, 8)}"


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def write_json(path: Path, obj: Any) -> None:
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8", errors="replace")


# ------------------------------
# Config
# ------------------------------

@dataclass(frozen=True)
class PipelineConfig:
    output_dir: Path
    require_references: bool = False
    allow_figures: bool = True
    allow_tables: bool = True
    mode: str = "strict"  # "strict" or "best-effort" (for this stage: strict affects validation)
    timeout_seconds: int = 600
    max_attempts: int = 2


# ------------------------------
# Docling Extraction (in subprocess)
# ------------------------------

def _docling_convert_worker(pdf_path_str: str, allow_tables: bool, allow_figures: bool, q: mp.Queue) -> None:
    """
    Runs inside a subprocess. Any crash stays contained.
    Returns a payload dict via Queue.
    """
    t0 = time.time()
    pdf_path = Path(pdf_path_str)

    try:
        # Docling import inside subprocess (important for Windows spawn)
        from docling.document_converter import DocumentConverter  # type: ignore

        converter = DocumentConverter()
        result = converter.convert(str(pdf_path))

        # Different docling versions expose document in slightly different ways.
        doc = getattr(result, "document", None) or getattr(result, "doc", None) or result

        # Structured export
        structured = None
        if hasattr(doc, "export_to_dict"):
            structured = doc.export_to_dict()
        elif hasattr(doc, "model_dump"):
            structured = doc.model_dump()
        elif hasattr(doc, "to_dict"):
            structured = doc.to_dict()
        else:
            # Last resort: try to serialize something basic
            structured = {"warning": "No structured export method found on DoclingDocument."}

        # Raw markdown export (for quick human review only)
        raw_md = None
        if hasattr(doc, "export_to_markdown"):
            raw_md = doc.export_to_markdown()
        else:
            # Fallback: build something minimal from structured texts if present
            raw_md = build_markdown_from_structured(structured)

        payload = {
            "ok": True,
            "structured": structured,
            "raw_markdown": raw_md or "",
            "seconds": round(time.time() - t0, 3),
        }
        q.put(payload)
        return

    except Exception as e:
        payload = {
            "ok": False,
            "error": f"{type(e).__name__}: {e}",
            "traceback": traceback.format_exc(),
            "seconds": round(time.time() - t0, 3),
        }
        q.put(payload)
        return


def run_with_timeout(pdf_path: Path, allow_tables: bool, allow_figures: bool, timeout_seconds: int) -> Dict[str, Any]:
    """
    Runs docling conversion in a subprocess with a hard timeout.
    """
    q: mp.Queue = mp.Queue()
    p = mp.Process(
        target=_docling_convert_worker,
        args=(str(pdf_path), allow_tables, allow_figures, q),
        daemon=True,
    )
    p.start()
    p.join(timeout_seconds)

    if p.is_alive():
        try:
            p.terminate()
        except Exception:
            pass
        return {
            "ok": False,
            "error": f"TimeoutError: Docling conversion exceeded {timeout_seconds}s",
            "traceback": "",
            "seconds": float(timeout_seconds),
        }

    try:
        if not q.empty():
            return q.get_nowait()
    except Exception:
        pass

    return {
        "ok": False,
        "error": "RuntimeError: No result returned from subprocess",
        "traceback": "",
        "seconds": 0.0,
    }


# ------------------------------
# Header/Footer Cleaning (deterministic)
# ------------------------------

def normalize_line(s: str) -> str:
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def looks_like_page_number(line: str) -> bool:
    line = line.strip()
    # "12", "Page 12", "12 / 34"
    if re.fullmatch(r"\d{1,4}", line):
        return True
    if re.fullmatch(r"page\s+\d{1,4}", line, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{1,4}\s*/\s*\d{1,4}", line):
        return True
    return False


def is_short_headerish_line(line: str) -> bool:
    line_n = normalize_line(line)
    if not line_n:
        return False
    if len(line_n) <= 4 and looks_like_page_number(line_n):
        return True
    # Short running header (journal name, article title fragment)
    if len(line_n) <= 80:
        return True
    return False


def extract_candidate_header_footer_lines(md: str, top_k: int = 3, bottom_k: int = 3) -> Tuple[List[str], List[str]]:
    """
    Heuristic: treat each "page" boundary as "---PAGE---" if docling inserted (it usually doesn't).
    So we do a generic approach:
      - Split into "chunks" by blank lines and large separations
      - Take first few non-empty lines as header candidates
      - Take last few non-empty lines as footer candidates
    This is conservative and works best when docling includes per-page patterns.
    """
    # We approximate pages by repeated figure placeholders or large separators,
    # but if not present we still compute repeated lines globally.
    lines = [normalize_line(x) for x in md.splitlines()]
    lines = [x for x in lines if x]  # remove empties
    if not lines:
        return [], []

    header_candidates = lines[: min(len(lines), top_k)]
    footer_candidates = lines[max(0, len(lines) - bottom_k):]
    return header_candidates, footer_candidates


def detect_repeated_noise_lines(md: str, min_occurrences: int = 6) -> Dict[str, int]:
    """
    Detect lines that repeat many times. Those are typical headers/footers/running headers.
    Deterministic: exact normalized line match.
    """
    counts: Dict[str, int] = {}
    for raw in md.splitlines():
        line = normalize_line(raw)
        if not line:
            continue
        # avoid counting real headings (start with #) too aggressively
        if line.startswith("#"):
            continue
        # avoid counting table separator rows
        if re.fullmatch(r"\|?\s*[-:]+\s*(\|\s*[-:]+\s*)+\|?", line):
            continue
        counts[line] = counts.get(line, 0) + 1

    # Keep only those above threshold and "headerish" enough
    repeated = {
        line: c
        for line, c in counts.items()
        if c >= min_occurrences and is_short_headerish_line(line)
    }
    return repeated


def remove_repeated_headers_footers(md: str, min_occurrences: int = 6) -> Tuple[str, Dict[str, Any]]:
    """
    Removes repeated lines across document that look like header/footer.
    Also removes pure page-number lines.
    """
    repeated = detect_repeated_noise_lines(md, min_occurrences=min_occurrences)

    removed_lines: List[str] = []
    out_lines: List[str] = []

    for raw in md.splitlines():
        norm = normalize_line(raw)
        if not norm:
            out_lines.append(raw)
            continue

        # Remove page-number-only lines
        if looks_like_page_number(norm):
            removed_lines.append(norm)
            continue

        # Remove repeated noise lines
        if norm in repeated:
            removed_lines.append(norm)
            continue

        out_lines.append(raw)

    audit = {
        "removed_repeated_lines_count": len(removed_lines),
        "removed_repeated_lines_top": sorted(list(set(removed_lines)))[:50],
        "repeated_noise_candidates": dict(sorted(repeated.items(), key=lambda x: -x[1]))[:50],
    }
    return "\n".join(out_lines).strip() + "\n", audit


# ------------------------------
# Markdown builder fallback
# ------------------------------

def build_markdown_from_structured(structured: Any) -> str:
    """
    Very conservative fallback markdown builder (only used if doc.export_to_markdown() doesn't exist).
    It tries to pull 'texts' and their content in order.
    """
    if not isinstance(structured, dict):
        return ""

    parts: List[str] = []
    # Docling exports often contain keys like 'texts', 'body', 'pages', etc.
    texts = structured.get("texts")
    if isinstance(texts, list):
        for t in texts:
            if not isinstance(t, dict):
                continue
            content = t.get("text") or t.get("content") or ""
            content = str(content).strip()
            if content:
                parts.append(content)

    # Fallback: if nothing, dump keys for debug
    if not parts:
        parts.append("<!-- No text extracted by fallback markdown builder -->")
        parts.append(f"<!-- structured keys: {list(structured.keys())} -->")

    return "\n\n".join(parts).strip() + "\n"


# ------------------------------
# Pipeline for one PDF
# ------------------------------

class PdfToCleanMarkdownPipeline:
    """
    Runs extraction + saves outputs for ONE pdf.

    This stage:
      - Extracts Docling structured JSON
      - Exports raw MD for review
      - Removes repetitive headers/footers to produce a cleaned MD
      - Writes audit/reject files
    """

    def __init__(self, cfg: PipelineConfig):
        self.cfg = cfg
        ensure_dir(cfg.output_dir)

    def run_one(self, pdf_path: Path) -> Tuple[int, Dict[str, Any]]:
        """
        Returns (exit_code, summary)
        exit_code: 0 PASS, 1 FAIL, 2 extraction error
        """
        t0 = time.time()
        doc_id = compute_doc_id(pdf_path)
        out_structured = self.cfg.output_dir / f"{doc_id}.structured.json"
        out_raw_md = self.cfg.output_dir / f"{doc_id}.raw.md"
        out_clean_md = self.cfg.output_dir / f"{doc_id}.clean.md"
        out_audit = self.cfg.output_dir / f"{doc_id}.audit.json"
        out_reject = self.cfg.output_dir / f"{doc_id}.reject.json"

        summary: Dict[str, Any] = {
            "doc_id": doc_id,
            "pdf": str(pdf_path),
            "status": "UNKNOWN",
            "seconds": None,
            "error": None,
            "violations": [],
        }

        # ---- Stage 1: Docling extraction (timeout-safe)
        result = run_with_timeout(
            pdf_path=pdf_path,
            allow_tables=self.cfg.allow_tables,
            allow_figures=self.cfg.allow_figures,
            timeout_seconds=self.cfg.timeout_seconds,
        )

        if not result.get("ok"):
            summary["status"] = "EXTRACTION_ERROR"
            summary["seconds"] = round(time.time() - t0, 3)
            summary["error"] = result.get("error", "Unknown extraction error")

            reject = {
                "doc_id": doc_id,
                "mode": self.cfg.mode,
                "stage": "extraction",
                "error": summary["error"],
                "traceback": result.get("traceback", ""),
                "recommendation": "Check if PDF is scanned/corrupted. Consider OCR; re-run with longer timeout; or isolate this PDF.",
            }
            write_json(out_reject, reject)
            return 2, summary

        structured = result.get("structured", {})
        raw_md = result.get("raw_markdown", "")

        # Save structured + raw
        try:
            write_json(out_structured, structured)
        except Exception as e:
            summary["status"] = "FAIL"
            summary["seconds"] = round(time.time() - t0, 3)
            summary["error"] = f"Failed to write structured JSON: {type(e).__name__}: {e}"
            write_json(out_reject, {
                "doc_id": doc_id,
                "mode": self.cfg.mode,
                "stage": "io",
                "error": summary["error"],
                "recommendation": "Check disk permissions/path length.",
            })
            return 1, summary

        try:
            write_text(out_raw_md, raw_md)
        except Exception as e:
            summary["status"] = "FAIL"
            summary["seconds"] = round(time.time() - t0, 3)
            summary["error"] = f"Failed to write raw markdown: {type(e).__name__}: {e}"
            write_json(out_reject, {
                "doc_id": doc_id,
                "mode": self.cfg.mode,
                "stage": "io",
                "error": summary["error"],
                "recommendation": "Check disk permissions/path length.",
            })
            return 1, summary

        # ---- Stage 2: Deterministic header/footer cleanup
        clean_md, hf_audit = remove_repeated_headers_footers(raw_md, min_occurrences=6)

        # Minimal validation for this stage:
        # - We do NOT enforce canonical roles here.
        # - We only ensure we didn't produce empty content.
        violations: List[str] = []
        if len(clean_md.strip()) < 200:
            violations.append("Clean markdown too short (possible extraction failure or OCR empty).")

        # Optional: require references later, but for now default is false.
        if self.cfg.require_references:
            # crude check; your later stages should implement robust checks
            if re.search(r"^\s*##\s*references\s*$", clean_md, flags=re.IGNORECASE | re.MULTILINE):
                # has heading; verify non-empty after it
                parts = re.split(r"^\s*##\s*references\s*$", clean_md, flags=re.IGNORECASE | re.MULTILINE)
                if len(parts) >= 2 and len(parts[-1].strip()) < 200:
                    violations.append("References heading exists but content appears empty.")
            else:
                violations.append("References section missing (require_references=True).")

        # Write outputs
        write_text(out_clean_md, clean_md)

        audit = {
            "doc_id": doc_id,
            "pdf": str(pdf_path),
            "timestamp_utc": utc_now_iso(),
            "docling_seconds": result.get("seconds"),
            "rules_applied": [
                "remove_repeated_headers_footers",
                "remove_page_number_lines",
            ],
            "header_footer_audit": hf_audit,
            "validation": {
                "mode": self.cfg.mode,
                "violations": violations,
                "status": "PASS" if not violations else ("FAIL" if self.cfg.mode == "strict" else "WARN"),
            },
        }
        write_json(out_audit, audit)

        summary["seconds"] = round(time.time() - t0, 3)
        summary["violations"] = violations

        if violations and self.cfg.mode == "strict":
            summary["status"] = "FAIL"
            write_json(out_reject, {
                "doc_id": doc_id,
                "mode": self.cfg.mode,
                "stage": "validation",
                "violations": violations,
                "recommendation": "Inspect .structured.json and .raw.md. Consider increasing timeout or isolating OCR-heavy PDFs.",
                "evidence_excerpt": clean_md[:2000],
            })
            return 1, summary

        summary["status"] = "PASS" if not violations else "WARN"
        return 0, summary


# ------------------------------
# Batch extractor (your requested runner style)
# ------------------------------

class DoclingBatchExtractor:
    """
    Batch extractor that:
      - loops PDFs in input_dir
      - runs Docling extraction in a timeout-safe subprocess
      - writes:
          structured JSON -> out_structured_dir
          raw MD         -> out_markdown_dir
          clean MD        -> out_markdown_dir
          audit/reject    -> out_structured_dir (default)
      - logs run results in a CSV
    """

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
        require_references: bool = False,  # you said at this step you WANT references; so don't require
        mode: str = "strict",
    ):
        self.input_dir = input_dir
        self.out_structured_dir = out_structured_dir
        self.out_markdown_dir = out_markdown_dir
        self.log_csv_path = log_csv_path

        self.timeout_seconds = timeout_seconds
        self.max_attempts = max_attempts
        self.allow_tables = allow_tables
        self.allow_figures = allow_figures
        self.require_references = require_references
        self.mode = mode

        ensure_dir(self.out_structured_dir)
        ensure_dir(self.out_markdown_dir)
        ensure_dir(self.log_csv_path.parent)

        self.pipeline = PdfToCleanMarkdownPipeline(
            PipelineConfig(
                output_dir=self.out_structured_dir,
                require_references=self.require_references,
                allow_figures=self.allow_figures,
                allow_tables=self.allow_tables,
                mode=self.mode,
                timeout_seconds=self.timeout_seconds,
                max_attempts=self.max_attempts,
            )
        )

    def _relocate_markdowns(self, doc_id: str) -> None:
        """
        Pipeline writes to out_structured_dir.
        We move *.raw.md and *.clean.md to out_markdown_dir for your folder layout.
        """
        raw_md = self.out_structured_dir / f"{doc_id}.raw.md"
        clean_md = self.out_structured_dir / f"{doc_id}.clean.md"

        if raw_md.exists():
            raw_md.replace(self.out_markdown_dir / raw_md.name)
        if clean_md.exists():
            clean_md.replace(self.out_markdown_dir / clean_md.name)

    def run(self) -> None:
        pdfs = sorted([p for p in self.input_dir.glob("*.pdf") if p.is_file()])

        if not pdfs:
            print(f"No PDFs found in: {self.input_dir}")
            return

        with self.log_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "pdf_original_name",
                    "doc_id",
                    "status",
                    "attempt",
                    "seconds",
                    "error_or_notes",
                ],
            )
            writer.writeheader()

            for i, pdf in enumerate(pdfs, start=1):
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] ({i}/{len(pdfs)}) Processing: {pdf.name}")

                last_summary: Dict[str, Any] = {}
                last_code: int = 2

                for attempt in range(1, self.max_attempts + 1):
                    print(f"  -> attempt {attempt}/{self.max_attempts}")

                    code, summary = self.pipeline.run_one(pdf)
                    last_code, last_summary = code, summary

                    # move md outputs if we have a doc_id
                    doc_id = summary.get("doc_id") or ""
                    if doc_id:
                        try:
                            self._relocate_markdowns(doc_id)
                        except Exception as e:
                            print(f"  !! Could not move markdown files: {type(e).__name__}: {e}")

                    # If PASS or WARN, stop retrying.
                    if summary.get("status") in ("PASS", "WARN"):
                        break

                    # If extraction error/timeout, retry might help (OCR etc.)
                    print(f"  !! status={summary.get('status')} error={summary.get('error')}")
                    if attempt < self.max_attempts:
                        time.sleep(1.0)

                # Log to CSV
                writer.writerow(
                    {
                        "timestamp": utc_now_iso(),
                        "pdf_original_name": pdf.name,
                        "doc_id": last_summary.get("doc_id"),
                        "status": last_summary.get("status"),
                        "attempt": attempt,
                        "seconds": last_summary.get("seconds"),
                        "error_or_notes": last_summary.get("error") or f"violations={last_summary.get('violations','')}",
                    }
                )
                f.flush()

                # Live feedback
                if last_summary.get("status") == "PASS":
                    print(f"  -> PASS: {last_summary.get('doc_id')}")
                elif last_summary.get("status") == "WARN":
                    print(f"  -> WARN: {last_summary.get('doc_id')} violations={last_summary.get('violations')}")
                else:
                    print(f"  -> FAIL/ERROR: {last_summary.get('doc_id')} ({last_summary.get('status')})")

        print(f"\nDone. Log: {self.log_csv_path}")


# ------------------------------
# Your requested __main__ runner
# ------------------------------

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
        timeout_seconds=120,   # e.g., 2 minutes per PDF
        max_attempts=1,        # retry once
        allow_tables=True,
        allow_figures=True,
        require_references=False,  # keep references if present; don't fail if missing/empty in this stage
        mode="strict",
    )
    extractor.run()
