"""
Docling batch extractor (Step 1)
================================
Goal:
- Loop through PDFs in:  data\leveraging_docling\1_raw_pdfs
- For each PDF:
  1) Convert using Docling
  2) Build a CLEANED markdown (header/footer removed) using Docling structure
  3) Save ONE JSON per paper that includes:
       - docling document structure (including tables)
       - cleaned_text (plain)
       - cleaned_markdown (markdown)
  4) Save a standalone .md for quick review
  5) Log runs + errors while running
  6) Maintain a CSV mapping original filename -> new safe name (slug + hash)

Outputs:
- data\leveraging_docling\2_docling_output\{safe_name}.json
- data\leveraging_docling\3_docling_markdown\{safe_name}.md
- data\leveraging_docling\2_docling_output\docling_run_log.csv

Notes:
- We do NOT rename your PDFs on disk (option 2 approach). We only generate safe output names.
- Tables ARE kept in the JSON (full docling structure + a markdown rendering).
- Images are NOT extracted as pixels; we keep captions/nearby text placeholders only.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Docling imports
from docling.document_converter import DocumentConverter


# -----------------------------
# Small utilities
# -----------------------------

def now_iso() -> str:
    """Return local timestamp string suitable for logs."""
    return time.strftime("%Y-%m-%d %H:%M:%S")


def sha1_8_of_file(file_path: Path) -> str:
    """
    Hash file bytes to create a stable ID for naming outputs.
    - SHA1 is fine here (non-security use), short and stable.
    - 8 hex chars gives low collision risk for 420 PDFs.
    """
    h = hashlib.sha1()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()[:8]


def normalize_to_ascii(text: str) -> str:
    """
    Replace non-ASCII characters with closest ASCII equivalents.
    Example: 'björling' -> 'bjorling'
    """
    # NFKD splits accents from base letters; then we drop non-ascii.
    return (
        unicodedata.normalize("NFKD", text)
        .encode("ascii", "ignore")
        .decode("ascii")
    )


def slugify_filename(stem: str, max_len: int = 80) -> str:
    """
    Turn a filename stem into a safe slug:
    - Lowercase
    - Replace accents
    - Replace spaces with hyphen
    - Keep [a-z0-9_-] only
    - Collapse multiple hyphens
    - Truncate to max_len (without losing readability too much)
    """
    s = normalize_to_ascii(stem).lower()
    s = s.replace("&", " and ")
    s = re.sub(r"\s+", "-", s.strip())
    s = re.sub(r"[^a-z0-9_-]+", "-", s)
    s = re.sub(r"-{2,}", "-", s).strip("-_")
    if len(s) > max_len:
        s = s[:max_len].rstrip("-_")
    return s or "paper"


# -----------------------------
# Docling structure helpers
# -----------------------------

def build_selfref_index(doc: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Create a mapping: self_ref -> element dict
    for texts, tables, pictures, groups, etc.
    """
    idx: Dict[str, Dict[str, Any]] = {}

    # Common keys in DoclingDocument dict
    for bucket in ("texts", "tables", "pictures", "groups", "key_value_items", "form_items"):
        items = doc.get(bucket, [])
        if isinstance(items, list):
            for it in items:
                sr = it.get("self_ref")
                if sr:
                    idx[sr] = it

    # Also index root containers if present
    for root_key in ("body", "furniture"):
        if isinstance(doc.get(root_key), dict) and doc[root_key].get("self_ref"):
            idx[doc[root_key]["self_ref"]] = doc[root_key]

    return idx


def ref_to_selfref(ref_obj: Any) -> Optional[str]:
    """
    Convert {"$ref": "#/texts/12"} style refs into "#/texts/12"
    """
    if isinstance(ref_obj, dict) and "$ref" in ref_obj:
        return ref_obj["$ref"]
    return None


def get_bbox_info(el: Dict[str, Any]) -> Optional[Tuple[int, float, float, float, float, str]]:
    """
    Return (page_no, l, t, r, b, coord_origin) from the first provenance entry if present.
    Many Docling elements have el["prov"][0]["bbox"].
    """
    prov = el.get("prov")
    if not prov or not isinstance(prov, list):
        return None
    p0 = prov[0]
    bbox = p0.get("bbox")
    if not bbox:
        return None
    page_no = p0.get("page_no", -1)
    l = float(bbox.get("l", 0.0))
    t = float(bbox.get("t", 0.0))
    r = float(bbox.get("r", 0.0))
    b = float(bbox.get("b", 0.0))
    origin = str(bbox.get("coord_origin", "BOTTOMLEFT"))
    return (int(page_no), l, t, r, b, origin)


def bbox_y_top_normalized(page_height: float, t: float, b: float, origin: str) -> float:
    """
    We want a consistent notion of "distance from top".
    Docling sometimes uses coord_origin = BOTTOMLEFT or TOPLEFT.
    - If TOPLEFT: smaller 't' means closer to top already.
    - If BOTTOMLEFT: larger 't' is closer to top; convert using page_height - t.
    We approximate page_height from doc["pages"] when possible.
    """
    if origin.upper() == "TOPLEFT":
        return t
    # BOTTOMLEFT
    return page_height - t


def bbox_y_bottom_normalized(page_height: float, t: float, b: float, origin: str) -> float:
    """
    Consistent notion of "distance from bottom".
    - TOPLEFT: bottom is (page_height - b)
    - BOTTOMLEFT: bottom is b
    """
    if origin.upper() == "TOPLEFT":
        return page_height - b
    # BOTTOMLEFT
    return b


def looks_like_page_number(text: str) -> bool:
    s = text.strip()
    # patterns: "11 / 22", "11/22", "Page 11", "11"
    if re.fullmatch(r"\d+\s*/\s*\d+", s):
        return True
    if re.fullmatch(r"page\s*\d+(\s*of\s*\d+)?", s, flags=re.IGNORECASE):
        return True
    if re.fullmatch(r"\d{1,3}", s):
        return True
    return False


# -----------------------------
# Cleaning + Markdown rendering
# -----------------------------

@dataclass
class CleanConfig:
    """
    Tuning knobs for header/footer removal.
    """
    # If an element is within top X% of page, treat as potential header
    header_zone_ratio: float = 0.12
    # If an element is within bottom X% of page, treat as potential footer
    footer_zone_ratio: float = 0.12
    # Require the same text to appear on at least this many pages to be considered a repeating header/footer
    repeat_min_pages: int = 3
    # If content_layer == furniture or label is page_header/page_footer => drop
    drop_furniture_labels: Tuple[str, ...] = ("page_header", "page_footer", "watermark")


class DoclingCleaner:
    """
    Builds a "clean" reading markdown/text from Docling structure
    while removing header/footer using:
      1) Docling furniture annotations (content_layer == 'furniture')
      2) label-based filter (page_header/page_footer/watermark)
      3) geometry-based repeated strings near page top/bottom (captures cases where footer leaks into body)
    """

    def __init__(self, cfg: CleanConfig):
        self.cfg = cfg

    def _page_heights(self, doc: Dict[str, Any]) -> Dict[int, float]:
        """
        Build page_no -> page_height. If missing, fall back to 800.
        """
        out: Dict[int, float] = {}
        pages = doc.get("pages", [])
        if isinstance(pages, list):
            for p in pages:
                # docling "pages" objects vary; we try common shapes
                pno = p.get("page_no") or p.get("page") or p.get("no")
                if pno is None:
                    continue
                # try height
                h = None
                if isinstance(p.get("size"), dict):
                    h = p["size"].get("h") or p["size"].get("height")
                if h is None:
                    h = p.get("height")
                if h is None:
                    # fallback: many PDFs around ~792 points height for letter
                    h = 800.0
                out[int(pno)] = float(h)
        return out

    def _collect_text_elements(self, doc: Dict[str, Any]) -> List[Dict[str, Any]]:
        texts = doc.get("texts", [])
        if isinstance(texts, list):
            return texts
        return []

    def _is_furniture(self, el: Dict[str, Any]) -> bool:
        if el.get("content_layer") == "furniture":
            return True
        if (el.get("label") or "").lower() in self.cfg.drop_furniture_labels:
            return True
        return False

    def _build_repeating_top_bottom_set(self, doc: Dict[str, Any]) -> set:
        """
        Identify repeated strings that appear near top/bottom across many pages.
        This catches cases where Docling didn't label them as 'furniture'.
        """
        page_heights = self._page_heights(doc)
        texts = self._collect_text_elements(doc)

        # Track occurrences: normalized_text -> set(page_no)
        top_hits: Dict[str, set] = {}
        bottom_hits: Dict[str, set] = {}

        for el in texts:
            if not isinstance(el, dict):
                continue
            # Ignore already-labeled furniture; we only want "leaked" furniture in body.
            if self._is_furniture(el):
                continue

            txt = (el.get("text") or "").strip()
            if not txt:
                continue

            bbox_info = get_bbox_info(el)
            if not bbox_info:
                continue
            page_no, l, t, r, b, origin = bbox_info
            if page_no <= 0:
                continue

            h = page_heights.get(page_no, 800.0)

            y_top = bbox_y_top_normalized(h, t, b, origin)          # distance from top
            y_bottom = bbox_y_bottom_normalized(h, t, b, origin)     # distance from bottom

            # Convert to ratio
            top_ratio = y_top / h
            bottom_ratio = y_bottom / h

            # Normalize key (collapse spaces) for repeat detection
            key = re.sub(r"\s+", " ", txt).strip()

            if top_ratio <= self.cfg.header_zone_ratio:
                top_hits.setdefault(key, set()).add(page_no)
            if bottom_ratio <= self.cfg.footer_zone_ratio:
                bottom_hits.setdefault(key, set()).add(page_no)

        repeating = set()
        for k, pages in top_hits.items():
            if len(pages) >= self.cfg.repeat_min_pages:
                repeating.add(k)
        for k, pages in bottom_hits.items():
            if len(pages) >= self.cfg.repeat_min_pages:
                repeating.add(k)

        return repeating

    def _should_drop_element(self, el: Dict[str, Any], repeating: set, page_heights: Dict[int, float]) -> bool:
        """
        Decide whether to drop a text element.
        """
        # 1) Explicit furniture
        if self._is_furniture(el):
            return True

        txt = (el.get("text") or "").strip()
        if not txt:
            return True

        # 2) Repeating header/footer string
        key = re.sub(r"\s+", " ", txt).strip()
        if key in repeating:
            return True

        # 3) Pure page number (very common footer leak)
        if looks_like_page_number(txt):
            bbox_info = get_bbox_info(el)
            if bbox_info:
                page_no, l, t, r, b, origin = bbox_info
                h = page_heights.get(page_no, 800.0)
                y_bottom = bbox_y_bottom_normalized(h, t, b, origin)
                bottom_ratio = y_bottom / h
                if bottom_ratio <= self.cfg.footer_zone_ratio:
                    return True

        return False

    def _table_to_markdown(self, table_el: Dict[str, Any]) -> str:
        """
        Render a Docling table element to a markdown table.
        Docling table structure varies; in your sample JSON, table cells appear in table_el["data"]["grid"].
        We implement a robust renderer for common shapes.
        """
        # Try common structures
        grid = None
        data = table_el.get("data")
        if isinstance(data, dict):
            grid = data.get("grid") or data.get("cells")

        # Fallback: sometimes "grid" sits directly
        if grid is None:
            grid = table_el.get("grid")

        if not grid:
            # As a safe fallback, dump a short JSON snippet so you can still inspect content
            short = json.dumps(table_el.get("data", {}) or {}, ensure_ascii=False)[:2000]
            return f"\n\n> [TABLE: could not render to markdown]\n>\n> {short}\n"

        # If grid is list of rows, each row list of cells
        rows: List[List[str]] = []

        if isinstance(grid, list):
            for row in grid:
                if isinstance(row, list):
                    out_row = []
                    for cell in row:
                        if isinstance(cell, dict):
                            out_row.append(str(cell.get("text", "")).strip())
                        else:
                            out_row.append(str(cell).strip())
                    rows.append(out_row)

        # Normalize row lengths
        max_cols = max((len(r) for r in rows), default=0)
        rows = [r + [""] * (max_cols - len(r)) for r in rows]

        if not rows:
            return "\n\n> [TABLE: empty]\n"

        # Use first row as header if it looks like headers exist
        header = rows[0]
        sep = ["---"] * len(header)
        body = rows[1:] if len(rows) > 1 else []

        md = "\n\n"
        md += "| " + " | ".join(header) + " |\n"
        md += "| " + " | ".join(sep) + " |\n"
        for r in body:
            md += "| " + " | ".join(r) + " |\n"
        return md

    def build_clean_markdown_and_text(self, doc: Dict[str, Any]) -> Tuple[str, str]:
        """
        Build:
          - cleaned_markdown (with headings + tables placeholders)
          - cleaned_text (plain text version)
        based on doc["body"]["children"] ordering.
        """
        idx = build_selfref_index(doc)
        page_heights = self._page_heights(doc)
        repeating = self._build_repeating_top_bottom_set(doc)

        body = doc.get("body", {})
        children = body.get("children", []) if isinstance(body, dict) else []

        md_parts: List[str] = []
        txt_parts: List[str] = []

        def add_paragraph(s: str):
            s = s.strip()
            if not s:
                return
            # Avoid insane horizontal rules / repeated hyphens like in your Selamet sample
            if re.fullmatch(r"[-–—]{20,}", s):
                return
            md_parts.append(s)
            txt_parts.append(s)

        def add_heading(s: str, level: int = 2):
            s = s.strip()
            if not s:
                return
            level = max(1, min(level, 6))
            md_parts.append("#" * level + " " + s)
            txt_parts.append(s)

        def add_table(md_table: str):
            md_parts.append(md_table.strip("\n"))
            # For text, keep a simplified placeholder to avoid huge tokens later
            txt_parts.append("[TABLE]")

        def add_figure_placeholder(caption: str):
            caption = caption.strip()
            if caption:
                md_parts.append(f"\n\n<!-- figure -->\n\n*{caption}*")
                txt_parts.append(f"[FIGURE] {caption}")
            else:
                md_parts.append("\n\n<!-- figure -->")
                txt_parts.append("[FIGURE]")

        # Iterate document body in order
        for ch in children:
            sr = ref_to_selfref(ch)
            if not sr:
                continue
            el = idx.get(sr)
            if not el:
                continue

            # TEXT elements
            if sr.startswith("#/texts/"):
                if self._should_drop_element(el, repeating, page_heights):
                    continue
                label = (el.get("label") or "").lower()
                t = el.get("text") or ""

                # Heuristic formatting by label
                if label in ("title",):
                    add_heading(t, level=1)
                elif label in ("section_header", "heading"):
                    add_heading(t, level=2)
                elif label in ("caption",):
                    # captions are useful (especially when you can't send images)
                    add_paragraph(f"*{t.strip()}*")
                else:
                    add_paragraph(t)

            # TABLE elements
            elif sr.startswith("#/tables/"):
                # Keep tables; they are usually in body layer already
                md_table = self._table_to_markdown(el)
                add_table(md_table)

            # PICTURE elements
            elif sr.startswith("#/pictures/"):
                # We can't send images to GPT (your choice), but we can keep captions
                # Collect caption-like texts under this picture
                caption_texts: List[str] = []
                pic_children = el.get("children", [])
                for pc in pic_children if isinstance(pic_children, list) else []:
                    psr = ref_to_selfref(pc)
                    if not psr:
                        continue
                    tel = idx.get(psr)
                    if not tel or not isinstance(tel, dict):
                        continue
                    if (tel.get("label") or "").lower() == "caption":
                        cap = (tel.get("text") or "").strip()
                        if cap:
                            caption_texts.append(cap)

                caption = " ".join(caption_texts).strip()
                add_figure_placeholder(caption)

            # GROUPS etc. (skip; their children are already in body order)
            else:
                continue

        cleaned_markdown = "\n\n".join([p for p in md_parts if p.strip()])
        cleaned_text = "\n\n".join([p for p in txt_parts if p.strip()])

        # Final cleanup: remove repeated blank lines
        cleaned_markdown = re.sub(r"\n{3,}", "\n\n", cleaned_markdown).strip() + "\n"
        cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text).strip() + "\n"

        return cleaned_markdown, cleaned_text


# -----------------------------
# Main runner (OOP)
# -----------------------------

@dataclass
class PathsConfig:
    input_dir: Path
    out_json_dir: Path
    out_md_dir: Path
    log_csv_path: Path


class DoclingBatchExtractor:
    """
    Batch processor:
    - Reads PDFs
    - Converts with Docling
    - Produces per-paper JSON + MD + CSV log mapping
    """

    def __init__(self, paths: PathsConfig, cleaner: DoclingCleaner):
        self.paths = paths
        self.cleaner = cleaner
        self.converter = DocumentConverter()

        # Ensure output dirs exist
        self.paths.out_json_dir.mkdir(parents=True, exist_ok=True)
        self.paths.out_md_dir.mkdir(parents=True, exist_ok=True)

        # Initialize CSV log if missing
        if not self.paths.log_csv_path.exists():
            with self.paths.log_csv_path.open("w", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(
                    f,
                    fieldnames=[
                        "timestamp",
                        "status",
                        "source_pdf",
                        "safe_name",
                        "pdf_hash8",
                        "json_path",
                        "md_path",
                        "n_pages",
                        "error_type",
                        "error_message",
                        "elapsed_sec",
                    ],
                )
                w.writeheader()

    def _append_log(self, row: Dict[str, Any]) -> None:
        with self.paths.log_csv_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "status",
                    "source_pdf",
                    "safe_name",
                    "pdf_hash8",
                    "json_path",
                    "md_path",
                    "n_pages",
                    "error_type",
                    "error_message",
                    "elapsed_sec",
                ],
            )
            w.writerow(row)

    def _safe_output_name(self, pdf_path: Path) -> Tuple[str, str]:
        """
        Output naming strategy:
        safe_name = {slugified_stem}__{sha1_8}
        """
        h8 = sha1_8_of_file(pdf_path)
        slug = slugify_filename(pdf_path.stem)
        safe_name = f"{slug}__{h8}"
        return safe_name, h8

    def _already_done(self, safe_name: str) -> bool:
        json_path = self.paths.out_json_dir / f"{safe_name}.json"
        md_path = self.paths.out_md_dir / f"{safe_name}.md"
        return json_path.exists() and md_path.exists()

    def run(self, *, overwrite: bool = False) -> None:
        pdfs = sorted(self.paths.input_dir.glob("*.pdf"))
        if not pdfs:
            print(f"[{now_iso()}] No PDFs found in: {self.paths.input_dir}")
            return

        print(f"[{now_iso()}] Found {len(pdfs)} PDFs.")
        print(f"[{now_iso()}] Output JSON dir: {self.paths.out_json_dir}")
        print(f"[{now_iso()}] Output MD dir:   {self.paths.out_md_dir}")
        print(f"[{now_iso()}] Log CSV:         {self.paths.log_csv_path}")
        print("")

        for i, pdf_path in enumerate(pdfs, start=1):
            start = time.time()
            safe_name, h8 = self._safe_output_name(pdf_path)

            json_out = self.paths.out_json_dir / f"{safe_name}.json"
            md_out = self.paths.out_md_dir / f"{safe_name}.md"

            # Skip if already processed
            if not overwrite and self._already_done(safe_name):
                msg = f"[{now_iso()}] ({i}/{len(pdfs)}) SKIP already exists: {pdf_path.name} -> {safe_name}"
                print(msg)
                self._append_log(
                    {
                        "timestamp": now_iso(),
                        "status": "SKIP_EXISTS",
                        "source_pdf": pdf_path.name,
                        "safe_name": safe_name,
                        "pdf_hash8": h8,
                        "json_path": str(json_out),
                        "md_path": str(md_out),
                        "n_pages": "",
                        "error_type": "",
                        "error_message": "",
                        "elapsed_sec": round(time.time() - start, 2),
                    }
                )
                continue

            print(f"[{now_iso()}] ({i}/{len(pdfs)}) Processing: {pdf_path.name}")
            print(f"  -> safe_name: {safe_name}")

            try:
                # 1) Convert with Docling
                result = self.converter.convert(str(pdf_path))
                doc = result.document

                # 2) Export as dict (full structure, includes tables)
                doc_dict = doc.export_to_dict()  # full docling structure

                # 3) Build cleaned markdown/text (removes header/footer aggressively)
                cleaned_md, cleaned_text = self.cleaner.build_clean_markdown_and_text(doc_dict)

                # 4) Assemble ONE JSON per paper including docling + cleaned outputs
                payload = {
                    "paper_uid": safe_name,
                    "source_filename": pdf_path.name,
                    "source_path": str(pdf_path),
                    "processed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "docling_output_keys": list(doc_dict.keys()) if isinstance(doc_dict, dict) else [],
                    "cleaning": {
                        "header_zone_ratio": self.cleaner.cfg.header_zone_ratio,
                        "footer_zone_ratio": self.cleaner.cfg.footer_zone_ratio,
                        "repeat_min_pages": self.cleaner.cfg.repeat_min_pages,
                        "dropped_labels": list(self.cleaner.cfg.drop_furniture_labels),
                    },
                    "cleaned_text": cleaned_text,
                    "cleaned_markdown": cleaned_md,
                    # Keep full structure so you can extract tables later:
                    "doc": doc_dict,
                }

                # Determine pages count
                n_pages = ""
                try:
                    pages = doc_dict.get("pages", [])
                    n_pages = len(pages) if isinstance(pages, list) else ""
                except Exception:
                    n_pages = ""

                # 5) Write outputs
                json_out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                md_out.write_text(cleaned_md, encoding="utf-8")

                elapsed = round(time.time() - start, 2)
                print(f"  ✅ Done in {elapsed}s")
                print(f"  JSON: {json_out}")
                print(f"  MD:   {md_out}")
                print("")

                self._append_log(
                    {
                        "timestamp": now_iso(),
                        "status": "OK",
                        "source_pdf": pdf_path.name,
                        "safe_name": safe_name,
                        "pdf_hash8": h8,
                        "json_path": str(json_out),
                        "md_path": str(md_out),
                        "n_pages": n_pages,
                        "error_type": "",
                        "error_message": "",
                        "elapsed_sec": elapsed,
                    }
                )

            except Exception as e:
                elapsed = round(time.time() - start, 2)
                err_type = type(e).__name__
                err_msg = str(e)

                print(f"  ❌ ERROR ({err_type}) after {elapsed}s")
                print(f"  Message: {err_msg}")
                print("")

                self._append_log(
                    {
                        "timestamp": now_iso(),
                        "status": "ERROR",
                        "source_pdf": pdf_path.name,
                        "safe_name": safe_name,
                        "pdf_hash8": h8,
                        "json_path": str(json_out),
                        "md_path": str(md_out),
                        "n_pages": "",
                        "error_type": err_type,
                        "error_message": err_msg,
                        "elapsed_sec": elapsed,
                    }
                )


# -----------------------------
# CLI entry point
# -----------------------------

def main() -> int:
    """
    Usage:
      python docling_step1_extract.py
      python docling_step1_extract.py --overwrite

    You can also adjust cleaning thresholds in CleanConfig below.
    """
    overwrite = "--overwrite" in sys.argv

    base = Path("data") / "leveraging_docling"
    paths = PathsConfig(
        input_dir=base / "1_raw_pdfs",
        out_json_dir=base / "2_docling_output",
        out_md_dir=base / "3_docling_markdown",
        log_csv_path=(base / "2_docling_output" / "docling_run_log.csv"),
    )

    cleaner = DoclingCleaner(
        CleanConfig(
            header_zone_ratio=0.12,
            footer_zone_ratio=0.12,
            repeat_min_pages=3,
            drop_furniture_labels=("page_header", "page_footer", "watermark"),
        )
    )

    runner = DoclingBatchExtractor(paths=paths, cleaner=cleaner)
    runner.run(overwrite=overwrite)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
