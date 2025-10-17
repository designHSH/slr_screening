# src/clean_pdf_pipeline.py
# Requires: pip install pymupdf

import re
from pathlib import Path
from collections import Counter
import fitz  # PyMuPDF

# --------- CONFIG ---------
INPUT_PDF_DIR = Path("data/test_data/files_pdf_to_txt/input")          # folder with PDFs
OUTPUT_TXT_DIR = Path("data/test_data/files_pdf_to_txt/output_re_cleaning")  # cleaned txt output
OUTPUT_TXT_DIR.mkdir(parents=True, exist_ok=True)

# Header/footer detection knobs
TOP_LINES_PER_PAGE = 7        # how many top lines to consider as header candidates
BOTTOM_LINES_PER_PAGE = 7     # how many bottom lines to consider as footer candidates
REPEAT_THRESHOLD = 3          # a line must repeat on >= this many pages to be removed
MIN_HEADER_LEN = 5            # ignore super-short tokens
MAX_HEADER_LEN = 140          # ignore very long lines (likely real content)

# ---------------------------------------
# 1) PAGE-BY-PAGE EXTRACTION
# ---------------------------------------
def extract_pages_separately(pdf_path: Path) -> list[str]:
    pages = []
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                txt = page.get_text("text")
                pages.append(txt if txt else "")
    except Exception as e:
        print(f"❌ Error reading {pdf_path.name}: {e}")
    return pages


# ---------------------------------------
# 2) DETECT REPEATED HEADERS/FOOTERS
# ---------------------------------------
_whitespace_collapse = re.compile(r"\s+")
def _normalize_line(line: str) -> str:
    # normalize whitespace and trim
    line = _whitespace_collapse.sub(" ", line).strip()
    return line

def get_candidate_hf_lines(pages: list[str]) -> set[str]:
    """
    Scan top/bottom lines from each page and count repeats.
    Return the set of normalized lines that repeat across pages.
    """
    counter = Counter()
    for page in pages:
        lines = [ln.strip() for ln in page.splitlines() if ln.strip()]
        if not lines:
            continue

        # top candidates
        top = lines[:TOP_LINES_PER_PAGE]
        # bottom candidates
        bottom = lines[-BOTTOM_LINES_PER_PAGE:]

        for ln in (top + bottom):
            norm = _normalize_line(ln)
            if MIN_HEADER_LEN <= len(norm) <= MAX_HEADER_LEN:
                counter[norm] += 1

    repeated = {ln for ln, cnt in counter.items() if cnt >= REPEAT_THRESHOLD}
    return repeated


# ---------------------------------------
# 3) REMOVE HEADERS/FOOTERS FROM PAGES
# ---------------------------------------
# common boilerplate patterns (page numbers etc.)
PAGE_NUMBER_PATTERNS = [
    r"(?i)^page\s*\d+\s*(of\s*\d+)?$",       # "Page 3" or "Page 3 of 12"
    r"^\d+\s*/\s*\d+$",                      # "3/12"
    r"^-\s*\d+\s*-$",                        # "- 5 -"
    r"^\d+$",                                # a bare page number line
]

def line_matches_boilerplate(line: str) -> bool:
    for pat in PAGE_NUMBER_PATTERNS:
        if re.match(pat, line.strip()):
            return True
    return False

def remove_headers_footers_from_pages(pages: list[str], repeated_lines: set[str]) -> list[str]:
    cleaned_pages = []
    for page in pages:
        out_lines = []
        for ln in page.splitlines():
            norm = _normalize_line(ln)
            if norm in repeated_lines:
                continue
            if line_matches_boilerplate(ln):
                continue
            out_lines.append(ln)
        cleaned_pages.append("\n".join(out_lines).strip())
    return cleaned_pages


# ---------------------------------------
# 4) MERGE CLEANED PAGES
# ---------------------------------------
def merge_pages(pages: list[str]) -> str:
    return "\n\n".join(p for p in pages if p)


# ---------------------------------------
# 5) REMOVE BACK MATTER (TRUNCATE) & INLINE AUTHOR INFO (SURGICAL)
# ---------------------------------------

# Group 1 (truncate everything after the earliest match)
BACK_MATTER_HEADINGS = [
    r"(?i)\bReferences\b",
    r"(?i)\bBibliography\b",
    r"(?i)\bAcknowledg(?:e)?ments?\b",
    r"(?i)\bFunding\b",
    r"(?i)\bAppendix\b",
    r"(?i)\bAppendices\b",
    r"(?i)\bSupplementary\s+Material(?:s)?\b",
]

def remove_back_matter(full_text: str) -> str:
    # Find the earliest match of any back-matter heading and truncate there.
    earliest = None
    for pat in BACK_MATTER_HEADINGS:
        m = re.search(pat, full_text)
        if m:
            idx = m.start()
            if earliest is None or idx < earliest:
                earliest = idx
    if earliest is not None:
        return full_text[:earliest].rstrip()
    return full_text


# Group 2 (remove just the matched block/line)
EMAIL_RE = re.compile(r"[\w\.-]+@[\w\.-]+\.\w+")
INLINE_BLOCK_PATTERNS = [
    r"(?im)^\s*Author(?:s)?\s*(Information|Details|Biography|Biographies)\s*:?.*?(?=\n\s*\n|\Z)",
    r"(?im)^\s*About the Author(?:s)?\s*:?.*?(?=\n\s*\n|\Z)",
    r"(?im)^\s*Author(?:s)?(?:’|')?\s*Contribution[s]?\s*:?.*?(?=\n\s*\n|\Z)",
    r"(?im)^\s*Corresponding Author\s*:?.*?(?=\n\s*\n|\Z)",
    r"(?im)^\s*Affiliation(?:s)?\s*:?.*?(?=\n\s*\n|\Z)",
    r"(?im)^\s*Institutional Affiliation\s*:?.*?(?=\n\s*\n|\Z)",
]

AFFILIATION_LINE_PATTERNS = [
    r"(?i)^\s*Department of [A-Za-z0-9 ,\-&]+$",
    r"(?i)^\s*Faculty of [A-Za-z0-9 ,\-&]+$",
    r"(?i)^\s*School of [A-Za-z0-9 ,\-&]+$",
    r"(?i)^\s*College of [A-Za-z0-9 ,\-&]+$",
    r"(?i)^\s*University of [A-Za-z0-9 ,\-&]+$",
    r"(?i)^\s*[A-Za-z0-9 ,\-&]+ University$",
]

def remove_inline_author_info(text: str) -> str:
    # Remove labeled paragraphs/blocks (up to next blank line)
    for pat in INLINE_BLOCK_PATTERNS:
        text = re.sub(pat, "", text, flags=re.DOTALL)

    # Remove affiliation lines
    lines = text.splitlines()
    kept = []
    for ln in lines:
        if any(re.match(p, ln) for p in AFFILIATION_LINE_PATTERNS):
            continue
        # remove email addresses inline
        ln = EMAIL_RE.sub("", ln)
        kept.append(ln)
    return "\n".join(kept).strip()


# ---------------------------------------
# HIGH-LEVEL PIPE
# ---------------------------------------
def process_paper(pdf_path: Path) -> str:
    # 1) extract
    pages = extract_pages_separately(pdf_path)

    # 2) detect headers/footers
    repeated = get_candidate_hf_lines(pages)

    # 3) remove headers/footers
    cleaned_pages = remove_headers_footers_from_pages(pages, repeated)

    # 4) merge
    merged = merge_pages(cleaned_pages)

    # 5) back-matter truncate
    truncated = remove_back_matter(merged)

    # 6) inline author/affiliation/email cleanup
    final_text = remove_inline_author_info(truncated)

    return final_text


def main():
    pdfs = sorted(INPUT_PDF_DIR.glob("*.pdf"))
    if not pdfs:
        print(f"⚠️ No PDFs found in {INPUT_PDF_DIR}")
        return

    print(f"🚀 Processing {len(pdfs)} PDFs from: {INPUT_PDF_DIR}")
    for i, pdf in enumerate(pdfs, 1):
        try:
            cleaned = process_paper(pdf)
            out_path = OUTPUT_TXT_DIR / f"{pdf.stem}.txt"
            out_path.write_text(cleaned, encoding="utf-8")
            print(f"✅ [{i}/{len(pdfs)}] Saved: {out_path.name}  (chars: {len(cleaned)})")
        except Exception as e:
            print(f"❌ [{i}/{len(pdfs)}] {pdf.name}: {e}")

    print(f"\n🎯 Done! Cleaned texts in: {OUTPUT_TXT_DIR}")


if __name__ == "__main__":
    main()
