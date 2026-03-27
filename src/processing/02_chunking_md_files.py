#!/usr/bin/env python3
"""
md_dir_to_chunked_json.py

Batch-convert a directory of CONTROLLED MARKDOWN (.md) files (section-extraction outputs)
into per-paper JSON files suitable for the next pipeline step (paragraph-level LLM calls).

Key features (aligned with your ID policy + quote_id metadata extraction logic):
- paper_key is generated from filename using the same author/year extraction logic used in your quote pipeline:
    PaperKey = AuthorTitleCase_Year
  Example: addamiano + 2024 -> Addamiano_2024

- Section IDs are deterministic from markdown heading levels and encounter order:
    S1, S1.1, S1.1.1, ...
  plus S0 for any content appearing before the first heading.

- Units are typed and filterable:
    paragraph -> P001, P002, ...
    table     -> T001, T002, ...
    figure    -> F001, F002, ...
  unit_id format:
    PaperKey::SectionID::UnitID
  Example:
    Addamiano_2024::S0::P001

Input assumptions (your controlled markdown):
- Title line:   "# PAPER TITLE: ..."
- Authors line: "**Authors:** ..."
- Headings: ## / ### / #### (and possibly deeper)
- Tables:  [TABLE] ... [/TABLE]
- Figures: [FIGURE] ... [/FIGURE]
- Paragraphs separated by blank lines

Usage:
  python md_dir_to_chunked_json.py --input_dir /path/to/mds --output_dir /path/to/jsons

Optional:
  --context_window 1        # store prev/next paragraph IDs within section for each paragraph unit
  --output_name_mode key    # output file name uses paper_key.json (default: "stem")
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ----------------------------
# Regex / markers (Controlled Markdown)
# ----------------------------

HEADING_RE = re.compile(r"^(#{2,6})\s+(.*\S)\s*$")
TITLE_RE = re.compile(r"^#\s+PAPER TITLE:\s*(.*\S)\s*$")
AUTHORS_RE = re.compile(r"^\*\*Authors:\*\*\s*(.*\S)\s*$")

TABLE_START = "[TABLE]"
TABLE_END = "[/TABLE]"
FIG_START = "[FIGURE]"
FIG_END = "[/FIGURE]"


# ----------------------------
# Filename metadata extraction (same logic as your quote pipeline)
# ----------------------------

def extract_metadata(filename: str) -> Tuple[str, str]:
    """Extracts author surname and year for deterministic IDs (from filename stem)."""
    base = Path(filename).stem
    parts = [p.strip() for p in base.split("_") if p.strip()]
    if len(parts) < 2:
        return "unknown", "0000"

    author_part = parts[0]
    author_part = re.sub(r"\s+et al\.\s*$", "", author_part, flags=re.IGNORECASE).strip()
    author_part = re.split(r"\s+and\s+", author_part, maxsplit=1, flags=re.IGNORECASE)[0].strip()

    year = "0000"
    remainder = "_".join(parts[1:])
    year_match = re.search(r"(?<!\d)((?:19|20)\d{2})(?!\d)", remainder)
    if year_match:
        year = year_match.group(1)

    # Preserve compound surnames/particles in source name; normalize for ID safety.
    surname = re.sub(r"[^a-z0-9\-]+", "", author_part.lower().replace(" ", "-"))
    return surname or "unknown", year


def paper_key_from_filename(filename: str) -> str:
    """
    Convert extracted (author, year) into TitleCase_HHHH format for unit IDs.
    Example: "alppay-bayazit", "2015" -> "Alppay-Bayazit_2015"
    """
    author, year = extract_metadata(filename)
    author_tc = "-".join((p[:1].upper() + p[1:]) for p in author.split("-") if p)
    return f"{author_tc}_{year}"


# ----------------------------
# Utilities
# ----------------------------

def normalize_ws_keep_lines(s: str) -> str:
    """Keep original line breaks; strip trailing spaces per line; preserve content."""
    return "\n".join(line.rstrip() for line in s.splitlines()).strip("\n")


def is_blank(line: str) -> bool:
    return line.strip() == ""


@dataclass
class Unit:
    unit_id: str
    unit_type: str  # "paragraph" | "table" | "figure"
    order_in_section: int
    raw_text: str
    caption: Optional[str] = None


@dataclass
class Section:
    section_id: str
    heading_text: Optional[str]
    heading_level: int
    order_in_document: int
    units: List[Unit]


# ----------------------------
# Deterministic section numbering by heading depth
# ----------------------------

class SectionNumbering:
    """
    Deterministic hierarchical numbering based purely on heading depth and encounter order.

    ##    -> S1, S2, ...
    ###   -> S1.1, S1.2, ...
    ####  -> S1.1.1, ...
    ##### -> S1.1.1.1, ...
    """

    def __init__(self) -> None:
        self.counters: Dict[int, int] = {}

    def next_id(self, level: int) -> str:
        # reset deeper levels
        for lvl in sorted(list(self.counters.keys())):
            if lvl > level:
                del self.counters[lvl]

        # increment this level
        self.counters[level] = self.counters.get(level, 0) + 1

        # Build numeric path using existing counters up to this level.
        base_level = 2 if 2 in self.counters else min(self.counters.keys())
        parts = [self.counters[base_level]]
        for lvl in sorted(self.counters.keys()):
            if lvl <= base_level:
                continue
            if lvl <= level:
                parts.append(self.counters[lvl])

        return "S" + ".".join(str(p) for p in parts)


def make_unit_id(paper_key: str, section_id: str, prefix: str, idx: int) -> str:
    return f"{paper_key}::{section_id}::{prefix}{idx:03d}"


# ----------------------------
# Core: parse one markdown file into chunked JSON
# ----------------------------

def parse_markdown_to_json(md_text: str, source_filename: str, context_window: int = 0) -> Dict:
    paper_key = paper_key_from_filename(source_filename)

    lines = md_text.splitlines()

    title: Optional[str] = None
    authors: Optional[str] = None

    sec_num = SectionNumbering()
    sections: List[Section] = []
    current_section: Optional[Section] = None

    # Track per-section unit counters
    unit_counts: Dict[str, Dict[str, int]] = {}     # section_id -> {"P": n, "T": n, "F": n}
    unit_order: Dict[str, int] = {}                 # section_id -> order counter

    section_order = -1  # will become 0..; S0 will be inserted at 0 if needed
    preamble_paras: List[str] = []

    def ensure_section(section_id: str, heading_text: Optional[str], heading_level: int) -> Section:
        nonlocal section_order
        section_order += 1
        sec = Section(
            section_id=section_id,
            heading_text=heading_text,
            heading_level=heading_level,
            order_in_document=section_order,
            units=[]
        )
        sections.append(sec)
        unit_counts[section_id] = {"P": 0, "T": 0, "F": 0}
        unit_order[section_id] = 0
        return sec

    def add_unit(sec: Section, unit_type: str, raw_text: str, caption: Optional[str] = None) -> None:
        prefix = {"paragraph": "P", "table": "T", "figure": "F"}[unit_type]
        unit_counts[sec.section_id][prefix] += 1
        unit_order[sec.section_id] += 1

        sec.units.append(
            Unit(
                unit_id=make_unit_id(paper_key, sec.section_id, prefix, unit_counts[sec.section_id][prefix]),
                unit_type=unit_type,
                order_in_section=unit_order[sec.section_id],
                raw_text=normalize_ws_keep_lines(raw_text),
                caption=caption.strip() if caption and caption.strip() else None
            )
        )

    # Paragraph buffer
    para_buf: List[str] = []

    def flush_paragraph(target: Optional[Section]) -> None:
        nonlocal para_buf
        if not para_buf:
            return
        text = "\n".join(para_buf).strip("\n")
        para_buf = []
        if not text.strip():
            return
        if target is None:
            preamble_paras.append(text)
        else:
            add_unit(target, "paragraph", text)

    # Table/Figure block states
    in_table = False
    in_figure = False
    block_lines: List[str] = []

    def finalize_block(content_lines: List[str]) -> Tuple[Optional[str], str]:
        stripped = [ln.rstrip("\n") for ln in content_lines]
        while stripped and is_blank(stripped[0]):
            stripped.pop(0)
        while stripped and is_blank(stripped[-1]):
            stripped.pop()
        if not stripped:
            return None, ""
        caption = stripped[0].strip()
        body = "\n".join(stripped[1:]).strip("\n")
        if not body:
            return caption, caption
        return caption, f"{caption}\n{body}"

    # Iterate
    for line in lines:
        # title/authors metadata
        if title is None:
            m = TITLE_RE.match(line)
            if m:
                title = m.group(1).strip()
                continue
        if authors is None:
            m = AUTHORS_RE.match(line)
            if m:
                authors = m.group(1).strip()
                continue

        # block handling
        if in_table:
            if line.strip() == TABLE_END:
                flush_paragraph(current_section)
                caption, raw = finalize_block(block_lines)
                if current_section is None:
                    # keep as preamble paragraph (rare)
                    preamble_paras.append(f"{TABLE_START}\n" + "\n".join(block_lines) + f"\n{TABLE_END}")
                else:
                    add_unit(current_section, "table", raw, caption=caption)
                in_table = False
                block_lines = []
            else:
                block_lines.append(line)
            continue

        if in_figure:
            if line.strip() == FIG_END:
                flush_paragraph(current_section)
                caption, raw = finalize_block(block_lines)
                if current_section is None:
                    preamble_paras.append(f"{FIG_START}\n" + "\n".join(block_lines) + f"\n{FIG_END}")
                else:
                    add_unit(current_section, "figure", raw, caption=caption)
                in_figure = False
                block_lines = []
            else:
                block_lines.append(line)
            continue

        # start blocks
        if line.strip() == TABLE_START:
            flush_paragraph(current_section)
            in_table = True
            block_lines = []
            continue

        if line.strip() == FIG_START:
            flush_paragraph(current_section)
            in_figure = True
            block_lines = []
            continue

        # heading
        hm = HEADING_RE.match(line)
        if hm:
            flush_paragraph(current_section)
            hashes, heading_text = hm.group(1), hm.group(2)
            level = len(hashes)
            section_id = sec_num.next_id(level)
            current_section = ensure_section(section_id, heading_text.strip(), level)
            continue

        # paragraph boundaries
        if is_blank(line):
            flush_paragraph(current_section)
        else:
            para_buf.append(line)

    # flush tail paragraph
    flush_paragraph(current_section)

    # Insert S0 if needed
    if preamble_paras:
        # shift existing section order; S0 should be order 0
        s0 = Section(
            section_id="S0",
            heading_text=None,
            heading_level=1,
            order_in_document=0,
            units=[]
        )
        unit_counts["S0"] = {"P": 0, "T": 0, "F": 0}
        unit_order["S0"] = 0
        for p in preamble_paras:
            add_unit(s0, "paragraph", p)
        sections.insert(0, s0)

        # renumber order_in_document deterministically
        for idx, sec in enumerate(sections):
            sec.order_in_document = idx

    # Build JSON
    out: Dict = {
        "paper_key": paper_key,
        "source_filename": source_filename,
        "paper_title": title,
        "paper_authors": authors,
        "sections": []
    }

    for sec in sections:
        sec_obj = {
            "section_id": sec.section_id,
            "section_heading": sec.heading_text,  # null for S0
            "heading_level": sec.heading_level,
            "order_in_document": sec.order_in_document,
            "units": []
        }

        # paragraph units for context pointers (within the same section)
        para_units = [u for u in sec.units if u.unit_type == "paragraph"]

        for u in sec.units:
            u_obj = {
                "unit_id": u.unit_id,
                "unit_type": u.unit_type,
                "order_in_section": u.order_in_section,
                "raw_text": u.raw_text
            }
            if u.unit_type in ("table", "figure") and u.caption:
                u_obj["caption"] = u.caption

            if context_window > 0 and u.unit_type == "paragraph":
                idx = next((k for k, pu in enumerate(para_units) if pu.unit_id == u.unit_id), None)
                if idx is not None:
                    prev_ids = [para_units[k].unit_id for k in range(max(0, idx - context_window), idx)]
                    next_ids = [para_units[k].unit_id for k in range(idx + 1, min(len(para_units), idx + 1 + context_window))]
                    u_obj["context_within_section"] = {
                        "prev_paragraph_unit_ids": prev_ids,
                        "next_paragraph_unit_ids": next_ids
                    }

            sec_obj["units"].append(u_obj)

        out["sections"].append(sec_obj)

    return out


# ----------------------------
# Batch CLI
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", "-i", help="Directory containing .md files")
    ap.add_argument("--output_dir", "-o", help="Directory to write .json files")
    ap.add_argument("--context_window", type=int, default=0,
                    help="Store prev/next paragraph unit ids within section (default: 0)")
    ap.add_argument("--output_name_mode", choices=["stem", "key"], default="stem",
                    help="Output filename: 'stem' -> <md_stem>.json (default), 'key' -> <paper_key>.json")
    args = ap.parse_args()

    if not args.input_dir:
        args.input_dir = input("Enter input_dir (directory containing .md files): ").strip()
    if not args.output_dir:
        args.output_dir = input("Enter output_dir (directory to write .json files): ").strip()

    if not args.input_dir or not args.output_dir:
        raise SystemExit("Both input_dir and output_dir are required.")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    md_files = sorted(in_dir.glob("*.md"))
    if not md_files:
        raise SystemExit(f"No .md files found in: {in_dir}")

    print(f"Found {len(md_files)} markdown files in {in_dir}")

    for md_path in md_files:
        md_text = md_path.read_text(encoding="utf-8", errors="replace")
        data = parse_markdown_to_json(md_text, source_filename=md_path.name, context_window=args.context_window)

        if args.output_name_mode == "key":
            out_name = f"{data['paper_key']}.json"
        else:
            out_name = f"{md_path.stem}.json"

        out_path = out_dir / out_name
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

        print(f"- {md_path.name} -> {out_name}  (paper_key={data['paper_key']})")

    print(f"Done. JSON files written to: {out_dir}")


if __name__ == "__main__":
    main()
