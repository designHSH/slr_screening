"""
rename_papers.py  (Interactive version)

HOW TO USE
----------
1. Open this script in VS Code.
2. Press the green "Run" button (top right) OR press F5.
3. The terminal will ASK you:
      - Enter input directory path
      - Enter output directory path (or press Enter to rename in place)
4. Script renames files OR creates new renamed copies.
5. It generates:
      - filename_mapping.csv
      - rename_log.txt

SUPPORTED FILENAME FORMAT
-------------------------
Author(s) - YEAR - Title.txt

Examples:
  Amirabdollahian et al. - 2014 - Design...txt
  Chartomatsidis M. and Goumopoulos C. - 2020 - Development...txt

OUTPUT FILENAME FORMAT
----------------------
SurnameOrTwo_Year_title-slug.txt

Examples:
  Amirabdollahian_2014_hand-wrist-exoskeleton.txt
  Chartomatsidis-Goumopoulos_2020_development-evaluation-motion-exe.txt
"""

import os
import csv
import logging
import re
from pathlib import Path
import shutil


# ---------------- CONFIG ---------------- #

STOPWORDS = {
    "the", "of", "and", "in", "on", "for", "to", "a", "an", "with",
    "using", "use", "from", "by", "at", "into", "after", "based",
    "during", "through", "user", "centered", "centred", "design"
}

MAX_TITLE_WORDS = 8
MAX_TITLE_SLUG_LEN = 40


# ---------------- AUTHOR HANDLING ---------------- #

def clean_author(author_part: str) -> str:
    """Extract surname(s) from author segment."""

    author = re.sub(r"\bet\s+al\.?\b", "", author_part, flags=re.IGNORECASE)

    author = re.sub(r"[.,;:!?\'\"()\[\]{}]", "", author)

    author = author.replace(" and ", ",").replace(" & ", ",")

    author = re.sub(r"\s+", " ", author).strip()

    if not author:
        return "UnknownAuthor"

    surnames = []
    for chunk in author.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = chunk.split()
        if not parts:
            continue
        surname = parts[0]
        surname = re.sub(r'[<>:"/\\|?*]', "", surname)
        if surname:
            surnames.append(surname)

    if not surnames:
        return "UnknownAuthor"

    surnames = surnames[:2]  # Keep max 2 authors
    return "-".join(surnames)


def extract_year(year_part: str) -> str:
    match = re.search(r"\b(\d{4})\b", year_part)
    return match.group(1) if match else "0000"


# ---------------- TITLE SLUG ---------------- #

def build_title_slug(title_part: str) -> str:
    title = os.path.splitext(title_part)[0]
    title = title.lower()
    title = re.sub(r"[^a-z0-9]+", " ", title)
    words = [w for w in title.split() if w and w not in STOPWORDS]
    words = words[:MAX_TITLE_WORDS]
    if not words:
        return "title"
    slug = "-".join(words)
    if len(slug) > MAX_TITLE_SLUG_LEN:
        slug = slug[:MAX_TITLE_SLUG_LEN].rstrip("-")
    slug = re.sub(r'[<>:"/\\|?*]', "", slug)
    return slug or "title"


# ---------------- RENAME LOGIC ---------------- #

def build_new_name(old_name: str) -> str:
    name_no_ext, ext = os.path.splitext(old_name)
    parts = [p.strip() for p in name_no_ext.split(" - ", maxsplit=2)]

    if len(parts) < 3:
        logging.warning(f"Unexpected name format: {old_name}")
        return f"Unknown_0000_{name_no_ext[:20]}{ext}"

    author_part, year_part, title_part = parts
    surname = clean_author(author_part)
    year = extract_year(year_part)
    slug = build_title_slug(title_part)

    return f"{surname}_{year}_{slug}{ext}"


def process_files(input_dir: Path, output_dir: Path | None) -> None:

    if output_dir is None:
        output_dir = input_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "rename_log.txt"
    logging.basicConfig(
        filename=str(log_path),
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    mapping_path = output_dir / "filename_mapping.csv"

    with mapping_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["old_name", "new_name", "old_path", "new_path"])

        for file_path in input_dir.glob("*.txt"):
            old_name = file_path.name
            old_full = str(file_path.resolve())

            try:
                new_name = build_new_name(old_name)
                target_path = output_dir / new_name
                new_full = str(target_path.resolve())

                counter = 1
                while target_path.exists() and target_path != file_path:
                    stem, ext = os.path.splitext(new_name)
                    new_name = f"{stem}_{counter}{ext}"
                    target_path = output_dir / new_name
                    new_full = str(target_path.resolve())
                    counter += 1

                if input_dir == output_dir:
                    file_path.rename(target_path)
                else:
                    shutil.copy2(file_path, target_path)

                writer.writerow([old_name, new_name, old_full, new_full])
                logging.info(f"{old_name} → {new_name}")

            except Exception as e:
                logging.error(f"Error processing {old_name}: {e}")

    print("\n===== DONE =====")
    print("CSV mapping:", mapping_path)
    print("Log file:", log_path)


# ---------------- INTERACTIVE MAIN ---------------- #

def main():
    print("=== PAPER RENAME TOOL ===\n")

    input_path = input("Enter INPUT directory (where .txt files are): ").strip()
    output_path = input("Enter OUTPUT directory (or press Enter to rename in place): ").strip()

    input_dir = Path(input_path)
    if not input_dir.exists():
        print("\n❌ ERROR: Input directory does not exist.")
        return

    output_dir = Path(output_path) if output_path else None

    process_files(input_dir, output_dir)


if __name__ == "__main__":
    main()
