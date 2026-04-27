# This script cleans and normalizes selected bibliographic fields
# from the master CSV file.
#
# It asks the user for:
# 1) the input CSV file path
# 2) the output directory
#
# Then it:
# - keeps the original columns unchanged
# - creates a cleaned authors column
# - extracts only the publication year from publication_date
# - creates a lightly cleaned author affiliations column
# - saves the result as a new CSV file in the output directory

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import pandas as pd


def clean_whitespace(text: str) -> str:
    """Normalize whitespace and line breaks."""
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_authors(value: object) -> str:
    """
    Light cleaning for authors field:
    - normalize whitespace
    - fix spacing around separators
    - remove repeated punctuation where obvious
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = clean_whitespace(text)

    # Normalize spacing around semicolons and commas
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*,\s*", ", ", text)

    # Collapse repeated separators
    text = re.sub(r"(;\s*){2,}", "; ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)

    # Remove accidental spaces before punctuation
    text = re.sub(r"\s+([,;:.])", r"\1", text)

    return text.strip(" ;,")


def extract_publication_year(value: object) -> str:
    """
    Extract a 4-digit year from publication_date.
    Keeps only the year, as requested.
    """
    if pd.isna(value):
        return ""

    text = clean_whitespace(str(value))

    # Look for 4-digit years in a reasonable range
    match = re.search(r"\b(19\d{2}|20\d{2}|21\d{2})\b", text)
    if match:
        return match.group(1)

    return ""


def clean_affiliations(value: object) -> str:
    """
    Light cleaning for author affiliations:
    - normalize whitespace
    - standardize separators
    - reduce obvious repeated punctuation/separators
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = clean_whitespace(text)

    # Normalize common separators between multiple affiliations
    text = re.sub(r"\s*;\s*", "; ", text)
    text = re.sub(r"\s*\|\s*", "; ", text)

    # Normalize spacing after commas
    text = re.sub(r"\s*,\s*", ", ", text)

    # Collapse repeated separators
    text = re.sub(r"(;\s*){2,}", "; ", text)
    text = re.sub(r"(,\s*){2,}", ", ", text)

    # Remove accidental spaces before punctuation
    text = re.sub(r"\s+([,;:.])", r"\1", text)

    return text.strip(" ;,")


def validate_required_columns(df: pd.DataFrame, required_columns: list[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Missing required column(s): {missing_str}")


def build_output_path(input_file: Path, output_dir: Path) -> Path:
    """
    Create output filename based on input filename.
    Example:
    _study_characteristics_master.csv
    -> _study_characteristics_master_clean_metadata.csv
    """
    stem = input_file.stem
    suffix = input_file.suffix if input_file.suffix else ".csv"
    output_name = f"{stem}_clean_metadata{suffix}"
    return output_dir / output_name


def main() -> None:
    print("Bibliographic metadata cleaning script")
    print("-" * 40)

    input_path_str = input("Enter the full path to the input CSV file: ").strip().strip('"')
    output_dir_str = input("Enter the full path to the output directory: ").strip().strip('"')

    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    if input_path.suffix.lower() != ".csv":
        raise ValueError("Input file must be a CSV file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nReading CSV...")
    df = pd.read_csv(input_path)

    required_columns = ["authors", "publication_date", "author_affiliations"]
    validate_required_columns(df, required_columns)

    print("Cleaning bibliographic fields...")

    # Keep original columns untouched; add clean columns
    df["authors_clean"] = df["authors"].apply(clean_authors)
    df["publication_year_clean"] = df["publication_date"].apply(extract_publication_year)
    df["author_affiliations_clean"] = df["author_affiliations"].apply(clean_affiliations)

    output_path = build_output_path(input_path, output_dir)

    print("Saving cleaned CSV...")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Output file saved to:\n{output_path}")


if __name__ == "__main__":
    main()