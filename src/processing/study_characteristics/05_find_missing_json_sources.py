import argparse
import csv
import json
from pathlib import Path
from typing import Iterable, Set, Tuple


def _normalize_stem(filename: str) -> str:
    if not filename:
        return ""
    return Path(filename).stem.strip().lower()


def load_csv_stems(csv_path: Path, field_name: str) -> Set[str]:
    stems: Set[str] = set()
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if field_name not in (reader.fieldnames or []):
            raise ValueError(
                f"Field '{field_name}' not found in CSV. Available: {reader.fieldnames}"
            )
        for row in reader:
            stem = _normalize_stem(row.get(field_name, ""))
            if stem:
                stems.add(stem)
    return stems


def iter_json_files(folder: Path) -> Iterable[Path]:
    return folder.glob("*.json")


def _get_nested_field(data: dict, field_path: str) -> str:
    if not field_path:
        return ""
    current = data
    for part in field_path.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    if current is None:
        return ""
    return str(current)


def load_json_stems(folder: Path, field_name: str) -> Tuple[Set[str], int, int]:
    stems: Set[str] = set()
    missing_field = 0
    parse_errors = 0
    for path in iter_json_files(folder):
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            parse_errors += 1
            continue

        value = _get_nested_field(data, field_name)
        if not value and field_name != "source_filename":
            # Backward-compatible fallback for top-level source_filename
            value = data.get("source_filename", "")
        if not value:
            missing_field += 1
            continue
        stem = _normalize_stem(str(value))
        if stem:
            stems.add(stem)
    return stems, missing_field, parse_errors


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Compare JSON source filenames vs CSV file names (ignoring extensions) "
            "and report missing items in the JSON folder."
        )
    )
    parser.add_argument(
        "--json-folder",
        type=Path,
        default=Path(
            "data/askr_extraction/gap_validation/v014/Total"
        ),
        help="Folder containing .json files with source filename field",
    )
    parser.add_argument(
        "--csv-path",
        type=Path,
        default=Path(
            "data/all_included_title_abstract_paper_cleaning/included_fulltext/"
            "01_study_characteristics/cleaning_and_processing/"
            "_study_characteristics_master_clean_metadata.csv"
        ),
        help="CSV file with file_name field (e.g., .pdf)",
    )
    parser.add_argument(
        "--json-field",
        default="paper.source_filename",
        help="Field name in JSON files that contains the source filename",
    )
    parser.add_argument(
        "--csv-field",
        default="file_name",
        help="Field name in CSV file that contains the file name",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output text file to write missing filenames (one per line)",
    )

    args = parser.parse_args()

    csv_path = args.csv_path
    json_folder = args.json_folder

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not json_folder.exists():
        raise FileNotFoundError(f"JSON folder not found: {json_folder}")

    csv_stems = load_csv_stems(csv_path, args.csv_field)
    json_stems, missing_field, parse_errors = load_json_stems(
        json_folder, args.json_field
    )

    missing = sorted(csv_stems - json_stems)

    print(f"CSV records (unique stems): {len(csv_stems)}")
    print(f"JSON records (unique stems): {len(json_stems)}")
    print(f"Missing in JSON folder: {len(missing)}")
    if missing_field:
        print(f"JSON files missing '{args.json_field}': {missing_field}")
    if parse_errors:
        print(f"JSON parse errors: {parse_errors}")

    if missing:
        print("\nMissing stems (extension removed):")
        for stem in missing:
            print(stem)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(missing) + ("\n" if missing else ""), encoding="utf-8")
        print(f"\nWrote missing list to: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
