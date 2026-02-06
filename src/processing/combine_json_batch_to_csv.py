#!/usr/bin/env python3
"""
Interactive JSON batch → CSV combiner.

- Prompts user for:
  1) Input directory containing JSON files
  2) Output directory for the CSV file

- Works with ANY JSON schema (same structure per batch)
- Automatically flattens nested dictionaries
- Handles lists safely
- Adds source_file column for traceability
"""

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union


JsonType = Union[Dict[str, Any], List[Any]]


def is_primitive(x: Any) -> bool:
    return x is None or isinstance(x, (str, int, float, bool))


def flatten_json(
    obj: Any,
    parent_key: str = "",
    sep: str = ".",
    list_sep: str = ";",
) -> Dict[str, Any]:
    """
    Flatten arbitrary JSON into a single-level dict.
    Nested keys become dotted paths.
    """
    items: Dict[str, Any] = {}

    def add(key: str, value: Any):
        items[key] = value

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(flatten_json(v, new_key, sep, list_sep))
        return items

    if isinstance(obj, list):
        key = parent_key if parent_key else "value"
        if all(is_primitive(x) for x in obj):
            add(key, list_sep.join("" if x is None else str(x) for x in obj))
        else:
            add(key, json.dumps(obj, ensure_ascii=False))
        return items

    add(parent_key if parent_key else "value", obj)
    return items


def load_json(path: Path) -> JsonType:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_record_list(container: Dict[str, Any]) -> Optional[str]:
    """
    Auto-detect a list of dicts inside a wrapper object.
    """
    for k, v in container.items():
        if isinstance(v, list) and v and all(isinstance(x, dict) for x in v):
            return k
    return None


def iter_records(data: JsonType) -> Iterable[Tuple[Dict[str, Any], Dict[str, Any]]]:
    """
    Yield (record, context) pairs.
    """
    if isinstance(data, list):
        for rec in data:
            yield (rec if isinstance(rec, dict) else {"value": rec}, {})
        return

    detected = detect_record_list(data)
    if detected:
        context = {k: v for k, v in data.items() if k != detected}
        for rec in data[detected]:
            yield (rec, context)
        return

    yield (data, {})


def combine_jsons(input_dir: Path) -> Tuple[List[Dict[str, Any]], List[str]]:
    json_files = sorted(input_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError("No JSON files found in input directory.")

    rows: List[Dict[str, Any]] = []
    all_fields: set[str] = set()

    for path in json_files:
        data = load_json(path)

        for record, context in iter_records(data):
            row = {}

            if context:
                row.update(flatten_json(context, parent_key="context"))

            row.update(flatten_json(record))
            row["source_file"] = path.name

            rows.append(row)
            all_fields.update(row.keys())

    header = sorted(all_fields)
    header.remove("source_file")
    header.append("source_file")

    return rows, header


def write_csv(rows: List[Dict[str, Any]], header: List[str], output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main():
    print("\n=== JSON Batch → CSV Combiner ===\n")

    input_dir = Path(input("Enter INPUT directory containing JSON files:\n> ").strip()).expanduser()
    if not input_dir.exists():
        raise FileNotFoundError("Input directory does not exist.")

    output_dir = Path(input("\nEnter OUTPUT directory for CSV file:\n> ").strip()).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    output_csv = output_dir / "combined_output.csv"

    rows, header = combine_jsons(input_dir)
    write_csv(rows, header, output_csv)

    print(f"\n✔ Done!")
    print(f"• Files processed: {len(set(r['source_file'] for r in rows))}")
    print(f"• Rows written: {len(rows)}")
    print(f"• Columns: {len(header)}")
    print(f"• Output file: {output_csv}\n")


if __name__ == "__main__":
    main()
