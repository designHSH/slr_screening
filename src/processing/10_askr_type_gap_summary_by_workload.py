from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Set


DEFAULT_INPUT_DIR = Path("data/02_askr_extraction/extracted_askr/v_014/Total")
DEFAULT_OUTPUT_CSV = Path(
    "data/02_askr_extraction/extracted_askr/v_014/askr_type_gap_summary_by_workload.csv"
)


def _normalize_extractions(record: Dict[str, Any]) -> List[Dict[str, Any]]:
    extraction_block = record.get("extraction") or record.get("extractions") or {}

    if isinstance(extraction_block, list):
        return [x for x in extraction_block if isinstance(x, dict)]

    if isinstance(extraction_block, dict):
        extractions = extraction_block.get("extractions")
        if isinstance(extractions, list):
            return [x for x in extractions if isinstance(x, dict)]

    return []


def _normalize_related_askr(extraction_obj: Dict[str, Any]) -> List[Dict[str, Any]]:
    related_askr = extraction_obj.get("related_askr")
    if related_askr is None:
        related_askr = extraction_obj.get("askr_items")
    if related_askr is None:
        related_askr = extraction_obj.get("askr")

    if related_askr is None:
        return []
    if isinstance(related_askr, list):
        return [x for x in related_askr if isinstance(x, dict)]
    if isinstance(related_askr, dict):
        return [related_askr]
    return []


def _iter_askr_items_from_file(path: Path) -> Iterable[tuple[str, str, bool]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])
    if not isinstance(records, list):
        return

    for record in records:
        if not isinstance(record, dict):
            continue

        for extraction_obj in _normalize_extractions(record):
            workload_type = extraction_obj.get("workload_type")
            if not workload_type or not isinstance(workload_type, str):
                continue

            for askr_item in _normalize_related_askr(extraction_obj):
                askr_type = askr_item.get("askr_type")
                if not askr_type or not isinstance(askr_type, str):
                    continue

                gap_signal = askr_item.get("gap_signal") is True
                yield workload_type, askr_type, gap_signal


def build_summary(input_dir: Path) -> tuple[Dict[str, Dict[str, Dict[str, int]]], Set[str]]:
    counts: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(
        lambda: defaultdict(lambda: {"total": 0, "gap_true": 0, "gap_false": 0})
    )
    askr_types_seen: Set[str] = set()

    json_files = sorted(p for p in input_dir.glob("*.json") if p.is_file())
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in: {input_dir}")

    for file_path in json_files:
        for workload_type, askr_type, gap_signal in _iter_askr_items_from_file(file_path):
            askr_types_seen.add(askr_type)
            counts[workload_type][askr_type]["total"] += 1
            if gap_signal:
                counts[workload_type][askr_type]["gap_true"] += 1
            else:
                counts[workload_type][askr_type]["gap_false"] += 1

    return counts, askr_types_seen


def write_summary_csv(
    output_csv: Path,
    counts: Dict[str, Dict[str, Dict[str, int]]],
    askr_types_seen: Set[str],
) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    askr_types_sorted = sorted(askr_types_seen)
    fieldnames = ["workload_type"]
    for askr_type in askr_types_sorted:
        fieldnames.extend(
            [f"{askr_type}_total", f"{askr_type}_gap_true", f"{askr_type}_gap_false"]
        )

    workload_keys = sorted(counts.keys())
    with output_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for workload_type in workload_keys:
            row: Dict[str, Any] = {"workload_type": workload_type}
            for askr_type in askr_types_sorted:
                metrics = counts[workload_type].get(
                    askr_type, {"total": 0, "gap_true": 0, "gap_false": 0}
                )
                row[f"{askr_type}_total"] = metrics["total"]
                row[f"{askr_type}_gap_true"] = metrics["gap_true"]
                row[f"{askr_type}_gap_false"] = metrics["gap_false"]
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize askr_type counts and gap_signal splits by workload_type."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Directory containing input JSON files (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help=f"Output CSV path (default: {DEFAULT_OUTPUT_CSV})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir: Path = args.input_dir
    output_csv: Path = args.output_csv

    if not input_dir.exists() or not input_dir.is_dir():
        raise NotADirectoryError(f"Input directory does not exist or is invalid: {input_dir}")

    counts, askr_types_seen = build_summary(input_dir)
    write_summary_csv(output_csv, counts, askr_types_seen)

    print(f"Input directory: {input_dir}")
    print(f"Workload types: {len(counts)}")
    print(f"ASKR types: {', '.join(sorted(askr_types_seen))}")
    print(f"Output CSV: {output_csv}")


if __name__ == "__main__":
    main()
