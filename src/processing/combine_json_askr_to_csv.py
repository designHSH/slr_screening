#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List


# One CSV per JSON file (all units + all askr rows inside)
# Added workload_quote + workload_summary (per your request)
HEADERS = [
    "unit_id",
    "primary_workload",
    "secondary_workloads",
    "workload_summary",
    "workload_quote",
    "askr_type",
    "linked_workload",
    "askr_summary",
    "verbatim_quote",
    "gap_signal",
    "short_reasoning",
    "confidence_level",
]


def secondary_workloads_to_cell(value: Any) -> str:
    """Convert secondary_workloads to a single CSV cell."""
    if value is None:
        return ""
    if isinstance(value, list):
        return "; ".join(str(x) for x in value)
    return str(value)


def extract_rows_from_json(data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Flattens the JSON into row dicts.
    Each ASK+R item becomes one row. If a unit has multiple askr items,
    unit-level fields (unit_id/workload fields) are repeated.
    """
    rows: List[Dict[str, Any]] = []

    records = data.get("records", [])
    if not isinstance(records, list):
        return rows

    for rec in records:
        if not isinstance(rec, dict):
            continue

        unit = rec.get("unit", {}) or {}
        extraction = rec.get("extraction", {}) or {}

        unit_id = unit.get("unit_id", "")

        workload = extraction.get("workload", {}) or {}
        primary_workload = workload.get("primary_workload", "")
        secondary_workloads = workload.get("secondary_workloads", "")
        workload_summary = workload.get("workload_summary", "")
        workload_quote = workload.get("workload_quote", "")

        askr_list = extraction.get("askr", [])
        if not isinstance(askr_list, list):
            # If a unit has no askr list, you can choose to skip it (current)
            # or output a single row with empty askr fields.
            continue

        for askr in askr_list:
            if not isinstance(askr, dict):
                continue

            rows.append({
                "unit_id": unit_id,
                "primary_workload": primary_workload,
                "secondary_workloads": secondary_workloads,
                "workload_summary": workload_summary,
                "workload_quote": workload_quote,
                "askr_type": askr.get("askr_type", ""),
                "linked_workload": askr.get("linked_workload", ""),
                "askr_summary": askr.get("askr_summary", ""),
                "verbatim_quote": askr.get("verbatim_quote", ""),
                "gap_signal": askr.get("gap_signal", ""),
                "short_reasoning": askr.get("short_reasoning", ""),
                "confidence_level": askr.get("confidence_level", ""),
            })

    return rows


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADERS, extrasaction="ignore")
        writer.writeheader()

        for r in rows:
            r2 = dict(r)
            r2["secondary_workloads"] = secondary_workloads_to_cell(r2.get("secondary_workloads"))
            writer.writerow(r2)


def main() -> None:
    print("=== ASK+R JSON -> CSV (one CSV per JSON file) ===")
    input_dir_str = input("Enter INPUT directory path (contains .json files): ").strip()
    output_dir_str = input("Enter OUTPUT directory path (where .csv files will be written): ").strip()

    input_dir = Path(input_dir_str).expanduser().resolve()
    output_dir = Path(output_dir_str).expanduser().resolve()

    if not input_dir.exists() or not input_dir.is_dir():
        raise SystemExit(f"Input directory not found or not a directory: {input_dir}")

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        raise SystemExit(f"No .json files found in: {input_dir}")

    for jp in json_files:
        try:
            with jp.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[SKIP] Failed to read {jp.name}: {e}")
            continue

        rows = extract_rows_from_json(data)

        out_name = f"{jp.stem}.csv"
        out_path = output_dir / out_name
        write_csv(out_path, rows)

        print(f"[OK] {jp.name}: {len(rows)} ASK+R rows -> {out_name}")

    print(f"\nDone. Output folder: {output_dir}")


if __name__ == "__main__":
    main()
