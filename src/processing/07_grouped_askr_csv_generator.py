from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


FIELDS = [
    "unit_id",
    "paper_key",
    "paper_title",
    "workload_type",
    "askr_type",
    "workload_summary",
    "askr_summary",
    "workload_quote",
    "verbatim_quote",
    "raw_text",
]


def _iter_records(obj: Any) -> Iterable[Dict[str, Any]]:
    if isinstance(obj, dict):
        if isinstance(obj.get("records"), list):
            for item in obj["records"]:
                if isinstance(item, dict):
                    yield item
            return
        yield obj
        return
    if isinstance(obj, list):
        for item in obj:
            if isinstance(item, dict):
                yield item


def _rows_from_record(rec: Dict[str, Any]) -> List[Dict[str, Any]]:
    paper = rec.get("paper") or {}
    unit = rec.get("unit") or {}
    extraction = rec.get("extraction") or {}

    base = {
        "unit_id": unit.get("unit_id", ""),
        "paper_key": paper.get("paper_key", ""),
        "paper_title": paper.get("paper_title", ""),
        "workload_type": extraction.get("workload_type", ""),
        "workload_summary": extraction.get("workload_summary", ""),
        "workload_quote": extraction.get("workload_quote", ""),
        "raw_text": unit.get("raw_text", ""),
    }

    related_askr = extraction.get("related_askr")
    if isinstance(related_askr, list) and related_askr:
        rows: List[Dict[str, Any]] = []
        for askr in related_askr:
            if not isinstance(askr, dict):
                continue
            row = dict(base)
            row.update(
                {
                    "askr_type": askr.get("askr_type", ""),
                    "askr_summary": askr.get("askr_summary", ""),
                    "verbatim_quote": askr.get("verbatim_quote", ""),
                }
            )
            rows.append(row)
        return rows

    row = dict(base)
    row.update({"askr_type": "", "askr_summary": "", "verbatim_quote": ""})
    return [row]


def _prompt_path(label: str) -> Path:
    raw = input(label).strip().strip('"').strip("'")
    if not raw:
        raise ValueError("Path cannot be empty.")
    return Path(raw)


def main() -> None:
    input_dir = _prompt_path("Input folder: ")
    output_dir = _prompt_path("Output folder: ")

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    json_files = sorted(p for p in input_dir.glob("*.json") if p.is_file())

    output_dir.mkdir(parents=True, exist_ok=True)

    total_rows = 0
    for path in json_files:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in {path}") from exc

        rows: List[Dict[str, Any]] = []
        for record in _iter_records(data):
            rows.extend(_rows_from_record(record))

        out_csv = output_dir / f"{path.stem}.csv"
        with out_csv.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        total_rows += len(rows)
        print(f"Wrote {len(rows)} rows to {out_csv}")

    print(f"Done. {len(json_files)} files processed, {total_rows} rows total.")


if __name__ == "__main__":
    main()
