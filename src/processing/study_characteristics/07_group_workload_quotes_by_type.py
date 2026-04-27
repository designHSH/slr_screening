import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import DefaultDict, Dict, List


def _prompt_path(prompt: str, default: Path) -> Path:
    value = input(f"{prompt} [{default}]: ").strip()
    return Path(value) if value else default


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_records(payload: dict, source_file: Path) -> List[Dict[str, str]]:
    paper = payload.get("paper", {}) if isinstance(payload, dict) else {}
    paper_title = str(paper.get("paper_title", ""))
    paper_key = str(paper.get("paper_key", ""))
    records = payload.get("records", []) if isinstance(payload, dict) else []

    collected: List[Dict[str, str]] = []
    if not isinstance(records, list):
        return collected

    for record in records:
        if not isinstance(record, dict):
            continue

        unit = record.get("unit", {})
        unit_id = str(unit.get("unit_id", "")) if isinstance(unit, dict) else ""

        extraction_block = record.get("extraction", {})
        extractions = (
            extraction_block.get("extractions", [])
            if isinstance(extraction_block, dict)
            else []
        )
        if not isinstance(extractions, list):
            continue

        for extraction in extractions:
            if not isinstance(extraction, dict):
                continue

            workload_type = str(extraction.get("workload_type", "")).strip()
            if not workload_type:
                continue

            workload_summary = str(extraction.get("workload_summary", "")).strip()
            workload_quote = str(extraction.get("workload_quote", "")).strip()
            if not workload_summary and not workload_quote:
                continue

            collected.append(
                {
                    "workload_type": workload_type,
                    "paper_title": paper_title,
                    "paper_key": paper_key,
                    "unit_id": unit_id,
                    "workload_summary": workload_summary,
                    "workload_quote": workload_quote,
                    "source_file": str(source_file),
                }
            )

    return collected


def _sort_workload_types(workload_types: List[str]) -> List[str]:
    def key_fn(value: str) -> List[int]:
        cleaned = value.upper().lstrip("W")
        nums: List[int] = []
        for part in cleaned.split("."):
            nums.append(int(part) if part.isdigit() else -1)
        return nums

    return sorted(workload_types, key=key_fn)


def _write_log(
    log_path: Path,
    input_dir: Path,
    output_dir: Path,
    files_seen: int,
    files_failed: int,
    grouped: DefaultDict[str, List[Dict[str, str]]],
) -> None:
    timestamp = datetime.now(timezone.utc).isoformat()
    workload_types = _sort_workload_types(list(grouped.keys()))
    total_items = sum(len(grouped[w]) for w in workload_types)

    lines = [
        f"[{timestamp}] Group workload run",
        f"input_dir={input_dir}",
        f"output_dir={output_dir}",
        f"files_seen={files_seen}",
        f"files_failed={files_failed}",
        f"workload_groups={len(workload_types)}",
        f"total_merged_items={total_items}",
        "items_per_workload_type:",
    ]
    for workload_type in workload_types:
        lines.append(f"  {workload_type}: {len(grouped[workload_type])}")
    lines.append("")

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main() -> int:
    default_input = Path("data/02_askr_extraction/extracted_askr/v_014/1_1_raw_askr_b11")
    default_output = Path("data/02_askr_extraction/extracted_askr/v_014/workload_grouped")

    input_dir = _prompt_path("Input folder containing .askr.json files", default_input)
    output_dir = _prompt_path("Output folder for grouped workload files", default_output)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    grouped: DefaultDict[str, List[Dict[str, str]]] = defaultdict(list)
    files_seen = 0
    files_failed = 0

    for path in sorted(input_dir.rglob("*.askr.json")):
        files_seen += 1
        try:
            payload = _load_json(path)
        except Exception as exc:  # pragma: no cover
            files_failed += 1
            print(f"Skipping invalid JSON: {path} ({exc})")
            continue

        for item in _collect_records(payload, path):
            grouped[item["workload_type"]].append(item)

    for workload_type in _sort_workload_types(list(grouped.keys())):
        records = grouped[workload_type]
        output_path = output_dir / f"{workload_type}.json"
        data = {
            "workload_type": workload_type,
            "records_count": len(records),
            "records": records,
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {len(records)} records -> {output_path}")

    log_path = output_dir / "workload_grouping.log"
    _write_log(
        log_path=log_path,
        input_dir=input_dir,
        output_dir=output_dir,
        files_seen=files_seen,
        files_failed=files_failed,
        grouped=grouped,
    )

    print(f"Processed files: {files_seen}")
    print(f"Failed files: {files_failed}")
    print(f"Workload groups: {len(grouped)}")
    print(f"Log file: {log_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
