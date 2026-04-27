import csv
import json
import re
from pathlib import Path
from typing import Dict, List, Set, Tuple


FILENAME_RE = re.compile(r"^(W\d+(?:\.\d+)*)_([A-Za-z])\.json$")


def _prompt_path(prompt: str, default: Path) -> Path:
    value = input(f"{prompt} [{default}]: ").strip()
    return Path(value) if value else default


def _parse_filename(path: Path) -> Tuple[str, str]:
    match = FILENAME_RE.match(path.name)
    if not match:
        return "", ""
    workload = match.group(1).upper()
    askr = match.group(2).upper()
    return workload, askr


def _sorted_workloads(workloads: Set[str]) -> List[str]:
    def key_fn(w: str) -> Tuple[int, int]:
        parts = w[1:].split(".")
        major = int(parts[0]) if parts[0].isdigit() else 0
        minor = int(parts[1]) if len(parts) > 1 and parts[1].isdigit() else 0
        return major, minor

    return sorted(workloads, key=key_fn)


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _extract_counts(data: dict) -> Tuple[int, int]:
    summary = data.get("summary", {})
    records_count = summary.get("records_count")
    papers_count = summary.get("papers_count")

    if isinstance(records_count, int) and isinstance(papers_count, int):
        return records_count, papers_count

    records = data.get("records", [])
    records_count = len(records) if isinstance(records, list) else 0
    paper_keys = set()
    if isinstance(records, list):
        for rec in records:
            paper = rec.get("paper", {}) if isinstance(rec, dict) else {}
            key = paper.get("paper_key")
            if key:
                paper_keys.add(str(key))
    papers_count = len(paper_keys)
    return records_count, papers_count


def _extract_paper_keys(data: dict) -> List[str]:
    records = data.get("records", [])
    keys: Set[str] = set()
    if isinstance(records, list):
        for rec in records:
            paper = rec.get("paper", {}) if isinstance(rec, dict) else {}
            key = paper.get("paper_key")
            if key:
                keys.add(str(key))
    return sorted(keys)


def _write_matrix_csv(
    output_path: Path,
    rows: List[str],
    cols: List[str],
    matrix: Dict[Tuple[str, str], str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + cols)
        for r in rows:
            writer.writerow([r] + [matrix.get((r, c), "") for c in cols])


def main() -> int:
    default_input = Path("data/askr_extraction/grouped_gap_by_batch/v014/Total")
    default_counts = Path("data/askr_extraction/grouped_gap_by_batch/v014/Total_counts_matrix.csv")
    default_keys = Path("data/askr_extraction/grouped_gap_by_batch/v014/Total_paper_keys_matrix.csv")

    input_dir = _prompt_path("Input folder with grouped JSON files", default_input)
    counts_csv = _prompt_path("Output CSV path for records/papers counts", default_counts)
    keys_csv = _prompt_path("Output CSV path for paper_key values", default_keys)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    workloads: Set[str] = set()
    askr_types: Set[str] = set()
    counts_matrix: Dict[Tuple[str, str], str] = {}
    keys_matrix: Dict[Tuple[str, str], str] = {}

    for path in input_dir.glob("*.json"):
        workload, askr = _parse_filename(path)
        if not workload or not askr:
            continue
        workloads.add(workload)
        askr_types.add(askr)

        data = _load_json(path)
        records_count, papers_count = _extract_counts(data)
        counts_matrix[(workload, askr)] = f"records={records_count}; papers={papers_count}"

        keys = _extract_paper_keys(data)
        keys_matrix[(workload, askr)] = "; ".join(keys)

    row_order = _sorted_workloads(workloads)
    col_order = sorted(askr_types)

    _write_matrix_csv(counts_csv, row_order, col_order, counts_matrix)
    _write_matrix_csv(keys_csv, row_order, col_order, keys_matrix)

    print(f"Wrote counts matrix: {counts_csv}")
    print(f"Wrote paper_key matrix: {keys_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
