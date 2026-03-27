import json
from pathlib import Path
from copy import deepcopy


SCHEMA_VERSION = "askr_gap_only_v0.1"
SOURCE_SCHEMA_VERSION = "askr_extraction_output_v0.1"
PROMPT_TYPE = "askr_gap_filtering"


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def filter_gap_only_record(record: dict) -> tuple[dict | None, int]:
    """
    Returns:
        filtered_record or None
        number_of_gap_askr_items_kept
    """
    extraction_block = record.get("extraction", {})
    extractions = extraction_block.get("extractions", [])

    kept_extractions = []
    kept_gap_items_count = 0

    for extraction in extractions:
        related_askr = extraction.get("related_askr", [])

        kept_related_askr = [
            askr_item
            for askr_item in related_askr
            if askr_item.get("gap_signal") is True
        ]

        if kept_related_askr:
            new_extraction = {
                "workload_type": extraction.get("workload_type"),
                "workload_summary": extraction.get("workload_summary"),
                "workload_quote": extraction.get("workload_quote"),
                "related_askr": kept_related_askr,
            }
            kept_extractions.append(new_extraction)
            kept_gap_items_count += len(kept_related_askr)

    if not kept_extractions:
        return None, 0

    filtered_record = {
        "section": deepcopy(record.get("section", {})),
        "unit": deepcopy(record.get("unit", {})),
        "extraction": {
            "extractions": kept_extractions
        },
        "run_meta": deepcopy(record.get("run_meta", {})),
    }

    return filtered_record, kept_gap_items_count


def build_gap_only_output(source_data: dict) -> dict:
    source_prompt_revision = source_data.get("prompt_revision")
    paper = deepcopy(source_data.get("paper", {}))
    records = source_data.get("records", [])

    kept_records = []
    gap_askr_items_count = 0

    for record in records:
        filtered_record, kept_count = filter_gap_only_record(record)
        if filtered_record is not None:
            kept_records.append(filtered_record)
            gap_askr_items_count += kept_count

    input_records_count = len(records)
    kept_records_count = len(kept_records)
    removed_records_count = input_records_count - kept_records_count

    output_data = {
        "schema_version": SCHEMA_VERSION,
        "source_schema_version": SOURCE_SCHEMA_VERSION,
        "prompt_type": PROMPT_TYPE,
        "source_prompt_type": source_data.get("prompt_type"),
        "source_prompt_revision": source_prompt_revision,
        "paper": paper,
        "records": kept_records,
        "summary": {
            "input_records_count": input_records_count,
            "kept_records_count": kept_records_count,
            "removed_records_count": removed_records_count,
            "gap_askr_items_count": gap_askr_items_count,
        },
    }

    return output_data


def process_directory(input_dir: Path, output_dir: Path) -> None:
    json_files = sorted(input_dir.glob("*.json"))

    if not json_files:
        print("No JSON files found in the input directory.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    total_files = len(json_files)
    print(f"Found {total_files} JSON files.")

    for idx, json_file in enumerate(json_files, start=1):
        try:
            source_data = load_json(json_file)
            paper_key = source_data.get("paper", {}).get("paper_key")

            if not paper_key:
                print(f"[{idx}/{total_files}] Skipped {json_file.name}: missing paper.paper_key")
                continue

            gap_only_data = build_gap_only_output(source_data)
            output_path = output_dir / f"{paper_key}.gap_only.json"
            save_json(output_path, gap_only_data)

            kept = gap_only_data["summary"]["kept_records_count"]
            gaps = gap_only_data["summary"]["gap_askr_items_count"]

            print(f"[{idx}/{total_files}] Done: {paper_key} | kept_records={kept} | gap_items={gaps}")

        except Exception as e:
            print(f"[{idx}/{total_files}] Error processing {json_file.name}: {e}")


def main():
    input_dir_str = input("Please enter the input directory: ").strip()
    output_dir_str = input("Please enter the output directory: ").strip()

    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    if not input_dir.exists() or not input_dir.is_dir():
        print("The input directory does not exist or is not a valid directory.")
        return

    process_directory(input_dir, output_dir)
    print("Finished.")


if __name__ == "__main__":
    main()