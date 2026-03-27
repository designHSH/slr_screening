import json
from pathlib import Path
from copy import deepcopy


SCHEMA_VERSION = "askr_grouped_gap_v0.1"

ASKR_TO_BARRIER_CATEGORY = {
    "A": "emotion",
    "S": "logic",
    "K": "knowledge",
    "R": "resource"
}


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def initialize_group_file(workload_type: str, askr_type: str, source_batch: str) -> dict:
    group_key = f"{workload_type}_{askr_type}"

    return {
        "schema_version": SCHEMA_VERSION,
        "group_key": group_key,
        "workload_type": workload_type,
        "askr_type": askr_type,
        "barrier_category": ASKR_TO_BARRIER_CATEGORY.get(askr_type, None),
        "source_batch": source_batch,
        "records": [],
        "summary": {
            "records_count": 0,
            "papers_count": 0
        }
    }


def build_grouped_record(
    paper_obj: dict,
    section_obj: dict,
    unit_obj: dict,
    extraction_obj: dict,
    askr_item: dict,
    source_batch: str,
    source_gap_file: str
) -> dict:
    return {
        "source_batch": source_batch,
        "source_gap_file": source_gap_file,
        "paper": {
            "paper_key": paper_obj.get("paper_key"),
            "source_filename": paper_obj.get("source_filename"),
            "paper_title": paper_obj.get("paper_title"),
            "paper_authors": paper_obj.get("paper_authors")
        },
        "section": deepcopy(section_obj),
        "unit": deepcopy(unit_obj),
        "extraction": {
            "workload_type": extraction_obj.get("workload_type"),
            "workload_summary": extraction_obj.get("workload_summary"),
            "workload_quote": extraction_obj.get("workload_quote"),
            "askr": deepcopy(askr_item)
        }
    }


def process_gap_only_file(file_path: Path, grouped_data: dict, source_batch: str) -> None:
    data = load_json(file_path)

    paper_obj = data.get("paper", {})
    records = data.get("records", [])
    source_gap_file = file_path.name

    for record in records:
        section_obj = record.get("section", {})
        unit_obj = record.get("unit", {})
        extraction_block = record.get("extraction", {})
        extractions = extraction_block.get("extractions", [])

        for extraction_obj in extractions:
            workload_type = extraction_obj.get("workload_type")
            related_askr = extraction_obj.get("related_askr", [])

            if not workload_type:
                continue

            for askr_item in related_askr:
                askr_type = askr_item.get("askr_type")
                gap_signal = askr_item.get("gap_signal")

                # Safety check: grouped files should include only gap-bearing items
                if gap_signal is not True:
                    continue

                if not askr_type:
                    continue

                group_key = f"{workload_type}_{askr_type}"

                if group_key not in grouped_data:
                    grouped_data[group_key] = initialize_group_file(
                        workload_type=workload_type,
                        askr_type=askr_type,
                        source_batch=source_batch
                    )

                grouped_record = build_grouped_record(
                    paper_obj=paper_obj,
                    section_obj=section_obj,
                    unit_obj=unit_obj,
                    extraction_obj=extraction_obj,
                    askr_item=askr_item,
                    source_batch=source_batch,
                    source_gap_file=source_gap_file
                )

                grouped_data[group_key]["records"].append(grouped_record)


def finalize_grouped_data(grouped_data: dict) -> None:
    for _, group_obj in grouped_data.items():
        records = group_obj.get("records", [])
        paper_keys = set()

        for record in records:
            paper_key = record.get("paper", {}).get("paper_key")
            if paper_key:
                paper_keys.add(paper_key)

        group_obj["summary"]["records_count"] = len(records)
        group_obj["summary"]["papers_count"] = len(paper_keys)


def process_directory(input_dir: Path, output_dir: Path) -> None:
    json_files = sorted(input_dir.glob("*.gap_only.json"))

    if not json_files:
        print("No .gap_only.json files found in the input directory.")
        return

    grouped_data = {}

    total_files = len(json_files)
    source_batch = input_dir.name

    print(f"Found {total_files} gap-only JSON files.")
    print(f"Source batch: {source_batch}")

    for idx, file_path in enumerate(json_files, start=1):
        try:
            process_gap_only_file(
                file_path=file_path,
                grouped_data=grouped_data,
                source_batch=source_batch
            )
            print(f"[{idx}/{total_files}] Processed: {file_path.name}")
        except Exception as e:
            print(f"[{idx}/{total_files}] Error processing {file_path.name}: {e}")

    finalize_grouped_data(grouped_data)

    output_dir.mkdir(parents=True, exist_ok=True)

    for group_key, group_obj in grouped_data.items():
        output_path = output_dir / f"{group_key}.json"
        save_json(output_path, group_obj)

    print(f"Generated {len(grouped_data)} grouped files.")
    print("Finished.")


def main():
    input_dir_str = input("Please enter the input directory: ").strip()
    output_dir_str = input("Please enter the output directory: ").strip()

    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)

    if not input_dir.exists() or not input_dir.is_dir():
        print("The input directory does not exist or is not a valid directory.")
        return

    process_directory(input_dir, output_dir)


if __name__ == "__main__":
    main()