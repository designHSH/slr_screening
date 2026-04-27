import json
from pathlib import Path


def safe_get(d, *keys, default=""):
    """Return the first existing key from a dict."""
    for key in keys:
        if isinstance(d, dict) and key in d and d[key] is not None:
            return d[key]
    return default


def flatten_grouped_gap_file(input_path: str, output_path: str, prefix: str):
    """
    Flatten a grouped ASK+R JSON file into a simple list of records.

    Parameters
    ----------
    input_path : str
        Path to the grouped JSON file.
    output_path : str
        Path to save the flattened JSON file.
    prefix : str
        Prefix for gap_id values, e.g. 'W7.3_A'
    """
    input_file = Path(input_path)
    output_file = Path(output_path)

    with input_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    flat_records = []
    counter = 1

    def add_record(item: dict, askr: dict | None = None):
        nonlocal counter

        unit_id = safe_get(item, "unit_id")
        paper_key = safe_get(item, "paper_key", "paper_id", "file_name")
        workload_type = safe_get(item, "workload_type", default="")
        askr_type = safe_get(item, "askr_type", default="")

        # Support either direct fields or nested unit fields
        raw_text = safe_get(item, "raw_text")
        if not raw_text and isinstance(item.get("unit"), dict):
            raw_text = safe_get(item["unit"], "raw_text")

        if not unit_id and isinstance(item.get("unit"), dict):
            unit_id = safe_get(item["unit"], "unit_id")

        if not paper_key and isinstance(item.get("paper"), dict):
            paper_key = safe_get(item["paper"], "paper_key")

        if not workload_type and isinstance(item.get("extraction"), dict):
            workload_type = safe_get(item["extraction"], "workload_type")

        if askr:
            askr_type = safe_get(askr, "askr_type", default=askr_type)

        record = {
            "gap_id": f"{prefix}_{counter:03d}",
            "unit_id": unit_id or safe_get(item.get("unit", {}), "unit_id"),
            "paper_key": paper_key,
            "workload_type": workload_type,
            "askr_type": askr_type,
            "askr_summary": safe_get(askr or item, "askr_summary"),
            "verbatim_quote": safe_get(askr or item, "verbatim_quote"),
            "reasoning": safe_get((askr or {}).get("validation", {}), "reasoning")
            or safe_get(askr or item, "reasoning", "short_reasoning"),
            "raw_text": raw_text
        }

        flat_records.append(record)
        counter += 1

    # Case 1: already a list of records
    if isinstance(data, list):
        for item in data:
            if isinstance(item, dict):
                add_record(item)

    # Case 2: records may be inside a known wrapper key
    elif isinstance(data, dict):
        if isinstance(data.get("records"), list):
            for item in data["records"]:
                if isinstance(item, dict):
                    extraction = item.get("extraction", {})
                    related_askr = extraction.get("related_askr")
                    if isinstance(related_askr, list) and related_askr:
                        for askr in related_askr:
                            if isinstance(askr, dict):
                                add_record(item, askr=askr)
                    else:
                        add_record(item)

        elif isinstance(data.get("items"), list):
            for item in data["items"]:
                if isinstance(item, dict):
                    add_record(item)

        else:
            # Case 3: recursively search for lists of dicts
            def walk(obj):
                if isinstance(obj, list):
                    for x in obj:
                        if isinstance(x, dict):
                            # Heuristic: treat dicts with askr_summary or verbatim_quote as records
                            if "askr_summary" in x or "verbatim_quote" in x:
                                add_record(x)
                            else:
                                walk(x)
                elif isinstance(obj, dict):
                    for v in obj.values():
                        walk(v)

            walk(data)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(flat_records, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(flat_records)} flat records to: {output_file}")


def flatten_grouped_gap_folder(input_dir: str, output_dir: str):
    """
    Flatten all grouped ASK+R JSON files in a folder.

    Parameters
    ----------
    input_dir : str
        Folder containing grouped JSON files (e.g., W7.3_A.json).
    output_dir : str
        Folder to save flattened JSON files (e.g., W7.3_A_flat.json).
    """
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for input_file in sorted(in_dir.glob("*.json")):
        prefix = input_file.stem
        output_file = out_dir / f"{prefix}_flat.json"
        flatten_grouped_gap_file(
            input_path=str(input_file),
            output_path=str(output_file),
            prefix=prefix
        )


if __name__ == "__main__":
    flatten_grouped_gap_folder(
        input_dir=r"data\03_barrier_indentification_by_askr_gap\01_input\grouped_askr",
        output_dir=r"data\03_barrier_indentification_by_askr_gap\01_input\flatten_askr"
    )
