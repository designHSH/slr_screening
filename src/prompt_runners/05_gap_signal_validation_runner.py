import json
import os
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Tuple

import yaml
from dotenv import load_dotenv
from openai import OpenAI


# ============================================================
# Constants
# ============================================================

VALIDATION_SCHEMA_VERSION = "gap_signal_validation_result_v1.1"
VALID_RECORDS_SCHEMA_VERSION = "gap_signal_valid_records_v1.1"


# ============================================================
# Basic IO
# ============================================================

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def safe_name(text: str) -> str:
    bad_chars = '<>:"/\\|?*'
    for ch in bad_chars:
        text = text.replace(ch, "_")
    return text


# ============================================================
# Prompt rendering
# ============================================================

def render_prompt(template: str, variables: Dict[str, Any]) -> str:
    rendered = template
    for key, value in variables.items():
        placeholder = "{{" + key + "}}"
        rendered = rendered.replace(placeholder, str(value))
    return rendered


# ============================================================
# OpenAI Client
# ============================================================

class EmptyResponseError(RuntimeError):
    def __init__(self, message: str, meta: Dict[str, Any]) -> None:
        super().__init__(message)
        self.meta = meta


def build_openai_client() -> OpenAI:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")
    return OpenAI(api_key=api_key)


def call_openai_json(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_output_tokens: int
) -> Tuple[str, Dict[str, Any]]:
    response = client.chat.completions.create(
        model=model,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        max_completion_tokens=max_output_tokens,
    )

    meta: Dict[str, Any] = {
        "id": getattr(response, "id", None),
        "model": getattr(response, "model", None),
        "usage": getattr(response, "usage", None),
    }

    if not response.choices:
        raise EmptyResponseError("empty_response: no choices returned", meta=meta)

    choice = response.choices[0]
    content = choice.message.content
    if not content or not content.strip():
        finish_reason = getattr(choice, "finish_reason", None)
        meta["finish_reason"] = finish_reason
        raise EmptyResponseError(
            f"empty_response: finish_reason={finish_reason}",
            meta=meta
        )

    meta["finish_reason"] = getattr(choice, "finish_reason", None)

    return content, meta


# ============================================================
# Validation helpers
# ============================================================

def validate_validator_output(
    result: Dict[str, Any],
    expected_askr_count: int
) -> Tuple[bool, str]:
    if not isinstance(result, dict):
        return False, "Output is not a JSON object."

    if "workload_relevance" not in result:
        return False, "Missing workload_relevance."
    if "askr_validations" not in result:
        return False, "Missing askr_validations."

    workload_relevance = result["workload_relevance"]
    if not isinstance(workload_relevance, dict):
        return False, "workload_relevance must be an object."
    if "is_relevant" not in workload_relevance:
        return False, "workload_relevance missing is_relevant."

    askr_validations = result["askr_validations"]
    if not isinstance(askr_validations, list):
        return False, "askr_validations must be a list."

    if len(askr_validations) != expected_askr_count:
        return False, (
            f"askr_validations length mismatch. "
            f"Expected {expected_askr_count}, got {len(askr_validations)}."
        )

    seen_indices = set()

    for item in askr_validations:
        if not isinstance(item, dict):
            return False, "Each askr_validations item must be an object."

        required_top = {
            "askr_index",
            "askr_alignment",
            "gap_signal_validation",
            "grounding_validation",
            "reasoning",
        }
        missing_top = required_top - set(item.keys())
        if missing_top:
            return False, f"Missing fields in askr_validations item: {missing_top}"

        askr_index = item["askr_index"]
        if not isinstance(askr_index, int):
            return False, "askr_index must be an integer."
        if askr_index in seen_indices:
            return False, f"Duplicate askr_index: {askr_index}"
        seen_indices.add(askr_index)

        askr_alignment = item["askr_alignment"]
        if not isinstance(askr_alignment, dict) or "is_aligned" not in askr_alignment:
            return False, f"askr_alignment invalid for askr_index {askr_index}"

        gap_signal_validation = item["gap_signal_validation"]
        if (
            not isinstance(gap_signal_validation, dict)
            or "is_valid_signal" not in gap_signal_validation
        ):
            return False, f"gap_signal_validation invalid for askr_index {askr_index}"

        grounding_validation = item["grounding_validation"]
        if (
            not isinstance(grounding_validation, dict)
            or "status" not in grounding_validation
        ):
            return False, f"grounding_validation invalid for askr_index {askr_index}"

        if not isinstance(item["reasoning"], str):
            return False, f"reasoning must be string for askr_index {askr_index}"

    expected_indices = set(range(expected_askr_count))
    if seen_indices != expected_indices:
        return False, (
            f"askr_index values mismatch. Expected {sorted(expected_indices)}, "
            f"got {sorted(seen_indices)}"
        )

    return True, ""


# ============================================================
# Logging
# ============================================================

def save_log_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write(text)


def build_log_base_name(paper_key: str, unit_id: str, extraction_index: int) -> str:
    return safe_name(f"{paper_key}__{unit_id}__ex{extraction_index}")


# ============================================================
# Output initialization
# ============================================================

def initialize_validation_output(source_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": VALIDATION_SCHEMA_VERSION,
        "source_schema_version": source_data.get("schema_version"),
        "paper": deepcopy(source_data.get("paper", {})),
        "records": deepcopy(source_data.get("records", [])),
        "summary": {}
    }


def initialize_valid_records_output(source_data: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "schema_version": VALID_RECORDS_SCHEMA_VERSION,
        "source_schema_version": VALIDATION_SCHEMA_VERSION,
        "paper": deepcopy(source_data.get("paper", {})),
        "records": [],
        "summary": {}
    }


# ============================================================
# Failure state helpers
# ============================================================

def build_failed_item_validation(error_type: str, error_message: str) -> Dict[str, Any]:
    return {
        "validation_status": "failed",
        "needs_review": True,
        "error_type": error_type,
        "error_message": error_message
    }


def build_failed_workload_validation(error_type: str, error_message: str) -> Dict[str, Any]:
    return {
        "validation_status": "failed",
        "needs_review": True,
        "error_type": error_type,
        "error_message": error_message
    }


# ============================================================
# Core transformation helpers
# ============================================================

def attach_validation_to_full_output(
    validation_record: Dict[str, Any],
    extraction_index: int,
    validator_output: Dict[str, Any]
) -> None:
    extraction_obj = validation_record["extraction"]["extractions"][extraction_index]

    extraction_obj["workload_validation"] = {
        "validation_status": "completed",
        "needs_review": False,
        **deepcopy(validator_output["workload_relevance"])
    }

    askr_validations_by_index = {
        item["askr_index"]: item for item in validator_output["askr_validations"]
    }

    related_askr = extraction_obj.get("related_askr", [])
    for idx, askr_item in enumerate(related_askr):
        askr_item["validation"] = {
            "validation_status": "completed",
            "needs_review": False,
            **deepcopy(askr_validations_by_index[idx])
        }


def attach_failed_validation_to_extraction(
    extraction_obj: Dict[str, Any],
    error_type: str,
    error_message: str
) -> None:
    extraction_obj["workload_validation"] = build_failed_workload_validation(
        error_type=error_type,
        error_message=error_message
    )

    for askr_item in extraction_obj.get("related_askr", []):
        askr_item["validation"] = build_failed_item_validation(
            error_type=error_type,
            error_message=error_message
        )


def build_valid_records_output(validation_output: Dict[str, Any]) -> Dict[str, Any]:
    valid_output = initialize_valid_records_output(validation_output)
    input_records_count = len(validation_output.get("records", []))
    kept_records_count = 0
    valid_askr_items_count = 0

    for record in validation_output.get("records", []):
        new_record = {
            "section": deepcopy(record.get("section", {})),
            "unit": deepcopy(record.get("unit", {})),
            "extraction": {"extractions": []},
            "run_meta": deepcopy(record.get("run_meta", {})),
        }

        for extraction_obj in record.get("extraction", {}).get("extractions", []):
            workload_validation = extraction_obj.get("workload_validation", {})
            if workload_validation.get("validation_status") != "completed":
                continue

            new_related_askr = []

            for askr_item in extraction_obj.get("related_askr", []):
                validation = askr_item.get("validation", {})

                if validation.get("validation_status") != "completed":
                    continue

                is_valid_signal = (
                    validation.get("gap_signal_validation", {})
                    .get("is_valid_signal", False)
                )

                if is_valid_signal is True:
                    new_related_askr.append(deepcopy(askr_item))
                    valid_askr_items_count += 1

            if new_related_askr:
                new_extraction_obj = {
                    "workload_type": extraction_obj.get("workload_type"),
                    "workload_summary": extraction_obj.get("workload_summary"),
                    "workload_quote": extraction_obj.get("workload_quote"),
                    "workload_validation": deepcopy(extraction_obj.get("workload_validation", {})),
                    "related_askr": new_related_askr,
                }
                new_record["extraction"]["extractions"].append(new_extraction_obj)

        if new_record["extraction"]["extractions"]:
            valid_output["records"].append(new_record)
            kept_records_count += 1

    valid_output["summary"] = {
        "input_records_count": input_records_count,
        "kept_records_count": kept_records_count,
        "removed_records_count": input_records_count - kept_records_count,
        "valid_askr_items_count": valid_askr_items_count,
    }
    return valid_output


def finalize_validation_summary(validation_output: Dict[str, Any], failed_calls_count: int) -> None:
    total_records = len(validation_output.get("records", []))
    total_extractions = 0
    total_askr_items = 0
    valid_signal_count = 0
    invalid_signal_count = 0
    failed_item_count = 0
    not_grounded_count = 0

    for record in validation_output.get("records", []):
        for extraction_obj in record.get("extraction", {}).get("extractions", []):
            total_extractions += 1
            for askr_item in extraction_obj.get("related_askr", []):
                total_askr_items += 1
                validation = askr_item.get("validation", {})

                if validation.get("validation_status") == "failed":
                    failed_item_count += 1
                    continue

                if validation.get("validation_status") == "completed":
                    if validation.get("gap_signal_validation", {}).get("is_valid_signal") is True:
                        valid_signal_count += 1
                    else:
                        invalid_signal_count += 1

                    grounding_status = validation.get("grounding_validation", {}).get("status")
                    if grounding_status == "not_grounded":
                        not_grounded_count += 1

    validation_output["summary"] = {
        "input_records_count": total_records,
        "validated_records_count": total_records,
        "validated_extractions_count": total_extractions,
        "validated_askr_items_count": total_askr_items,
        "valid_signal_count": valid_signal_count,
        "invalid_signal_count": invalid_signal_count,
        "failed_item_count": failed_item_count,
        "not_grounded_count": not_grounded_count,
        "failed_validation_calls_count": failed_calls_count,
    }


# ============================================================
# Skip logic
# ============================================================

def already_processed(
    file_name: str,
    validation_output_dir: Path,
    valid_records_output_dir: Path
) -> bool:
    validation_file = validation_output_dir / file_name
    valid_file = valid_records_output_dir / file_name
    return validation_file.exists() and valid_file.exists()


# ============================================================
# Error classification
# ============================================================

def classify_failure(error_message: str) -> str:
    msg = error_message.lower()

    if "timeout" in msg:
        return "timeout"
    if "json" in msg:
        return "invalid_json"
    if "schema" in msg or "missing fields" in msg or "length mismatch" in msg:
        return "schema_mismatch"
    if "api" in msg or "rate limit" in msg or "quota" in msg:
        return "api_error"
    if "empty_response" in msg:
        return "empty_response"
    return "runtime_error"


# ============================================================
# Per-file processing
# ============================================================

def process_file(
    file_path: Path,
    validation_output_dir: Path,
    valid_records_output_dir: Path,
    logs_dir: Path,
    prompt_data: Dict[str, Any],
    client: OpenAI,
) -> None:
    source_data = load_json(file_path)
    paper = source_data.get("paper", {})
    paper_key = paper.get("paper_key", file_path.stem)

    validation_output = initialize_validation_output(source_data)

    model = prompt_data["config"]["model"]
    max_output_tokens = int(prompt_data["config"].get("max_output_tokens", 700))
    system_template = prompt_data["system_command"]
    user_template = prompt_data["user_command"]

    failed_calls_count = 0

    records = validation_output.get("records", [])

    for record in records:
        unit = record.get("unit", {})
        raw_text = unit.get("raw_text", "")
        unit_id = unit.get("unit_id", "")

        extractions = record.get("extraction", {}).get("extractions", [])

        for extraction_index, extraction_obj in enumerate(extractions):
            workload_type = extraction_obj.get("workload_type", "")
            workload_summary = extraction_obj.get("workload_summary", "")
            workload_quote = extraction_obj.get("workload_quote", "")
            related_askr = extraction_obj.get("related_askr", [])

            if not related_askr:
                continue

            variables = {
                "workload_type": workload_type,
                "raw_text": raw_text,
                "workload_summary": workload_summary,
                "workload_quote": workload_quote,
                "related_askr_json": json.dumps(related_askr, ensure_ascii=False, indent=2),
            }

            system_prompt = render_prompt(system_template, variables)
            user_prompt = render_prompt(user_template, variables)

            log_base = build_log_base_name(paper_key, unit_id, extraction_index)
            request_log_path = logs_dir / paper_key / "requests" / f"{log_base}.txt"
            response_log_path = logs_dir / paper_key / "responses" / f"{log_base}.txt"
            error_log_path = logs_dir / paper_key / "errors" / f"{log_base}.txt"

            request_text = (
                "=== SYSTEM ===\n"
                f"{system_prompt}\n\n"
                "=== USER ===\n"
                f"{user_prompt}\n"
            )
            save_log_text(request_log_path, request_text)

            validator_output = None
            last_error = ""
            last_meta: Dict[str, Any] = {}

            for attempt in range(3):
                try:
                    raw_response, meta = call_openai_json(
                        client=client,
                        model=model,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_output_tokens=max_output_tokens,
                    )
                    last_meta = meta
                    save_log_text(response_log_path, raw_response)

                    parsed = json.loads(raw_response)
                    ok, error_msg = validate_validator_output(
                        parsed,
                        expected_askr_count=len(related_askr)
                    )
                    if ok:
                        validator_output = parsed
                        break
                    else:
                        last_error = error_msg

                except Exception as e:
                    if isinstance(e, EmptyResponseError):
                        last_meta = e.meta
                        save_log_text(response_log_path, "")
                    last_error = str(e)
                    time.sleep(1 * (2 ** attempt))

            if validator_output is None:
                failed_calls_count += 1
                error_type = classify_failure(last_error)
                meta_text = json.dumps(last_meta, ensure_ascii=False)
                save_log_text(
                    error_log_path,
                    f"Validation failed after retry.\n"
                    f"Error type: {error_type}\n"
                    f"Error: {last_error}\n"
                    f"Response meta: {meta_text}\n"
                )

                attach_failed_validation_to_extraction(
                    extraction_obj=extraction_obj,
                    error_type=error_type,
                    error_message=last_error
                )
            else:
                attach_validation_to_full_output(
                    validation_record=record,
                    extraction_index=extraction_index,
                    validator_output=validator_output
                )

    finalize_validation_summary(validation_output, failed_calls_count)
    valid_output = build_valid_records_output(validation_output)

    validation_output_path = validation_output_dir / file_path.name
    valid_output_path = valid_records_output_dir / file_path.name

    save_json(validation_output_path, validation_output)
    save_json(valid_output_path, valid_output)


# ============================================================
# Main
# ============================================================

def main() -> None:
    print("Gap Signal Validation Runner")
    input_dir_str = input("Please enter the input directory: ").strip()
    output_dir_str = input("Please enter the output directory: ").strip()
    prompt_file_path_str = input("Please enter the prompt file path: ").strip()

    input_dir = Path(input_dir_str)
    output_dir = Path(output_dir_str)
    validation_output_dir = output_dir / "validation"
    valid_records_output_dir = output_dir / "valid_records"
    prompt_file_path = Path(prompt_file_path_str)

    if not input_dir.exists() or not input_dir.is_dir():
        print("Invalid input directory.")
        return

    if not prompt_file_path.exists():
        print("Prompt file path does not exist.")
        return

    prompt_data = load_yaml(prompt_file_path)

    provider = prompt_data.get("config", {}).get("provider")
    if provider != "openai":
        print(f"Unsupported provider in this runner: {provider}")
        return

    client = build_openai_client()

    validation_output_dir.mkdir(parents=True, exist_ok=True)
    valid_records_output_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = validation_output_dir / "_logs"

    json_files = sorted(input_dir.glob("*.json"))
    if not json_files:
        print("No JSON files found in the input directory.")
        return

    total_files = len(json_files)
    print(f"Found {total_files} JSON files.")

    processed_count = 0
    skipped_count = 0

    for idx, file_path in enumerate(json_files, start=1):
        try:
            if already_processed(
                file_name=file_path.name,
                validation_output_dir=validation_output_dir,
                valid_records_output_dir=valid_records_output_dir,
            ):
                skipped_count += 1
                print(f"[{idx}/{total_files}] Skipped already processed: {file_path.name}")
                continue

            print(f"[{idx}/{total_files}] Processing: {file_path.name}")
            process_file(
                file_path=file_path,
                validation_output_dir=validation_output_dir,
                valid_records_output_dir=valid_records_output_dir,
                logs_dir=logs_dir,
                prompt_data=prompt_data,
                client=client,
            )
            processed_count += 1
            print(f"[{idx}/{total_files}] Done: {file_path.name}")

        except Exception as e:
            print(f"[{idx}/{total_files}] ERROR: {file_path.name} -> {e}")

    print("Finished.")
    print(f"Processed files: {processed_count}")
    print(f"Skipped files: {skipped_count}")


if __name__ == "__main__":
    main()
