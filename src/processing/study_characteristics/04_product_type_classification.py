from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from dotenv import load_dotenv
from google import genai
from google.genai import types


# ---------------------------------------------------------
# This script:
# - reads a CSV file
# - reads a prompt from a YAML file
# - sends designed_solution_quote to Gemini
# - parses the JSON response
# - adds the response as new columns
# - saves the result to a new CSV file
#
# Required CSV columns:
# - paper_key
# - designed_solution_quote
#
# Required .env variable:
# - GEMINI_API_KEY
# ---------------------------------------------------------


def normalize_empty(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def load_prompt_yaml(yaml_path: Path) -> dict[str, Any]:
    with yaml_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    required_top_keys = ["config", "user_command", "system_command"]
    missing = [key for key in required_top_keys if key not in data]
    if missing:
        raise ValueError(f"Missing required YAML key(s): {', '.join(missing)}")

    if "provider" not in data["config"] or "model" not in data["config"]:
        raise ValueError("YAML config must contain 'provider' and 'model'.")

    return data


def build_user_prompt(template: str, designed_solution_quote: str) -> str:
    has_placeholders = "{designed_solution_quote}" in template
    if has_placeholders:
        try:
            return template.format(
                designed_solution_quote=designed_solution_quote,
            )
        except KeyError as e:
            raise ValueError(f"Missing placeholder in user_command template: {e}")

    return (
        template.rstrip()
        + "\n\n"
        + f"designed_solution_quote: {designed_solution_quote}"
    )


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if match:
        candidate = match.group(0)
        parsed = json.loads(candidate)
        if isinstance(parsed, dict):
            return parsed

    raise ValueError("Could not parse JSON from model response.")


def classify_product_type(
    client: genai.Client,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    response_mime_type: str,
    thinking_level: str | None,
    logger: logging.Logger,
    paper_key: str,
    row_index: int,
) -> dict[str, str]:
    try:
        cfg_kwargs = {
            "system_instruction": system_prompt,
            "temperature": temperature,
            "response_mime_type": response_mime_type,
        }
        if thinking_level:
            cfg_kwargs["thinking_config"] = types.ThinkingConfig(thinking_level=thinking_level)

        logger.info("Calling Gemini | row=%d | paper_key=%s", row_index, paper_key)
        t0 = time.time()
        response = client.models.generate_content(
            model=model_name,
            contents=user_prompt,
            config=types.GenerateContentConfig(**cfg_kwargs),
        )
        elapsed = time.time() - t0
        logger.info("Gemini response received | row=%d | paper_key=%s | elapsed=%.2fs", row_index, paper_key, elapsed)
    except Exception:
        logger.exception("Gemini generate_content failed.")
        raise

    raw_text = response.text or ""
    logger.info(
        "Raw response length | row=%d | paper_key=%s | chars=%d",
        row_index,
        paper_key,
        len(raw_text),
    )
    parsed = extract_json(raw_text)
    logger.info(
        "Parsed keys | row=%d | paper_key=%s | keys=%s",
        row_index,
        paper_key,
        sorted(parsed.keys()),
    )

    return {
        "product_type_class": normalize_empty(parsed.get("product_type_class")),
        "product_type_subclass": normalize_empty(parsed.get("product_type_subclass")),
        "manual_review_needed": normalize_empty(parsed.get("manual_review_needed")),
        "coding_note": normalize_empty(parsed.get("coding_note")),
        "product_type_llm_raw_response": raw_text,
    }


def setup_logging(log_file: Path) -> logging.Logger:
    logger = logging.getLogger("product_type_classification")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    return logger


def main() -> None:
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY was not found in the .env file.")

    print("Product type classification script using Gemini")
    print("-" * 50)

    input_file_str = input("Enter the full path to the input CSV file: ").strip().strip('"')
    output_file_str = input("Enter the full path to the output CSV file (include file name): ").strip().strip('"')
    prompt_yaml_str = input("Enter the full path to the prompt YAML file: ").strip().strip('"')

    input_file = Path(input_file_str)
    output_file = Path(output_file_str)
    prompt_yaml_file = Path(prompt_yaml_str)

    if not input_file.exists():
        raise FileNotFoundError(f"Input CSV file not found: {input_file}")

    if input_file.suffix.lower() not in [".csv"]:
        raise ValueError("Input file must be a CSV file (.csv).")

    if not prompt_yaml_file.exists():
        raise FileNotFoundError(f"Prompt YAML file not found: {prompt_yaml_file}")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    log_file = output_file.parent / "log.txt"
    logger = setup_logging(log_file)

    prompt_data = load_prompt_yaml(prompt_yaml_file)

    provider = normalize_empty(prompt_data["config"].get("provider"))
    model_name = normalize_empty(prompt_data["config"].get("model"))
    temperature = float(prompt_data["config"].get("temperature", 0.0))
    response_mime_type = normalize_empty(prompt_data["config"].get("response_mime_type", "application/json"))
    thinking_level = normalize_empty(
        prompt_data["config"].get("THINKING_LEVEL") or prompt_data["config"].get("thinking_level")
    ) or None
    system_prompt = normalize_empty(prompt_data.get("system_command"))
    user_template = normalize_empty(prompt_data.get("user_command"))

    if provider.lower() not in {"gemini", "google"}:
        raise ValueError(
            f"Unsupported provider in YAML: {provider}. This script currently supports only 'google'/'gemini'."
        )

    client = genai.Client(api_key=api_key)

    logger.info("Product type classification script starting.")
    logger.info("Log file: %s", log_file)
    logger.info("Provider: %s", provider)
    logger.info("Model: %s", model_name)
    logger.info("Temperature: %s", temperature)
    logger.info("Response MIME type: %s", response_mime_type)
    if thinking_level:
        logger.info("Thinking level: %s", thinking_level)

    logger.info("Reading CSV file...")
    df = pd.read_csv(input_file, dtype=str, keep_default_na=False)

    required_columns = ["paper_key", "designed_solution_quote"]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required CSV column(s): {', '.join(missing)}")

    # Resume mode: if output exists, skip rows that already have product_type_class
    existing_map: dict[str, dict[str, str]] = {}
    if output_file.exists():
        logger.info("Resume mode enabled (output exists). Loading: %s", output_file)
        out_df = pd.read_csv(output_file, dtype=str, keep_default_na=False)
        if "paper_key" in out_df.columns and "product_type_class" in out_df.columns:
            dup_keys = out_df["paper_key"][out_df["paper_key"].duplicated()].unique().tolist()
            if dup_keys:
                logger.warning("Duplicate paper_key(s) in output: %s", dup_keys[:10])
            for _, r in out_df.iterrows():
                pk = normalize_empty(r.get("paper_key", ""))
                if not pk:
                    continue
                existing_map[pk] = {
                    "product_type_class": normalize_empty(r.get("product_type_class", "")),
                    "product_type_subclass": normalize_empty(r.get("product_type_subclass", "")),
                    "manual_review_needed": normalize_empty(r.get("manual_review_needed", "")),
                    "coding_note": normalize_empty(r.get("coding_note", "")),
                    "product_type_llm_raw_response": normalize_empty(r.get("product_type_llm_raw_response", "")),
                }
        else:
            logger.warning("Resume mode skipped: output missing required columns.")

    results = []
    total_rows = len(df)
    logger.info("Processing %d rows...", total_rows)

    success_count = 0
    fail_count = 0
    skipped_count = 0

    try:
        for idx, row in df.iterrows():
            paper_key = normalize_empty(row.get("paper_key", ""))
            designed_solution_quote = normalize_empty(row.get("designed_solution_quote", ""))

            logger.info("Processing row %d/%d | paper_key=%s", idx + 1, total_rows, paper_key)
            logger.info(
                "Inputs | paper_key=%s | designed_solution_quote=%s",
                paper_key,
                designed_solution_quote,
            )

            try:
                existing = existing_map.get(paper_key)
                if existing and normalize_empty(existing.get("product_type_class")):
                    skipped_count += 1
                    logger.info(
                        "Skipping row %d | paper_key=%s | reason=already classified",
                        idx + 1,
                        paper_key,
                    )
                    result = existing
                elif not designed_solution_quote:
                    skipped_count += 1
                    logger.warning(
                        "Skipped row %d | paper_key=%s | reason=empty designed_solution_quote",
                        idx + 1,
                        paper_key,
                    )
                    result = {
                        "product_type_class": "",
                        "product_type_subclass": "",
                        "manual_review_needed": "Yes",
                        "coding_note": "designed_solution_quote is empty.",
                        "product_type_llm_raw_response": "",
                    }
                else:
                    user_prompt = build_user_prompt(
                        template=user_template,
                        designed_solution_quote=designed_solution_quote,
                    )
                    logger.info(
                        "User prompt prepared | row=%d | paper_key=%s | chars=%d",
                        idx + 1,
                        paper_key,
                        len(user_prompt),
                    )
                    result = classify_product_type(
                        client=client,
                        model_name=model_name,
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        temperature=temperature,
                        response_mime_type=response_mime_type,
                        thinking_level=thinking_level,
                        logger=logger,
                        paper_key=paper_key,
                        row_index=idx + 1,
                    )
                    success_count += 1

            except Exception as e:
                fail_count += 1
                logger.exception("Row %d failed due to API/parsing error. paper_key=%s", idx + 1, paper_key)
                result = {
                    "product_type_class": "",
                    "product_type_subclass": "",
                    "manual_review_needed": "Yes",
                    "coding_note": f"API/parsing error: {str(e)}",
                    "product_type_llm_raw_response": "",
                }

            results.append(result)
            if (idx + 1) % 25 == 0:
                logger.info(
                    "Progress checkpoint | processed=%d | success=%d | skipped=%d | failed=%d",
                    idx + 1,
                    success_count,
                    skipped_count,
                    fail_count,
                )

    finally:
        client.close()

    results_df = pd.DataFrame(results)
    output_df = pd.concat([df, results_df], axis=1)

    logger.info("Saving output CSV file...")
    output_df.to_csv(output_file, index=False)

    logger.info("Done.")
    logger.info("Saved file: %s", output_file)
    logger.info(
        "Run summary | total=%d | success=%d | skipped=%d | failed=%d",
        total_rows,
        success_count,
        skipped_count,
        fail_count,
    )


if __name__ == "__main__":
    main()
