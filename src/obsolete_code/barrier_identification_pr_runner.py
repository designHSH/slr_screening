"""
This code does not work properly it includes other functionality that impact its original task.
Prompt Runner for HCD Barrier Extraction
Compatible with both GPT-4.1 and GPT-5 families
------------------------------------------------
Author: Hamed
Date:   2025-11-12
"""

import os, json, time, csv, glob, uuid, hashlib, yaml
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# ----------------------------
# 1. Load environment variables
# ----------------------------
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ----------------------------
# 2. Utility helpers
# ----------------------------
def sha1_of_file(path):
    with open(path, "rb") as f:
        return hashlib.sha1(f.read()).hexdigest()

def read_yaml(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def safe_filename(stem):
    return os.path.splitext(os.path.basename(stem))[0]

def timestamp_utc():
    return datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")

# ----------------------------
# 3. Core runner functions
# ----------------------------
def build_messages(system_text, user_template, input_text):
    """Fill placeholder and build OpenAI messages list."""
    user_filled = user_template.format(input_text=input_text)
    return [
        {"role": "system", "content": system_text},
        {"role": "user", "content": user_filled},
    ]

def call_model(messages, model, temperature, top_p,
               max_tokens, timeout, retries, backoff):
    """
    Send a request to GPT model with retry & timeout logic.
    Compatible with both GPT-4.1 and GPT-5 families.
    """

    is_gpt5 = model.startswith("gpt-5")

    for attempt in range(retries + 1):
        try:
            # Choose correct token parameter name
            token_param = (
                {"max_completion_tokens": max_tokens}
                if is_gpt5
                else {"max_output_tokens": max_tokens}
            )

            # Build parameter dictionary dynamically
            params = {
                "model": model,
                "messages": messages,
                "top_p": top_p,
                "timeout": timeout,
                **token_param,
            }

            # GPT-5: do NOT include temperature (rejects non-default values)
            # GPT-4.1: include temperature from YAML
            if not is_gpt5:
                params["temperature"] = temperature

            # Call the model
            response = client.chat.completions.create(**params)

            return response.choices[0].message.content.strip()

        except Exception as e:
            if attempt < retries:
                wait = backoff * (2 ** attempt)
                print(f"⚠️  Error: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                raise e

def validate_json(raw_text, file_name):
    """Ensure returned text is valid JSON and minimal schema is present."""
    if not raw_text.strip():
        raise ValueError(f"{file_name}: empty output from model")

    try:
        data = json.loads(raw_text)
        if "barriers" not in data:
            raise ValueError("Missing 'barriers' field")
        return data
    except json.JSONDecodeError as e:
        # Optional: try simple JSON repair or fallback
        if raw_text.strip().startswith("No barriers"):
            # Convert to valid JSON
            return {"file_name": file_name, "barriers": [], "note": "No barriers identified"}
        raise ValueError(f"{file_name}: invalid JSON output ({e})")


# ----------------------------
# 4. Main run procedure
# ----------------------------
def run_prompt(yaml_path, input_dir, output_dir):
    cfg = read_yaml(yaml_path)

    # Extract configuration
    model       = cfg["config"]["model"]
    temperature = cfg["config"].get("temperature", 0)
    top_p       = cfg["config"].get("top_p", 1)
    max_tokens  = cfg["config"].get("max_output_tokens", 2000)
    retries     = cfg["policy"].get("max_retries", 3)
    backoff     = cfg["policy"].get("retry_backoff_sec", 2)
    timeout     = cfg["policy"].get("request_timeout_sec", 180)
    system_text = cfg["system_command"]
    user_text   = cfg["user_command"]
    prompt_version = cfg.get("version", "unknown")

    # Prepare folders
    os.makedirs(output_dir, exist_ok=True)
    papers_json_dir = os.path.join(output_dir, "papers_json")
    os.makedirs(papers_json_dir, exist_ok=True)
    review_csv_path = os.path.join(output_dir, "barriers_review.csv")
    errors_log_path = os.path.join(output_dir, "errors.log")

    # Init CSV
    with open(review_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "paper_id", "file_name", "barrier_local_id", "actor",
            "barrier_type", "process_workloads", "operational_workloads",
            "description", "reference_excerpt", "certainty",
            "reviewer_decision", "reviewer_notes"
        ])

    manifest = {
        "run_id": f"{timestamp_utc()}_{model}_{prompt_version}",
        "created_at_utc": timestamp_utc(),
        "model": model,
        "prompt_version": prompt_version,
        "temperature": temperature,
        "input_dir": input_dir,
        "output_dir": output_dir,
        "n_papers": 0,
        "n_success": 0,
        "n_errors": 0
    }

    for path in glob.glob(os.path.join(input_dir, "*.txt")):
        file_name = os.path.basename(path)
        print(f"▶ Processing {file_name}")
        manifest["n_papers"] += 1

        try:
            # Read paper text
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            source_sha1 = sha1_of_file(path)

            # Build messages and call GPT
            messages = build_messages(system_text, user_text, text)
            raw_output = call_model(messages, model, temperature, top_p,
                                    max_tokens, timeout, retries, backoff)
            data = validate_json(raw_output, file_name)

            print(f"DEBUG raw_output for {file_name}:\n{repr(raw_output)}\n---")


            # Add metadata
            data["file_name"] = file_name
            data["paper_id"] = hashlib.sha1(file_name.encode()).hexdigest()
            data["model"] = model
            data["prompt_version"] = prompt_version
            data["source_sha1"] = source_sha1
            data["timestamp_utc"] = timestamp_utc()

            # Write per-paper JSON
            out_path = os.path.join(papers_json_dir,
                                    safe_filename(file_name) + "_barriers.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

            # Append to review CSV
            with open(review_csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                for b in data.get("barriers", []):
                    process_w = ";".join(
                        b.get("workloads_blocked", {}).get("process", []))
                    oper_w = ";".join(
                        b.get("workloads_blocked", {}).get("operational", []))
                    writer.writerow([
                        data["paper_id"], file_name,
                        b.get("barrier_id", ""),
                        b.get("actor", ""),
                        b.get("barrier_type", ""),
                        process_w, oper_w,
                        b.get("description", ""),
                        b.get("reference_excerpt", ""),
                        b.get("certainty", ""),
                        "", ""  # reviewer_decision, reviewer_notes
                    ])

            manifest["n_success"] += 1

        except Exception as e:
            manifest["n_errors"] += 1
            with open(errors_log_path, "a", encoding="utf-8") as f:
                f.write(f"{timestamp_utc()} {file_name}: {e}\n")
            print(f"❌ Error in {file_name}: {e}")

    # Write manifest
    with open(os.path.join(output_dir, "manifest.json"),
              "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✅ Completed run. {manifest['n_success']} succeeded, "
          f"{manifest['n_errors']} failed out of {manifest['n_papers']} papers.")
    print(f"Outputs saved in: {output_dir}")


# ----------------------------
# 5. Example usage (manual)
# ----------------------------
if __name__ == "__main__":
    run_prompt(
        yaml_path="prompt/barrier_identification_pr.yaml",          # your YAML prompt file
        input_dir=r"data\test_data\barrier_identification_test\input_sample",       # folder with paper text files
        output_dir=r"data\test_data\barrier_identification_test\output\1st_run" # where outputs go
    )
