"""
Barrier identification Prompt Runner
-------------------------
This lightweight script automates running a YAML-defined GPT prompt over a folder of text files.

Workflow:
1. Loads environment variables from `.env` (expects OPENAI_API_KEY).
2. Parses the YAML prompt (system + user commands, model config, metadata).
3. Iterates through all `.txt` files in the input directory.
4. Sends each file’s text to the specified model (e.g., gpt-4.1).
5. Saves the model’s JSON output with the same base name in the output folder.
6. Ensures the "file_name" field appears first in each JSON output.
7. Logs progress, timing, and errors to `run.log` inside the output folder.

Minimal dependencies: openai, pyyaml, python-dotenv.
Edit the three paths (prompt, input, output) in the __main__ block before running.

Revision: 0.1.0
"""



import os
import json
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import yaml
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class PromptSpec:
    task: str
    version: str
    system_command: str
    user_command: str
    model: str
    top_p: float | int | None


class PromptRunner:
    def __init__(self, prompt_yaml_path: Path):
        self.spec = self._load_prompt(prompt_yaml_path)
        self.client = OpenAI()

    @staticmethod
    def _load_prompt(path: Path) -> PromptSpec:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        task = data.get("task", "")
        version = str(data.get("version", ""))
        system_command = data.get("system_command", "")
        user_command = data.get("user_command", "")
        cfg = data.get("config", {}) or {}
        return PromptSpec(
            task=task,
            version=version,
            system_command=system_command,
            user_command=user_command,
            model=cfg.get("model", "gpt-4.1"),
            top_p=cfg.get("top_p"),
        )

    def build_messages(self, input_text: str) -> list[dict]:
        user = self.spec.user_command.replace("{input_text}", input_text)
        return [
            {"role": "system", "content": self.spec.system_command},
            {"role": "user", "content": user},
        ]

    def run_once(self, input_text: str) -> str:
        messages = self.build_messages(input_text)
        resp = self.client.chat.completions.create(
            model=self.spec.model,
            messages=messages,
            top_p=self.spec.top_p if self.spec.top_p is not None else 1,
            temperature=0,
        )
        return resp.choices[0].message.content.strip()


def ensure_filename_first(model_json_text: str, file_name: str) -> str:
    try:
        data = json.loads(model_json_text)
        if not isinstance(data, dict):
            raise ValueError("Top-level JSON is not an object")
        data["file_name"] = file_name
        ordered = OrderedDict()
        ordered["file_name"] = data["file_name"]
        for k, v in data.items():
            if k == "file_name":
                continue
            ordered[k] = v
        return json.dumps(ordered, ensure_ascii=False, indent=2)
    except Exception:
        fallback = OrderedDict(
            [
                ("file_name", file_name),
                ("barriers", []),
                ("note", "No barriers identified (model output not valid JSON)"),
            ]
        )
        return json.dumps(fallback, ensure_ascii=False, indent=2)


def append_log(log_path: Path, line: str):
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line.rstrip() + "\n")


def stamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main(prompt_yaml: Path, input_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "run.log"

    load_dotenv()
    runner = PromptRunner(prompt_yaml)

    txt_files = sorted(p for p in input_dir.glob("*.txt") if p.is_file())
    total = len(txt_files)
    header = f"{stamp()} | START | model={runner.spec.model} | prompt={runner.spec.task} v{runner.spec.version} | files={total}"
    print(header); append_log(log_file, header)

    for idx, txt_path in enumerate(txt_files, start=1):
        start_line = f"{stamp()} | [{idx}/{total}] | START | {txt_path.name}"
        print(start_line); append_log(log_file, start_line)

        try:
            with open(txt_path, "r", encoding="utf-8") as f:
                input_text = f.read()

            call_line = f"{stamp()} | [{idx}/{total}] | CALL  | sending to {runner.spec.model}"
            print(call_line); append_log(log_file, call_line)
            t0 = time.time()

            raw = runner.run_once(input_text)

            dt = time.time() - t0
            recv_line = f"{stamp()} | [{idx}/{total}] | RECV  | {len(raw)} chars in {dt:.2f}s"
            print(recv_line); append_log(log_file, recv_line)

            final_json_text = ensure_filename_first(raw, txt_path.name)
            out_path = output_dir / (txt_path.stem + ".json")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(final_json_text)

            save_line = f"{stamp()} | [{idx}/{total}] | SAVE  | {out_path.name}"
            print(save_line); append_log(log_file, save_line)

        except Exception as e:
            err_line = f"{stamp()} | [{idx}/{total}] | ERROR | {txt_path.name} | {type(e).__name__}: {e}"
            print(err_line); append_log(log_file, err_line)
            # Save a minimal fallback JSON to keep the run moving
            out_path = output_dir / (txt_path.stem + ".json")
            fallback = OrderedDict(
                [
                    ("file_name", txt_path.name),
                    ("barriers", []),
                    ("note", f"Processing error: {type(e).__name__}"),
                ]
            )
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(json.dumps(fallback, ensure_ascii=False, indent=2))
            saved_err = f"{stamp()} | [{idx}/{total}] | SAVE  | fallback -> {out_path.name}"
            print(saved_err); append_log(log_file, saved_err)

    footer = f"{stamp()} | DONE  | processed={total}"
    print(footer); append_log(log_file, footer)


if __name__ == "__main__":
    # === EDIT THESE THREE PATHS ===
    PROMPT_YAML_PATH = Path(r"prompt\barrier_identification_pr.yaml")
    INPUT_DIR = Path(r"data\test_data\barrier_identification_test\1st_run\input_sample")
    OUTPUT_DIR = Path(r"data\test_data\barrier_identification_test\4rd-run-gpt4.1")
    # ==============================

    main(PROMPT_YAML_PATH, INPUT_DIR, OUTPUT_DIR)
