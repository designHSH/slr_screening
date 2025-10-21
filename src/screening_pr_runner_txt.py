# src/prompt_runner_text.py
import yaml
import json
import csv
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI


class PromptRunnerText:
    """
    Executes structured YAML-based prompts on text files using GPT models (e.g., gpt-4.1-mini).
    Reads .txt input files, injects content into the user_command ({{paper_text}}),
    and outputs aggregated results as a single CSV file.
    """

    def __init__(self, prompt_path: str, model: str = "gpt-4.1-mini"):
        load_dotenv()
        self.prompt_cfg = self._load_prompt(prompt_path)
        self.model = model
        self.client = OpenAI()  # Uses OPENAI_API_KEY from environment

    def _load_prompt(self, path: str) -> dict:
        """Load and validate YAML prompt configuration."""
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "system_command" not in cfg or "user_command" not in cfg:
            raise ValueError("❌ YAML must contain both 'system_command' and 'user_command'.")
        return cfg

    def _read_text_file(self, file_path: Path) -> str:
        """Read and return content of a text file."""
        if not file_path.exists():
            raise FileNotFoundError(f"❌ Text file not found: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text

    def _build_messages(self, text_content: str):
        """Construct system and user messages for GPT."""
        user_command = self.prompt_cfg["user_command"].replace("{{paper_text}}", text_content)

        system_msg = {"role": "system", "content": self.prompt_cfg["system_command"]}
        user_msg = {"role": "user", "content": user_command}
        return [system_msg, user_msg]

    def run(self, text_path: Path) -> dict:
        """Run the prompt on a single text file and return structured result."""
        text_content = self._read_text_file(text_path)
        messages = self._build_messages(text_content)

        task_name = self.prompt_cfg.get("prompt_task", "unnamed-task")
        print(f"🚀 Running task '{task_name}' on {text_path.name} using model {self.model}...")

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=6000,  # allows for large 8–10k token inputs
        )

        result_text = response.choices[0].message.content.strip()

        # Parse and normalize output
        result = {"file_name": text_path.name}
        try:
            parsed = json.loads(result_text)
            print("✅ Parsed JSON successfully.")
            result.update(parsed)
        except json.JSONDecodeError:
            print("⚠️ Output not valid JSON — returning raw text.")
            result["raw_output"] = result_text

        return result

    @staticmethod
    def _flatten_result(result: dict) -> dict:
        """Flatten nested lists/dicts into CSV-friendly strings."""
        flat = {}
        for k, v in result.items():
            if isinstance(v, list):
                flat[k] = "; ".join(
                    [json.dumps(i, ensure_ascii=False) if isinstance(i, dict) else str(i) for i in v]
                )
            elif isinstance(v, dict):
                flat[k] = json.dumps(v, ensure_ascii=False)
            else:
                flat[k] = v
        return flat

    @staticmethod
    def save_results_csv(results: list, output_path: Path):
        """Save all aggregated results into a single CSV file."""
        if not results:
            print("⚠️ No results to save.")
            return

        flattened = [PromptRunnerText._flatten_result(r) for r in results]
        fieldnames = sorted({key for d in flattened for key in d.keys()})

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(flattened)

        print(f"💾 Saved aggregated CSV results → {output_path}")


if __name__ == "__main__":
    # === CONFIGURATION ===
    input_folder = Path(r"data\test_data\validation\claned_text")        # Folder containing .txt files
    prompt_file = Path(r"prompt\full_text_screening_pr.yaml")
    output_csv_path = Path(r"data\test_data\validation\screening _result\full_screening_results.csv")

    runner = PromptRunnerText(prompt_file, model="gpt-4.1-mini")

    text_files = list(input_folder.glob("*.txt"))
    if not text_files:
        print("⚠️ No text files found in input folder.")
    else:
        all_results = []
        for text_file in text_files:
            try:
                result = runner.run(text_path=text_file)
                all_results.append(result)
            except Exception as e:
                print(f"❌ Error processing {text_file.name}: {e}")

        # Save all results as a single CSV file
        runner.save_results_csv(all_results, output_csv_path)
