# src/prompt_runner.py
import yaml
from pathlib import Path
from openai import OpenAI
import json


class PromptRunner:
    def __init__(self, prompt_path: str, model: str = "gpt-4o-mini"):
        self.prompt_cfg = self._load_prompt(prompt_path)
        self.model = model
        self.client = OpenAI()  # Reads OPENAI_API_KEY from env

    def _load_prompt(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_messages(self, **kwargs):
        """Inject placeholders into the prompt YAML."""
        system_msg = {"role": "system", "content": self.prompt_cfg["system"]}
        user_msg = {
            "role": "user",
            "content": self.prompt_cfg["user"].format(**kwargs),
        }
        return [system_msg, user_msg]

    def run(self, **kwargs):
        """Run the prompt with given paper content and return JSON result."""
        messages = self.build_messages(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Example usage for quick test
    runner = PromptRunner("prompts/screening_prompt.yaml")

    paper_data = {
        "criteria": "Include if full HCD cycle AND discusses TASKS/barriers from designer perspective",
        "title": "Human-Centered Design in Healthcare IT",
        "abstract": "This study applies all four phases of HCD...",
        "methodology": "We conducted participatory design...",
        "results": "Designers reported workload issues...",
        "discussion": "Findings suggest barriers align with TASKS elements...",
        "conclusion": "The study highlights...",
    }

    result = runner.run(**paper_data)

    try:
        parsed = json.loads(result)
        print("✅ Parsed JSON:", parsed)
    except json.JSONDecodeError:
        print("⚠️ Raw output (not valid JSON):", result)
