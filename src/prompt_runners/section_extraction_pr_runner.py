import yaml
import os
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI

# Load environment variables (including OPENAI_API_KEY)
load_dotenv()

class PromptRunner:
    def __init__(self, prompt_path: str, model: str = "gpt-4.1-mini"):
        """Initialize the runner with YAML prompt config and OpenAI client."""
        self.prompt_cfg = self._load_prompt(prompt_path)
        self.model = model
        self.client = OpenAI()

    def _load_prompt(self, path: str):
        """Load the YAML prompt configuration file."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def build_messages(self, **kwargs):
        """Inject placeholders into YAML prompt and prepare messages."""
        system_msg = {"role": "system", "content": self.prompt_cfg["system_command"]}
        user_msg = {
            "role": "user",
            "content": self.prompt_cfg["user_command"].format(**kwargs),
        }
        return [system_msg, user_msg]

    def run(self, **kwargs):
        """Send messages to GPT-4.1-mini and return plain text output."""
        messages = self.build_messages(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1
            
        )
        return response.choices[0].message.content.strip()


def clean_paper_text(prompt_yaml: str, input_path: str, output_dir: str = None):
    """Clean a single paper text file and save output."""
    runner = PromptRunner(prompt_yaml)

    with open(input_path, "r", encoding="utf-8") as f:
        paper_text = f.read()

    cleaned_text = runner.run(paper_text=paper_text)

    input_file = Path(input_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = Path(output_dir) / input_file.name
    else:
        output_path = input_file

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)

    print(f"✅ Cleaned text saved to: {output_path}")
    return cleaned_text


def batch_clean_papers(prompt_yaml: str, input_folder: str, output_folder: str):
    """Loop through all .txt files in input_folder and clean them."""
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    os.makedirs(output_folder, exist_ok=True)

    txt_files = list(input_folder.glob("*.txt"))
    print(f"Found {len(txt_files)} files to process.\n")

    for file_path in txt_files:
        print(f"🧾 Processing: {file_path.name}")
        try:
            clean_paper_text(prompt_yaml, str(file_path), str(output_folder))
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")


if __name__ == "__main__":
    """
    Example batch usage:
    python src/prompt_runner_cleaner.py
    """
    prompt_yaml = r"prompt\section_exctraction_pr.yaml"
    input_folder = r"data\test_data\sorted_by_decision\included"
    output_folder = r"data\test_data\included_section_extraction"

    batch_clean_papers(prompt_yaml, input_folder, output_folder)
