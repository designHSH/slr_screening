# src/prompt_runner_pdf.py
import yaml
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import json

load_dotenv()  # Loads OPENAI_API_KEY from .env


class PromptRunnerPDF:
    """
    A Prompt Runner that uploads a PDF file to GPT (vision-capable model)
    and runs a structured prompt with it.
    """

    def __init__(self, prompt_path: str, model: str = "gpt-4o"):
        self.prompt_cfg = self._load_prompt(prompt_path)
        self.model = model
        self.client = OpenAI()  # Reads OPENAI_API_KEY from environment

    def _load_prompt(self, path: str):
        """Load YAML prompt configuration."""
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def upload_pdf(self, pdf_path: str) -> str:
        """Uploads a PDF to OpenAI's temporary file storage and returns its file ID."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"❌ PDF file not found: {pdf_path}")

        with open(pdf_path, "rb") as f:
            uploaded = self.client.files.create(file=f, purpose="assistants")

        print(f"✅ Uploaded PDF: {uploaded.id}")
        return uploaded.id

    def build_messages(self, pdf_file_id: str):
        """Builds chat messages combining YAML text and attached PDF file."""
        system_msg = {
            "role": "system",
            "content": self.prompt_cfg["system_command"],
        }

        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt_cfg["user_command"]},
                {"type": "file", "file": {"file_id": pdf_file_id}},
            ],
        }

        return [system_msg, user_msg]

    def run(self, pdf_path: str):
        """
        Uploads the PDF, runs the prompt, and returns structured JSON output.
        """
        pdf_id = self.upload_pdf(pdf_path)
        messages = self.build_messages(pdf_id)

        print("🚀 Sending request to GPT-4o...")
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=2000,
        )

        result = response.choices[0].message.content.strip()

        # Clean up uploaded file
        try:
            self.client.files.delete(pdf_id)
            print(f"🗑️ Deleted temporary file: {pdf_id}")
        except Exception as e:
            print(f"⚠️ Warning: could not delete file {pdf_id} ({e})")

        # Try parsing JSON output
        try:
            parsed = json.loads(result)
            print("✅ Parsed JSON successfully.")
            return parsed
        except json.JSONDecodeError:
            print("⚠️ Output was not valid JSON, returning raw text.")
            return {"raw_output": result}


if __name__ == "__main__":
    # Example usage
    pdf_file_path = Path(r"data/test_data/Tetiranont_et_al_2024.pdf")
    prompt_file_path = Path(r"prompt/full_text_screening_pr_pdf.yaml")

    runner = PromptRunnerPDF(prompt_file_path)

    result = runner.run(pdf_path=pdf_file_path)

    print(json.dumps(result, indent=2, ensure_ascii=False))
