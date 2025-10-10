# src/prompt_runner_pdf.py
import yaml
from dotenv import load_dotenv
from pathlib import Path
from openai import OpenAI
import json

load_dotenv()

class PromptRunnerPDF:
    """
    A Prompt Runner that uploads a PDF file to GPT (vision-capable model)
    and runs a structured prompt with it.
    """
    def __init__(self, prompt_path: str, model: str = "gpt-4o"):
        self.prompt_cfg = self._load_prompt(prompt_path)
        self.model = model
        self.client = OpenAI()  # Reads OPENAI_API_KEY from .env

    def _load_prompt(self, path: str):
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def upload_pdf(self, pdf_path: str):
        """Uploads a PDF to OpenAI's temporary file storage and returns file_id."""
        pdf_file = self.client.files.create(
            file=open(pdf_path, "rb"),
            purpose="assistants"
        )
        return pdf_file.id

    def build_messages(self, pdf_file_id: str, **kwargs):
        """
        Builds messages that combine text instructions with an attached PDF file.
        Your YAML should have 'system_command' and 'user_command' placeholders.
        """
        system_msg = {"role": "system", "content": self.prompt_cfg["system_command"]}

        user_msg = {
            "role": "user",
            "content": [
                {"type": "text", "text": self.prompt_cfg["user_command"].format(**kwargs)},
                {"type": "file", "file_id": pdf_file_id}
            ]
        }
        return [system_msg, user_msg]

    def run(self, pdf_path: str, **kwargs):
        """
        Uploads the PDF, runs the prompt, and returns structured output.
        """
        pdf_id = self.upload_pdf(pdf_path)
        messages = self.build_messages(pdf_id, **kwargs)

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=1500,
        )

        result = response.choices[0].message.content.strip()

        # Optional cleanup: delete uploaded PDF (temporary storage)
        self.client.files.delete(pdf_id)

        # Try to parse JSON
        try:
            return json.loads(result)
        except json.JSONDecodeError:
            return {"raw_output": result}


if __name__ == "__main__":
    # Example usage
    runner = PromptRunnerPDF("prompt\full_text_screening_pr_pdf.yaml")

    result = runner.run(
        pdf_path="data\test_data\Coffen-Burke_et_al- 2023.pdf",
        
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))
