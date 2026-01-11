import os
import yaml
import json
import csv
import time
from pathlib import Path
from dotenv import load_dotenv
from google import genai
from google.genai import types, errors

class GeminiScreeningRunner:
    def __init__(self, prompt_path: str, model: str = "gemini-2.0-flash-001"):
        load_dotenv()
        self.prompt_cfg = self._load_prompt(prompt_path)
        self.model_id = model
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

    def _load_prompt(self, path: str) -> dict:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get_already_processed(self, csv_path: Path) -> set:
        """Reads the CSV to see which files are already finished."""
        if not csv_path.exists():
            return set()
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return {row["file_name"] for row in reader if "file_name" in row}

    def run_with_limit(self, pdf_path: Path) -> dict:
        """Processes a single PDF with automatic retry for rate limits."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                print(f"🚀 Analyzing: {pdf_path.name}...")
                uploaded_file = self.client.files.upload(file=str(pdf_path))
                
                # Standard call (Implicit Caching handles the system_instruction)
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[self.prompt_cfg["user_command"].replace("{{paper_text}}", "See file."), uploaded_file],
                    config=types.GenerateContentConfig(
                        system_instruction=self.prompt_cfg["system_command"],
                        response_mime_type="application/json",
                        temperature=0.0
                    )
                )
                
                # Success! Wait 15 seconds to stay under 5 RPM limit
                print("✅ Done. Cooling down (15s)...")
                time.sleep(15) 
                
                result = {"file_name": pdf_path.name}
                result.update(json.loads(response.text.strip()))
                return result

            except errors.ClientError as e:
                if "429" in str(e):
                    wait_time = 30 * (attempt + 1)
                    print(f"⏳ Rate limit hit. Waiting {wait_time}s before retry...")
                    time.sleep(wait_time)
                else:
                    raise e
        return {"file_name": pdf_path.name, "error": "Max retries exceeded"}

    def append_to_csv(self, result: dict, output_path: Path):
        """Appends a single result to CSV immediately to prevent data loss."""
        file_exists = output_path.exists()
        flat_result = self._flatten_result(result)
        
        with open(output_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(flat_result.keys()))
            if not file_exists:
                writer.writeheader()
            writer.writerow(flat_result)

    def _flatten_result(self, result: dict) -> dict:
        return {k: (json.dumps(v) if isinstance(v, (list, dict)) else v) for k, v in result.items()}

if __name__ == "__main__":
    # CONFIGURATION
    input_folder = Path(r"data\test_data\gemini\input") 
    prompt_file = Path(r"prompt\full_text_screening_pr_v012.yaml")
    output_csv = Path(r"data\test_data\gemini\output\gemini_screening_v012.csv")
    

    runner = GeminiScreeningRunner(prompt_file)
    
    # 1. Check what's already done
    processed_files = runner.get_already_processed(output_csv)
    
    # 2. Get 42 new files
    all_pdfs = list(input_folder.glob("*.pdf"))
    queue = [f for f in all_pdfs if f.name not in processed_files][:42]

    print(f"📋 Starting queue: {len(queue)} papers (Skipping {len(processed_files)} already done)")

    for pdf in queue:
        try:
            res = runner.run_with_limit(pdf)
            runner.append_to_csv(res, output_csv)
        except Exception as e:
            print(f"❌ Fatal error on {pdf.name}: {e}")
            break # Stop the loop if something unexpected happens