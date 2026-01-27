import os
import re
import yaml
import json
import csv
import time
import threading
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List, Optional

# --- 1. RESPONSE SCHEMA ---
class EvidenceSchema(BaseModel):
    HCD_fullcycle: List[str]
    TASKS_elements: List[str]

class ScreeningResult(BaseModel):
    title: str
    authors: str
    decision: str
    confidence: str
    evidence: EvidenceSchema
    reasoning: List[str]
    exclusion_reason: Optional[str] = None

# --- 2. CORE RUNNER CLASS ---
class Gemini3ScreeningRunner:
    def __init__(self, prompt_path: Path, model_id: str, thinking_level: str):
        load_dotenv()
        self.client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
        self.model_id = model_id
        # Supported for Gemini 3 Flash: 'minimal', 'low', 'medium', 'high'
        self.thinking_level = thinking_level.upper() 
        
        with open(prompt_path, "r", encoding="utf-8") as f:
            self.prompt_cfg = yaml.safe_load(f)

    def _call_model_with_timeout(self, uploaded_file, timeout_s: int):
        result = [None]
        error = [None]

        def target():
            try:
                # Map string to SDK Enum
                t_level = getattr(types.ThinkingLevel, self.thinking_level, types.ThinkingLevel.MEDIUM)
                
                config = types.GenerateContentConfig(
                    system_instruction=self.prompt_cfg["system_command"],
                    response_mime_type="application/json",
                    response_schema=ScreeningResult,
                    temperature=0.0,
                    thinking_config=types.ThinkingConfig(
                        include_thoughts=True,
                        thinking_level=t_level
                    )
                )
                
                response = self.client.models.generate_content(
                    model=self.model_id,
                    contents=[self.prompt_cfg["user_command"], uploaded_file],
                    config=config
                )
                result[0] = response
            except Exception as e:
                error[0] = e

        thread = threading.Thread(target=target)
        thread.start()
        thread.join(timeout=timeout_s)

        if thread.is_alive():
            return None, "TIMEOUT"
        if error[0]:
            return None, error[0]
        return result[0], None

    def process_pdf(self, pdf_path: Path, max_retries: int = 3, timeout_s: int = 600) -> dict:
        print(f"\n--- [{datetime.now().strftime('%H:%M:%S')}] Processing: {pdf_path.name} ---")
        
        # Upload Log
        print(f"  📤 Uploading file ({pdf_path.stat().st_size / 1024:.1f} KB)...")
        uploaded_file = self.client.files.upload(file=str(pdf_path))
        print(f"  ✅ Uploaded as: {uploaded_file.name}")

        for attempt in range(max_retries):
            print(f"  🧠 Attempt {attempt+1}/{max_retries} | Level: {self.thinking_level}...")
            start_time = time.time()
            
            response, err = self._call_model_with_timeout(uploaded_file, timeout_s)
            duration = time.time() - start_time

            if err == "TIMEOUT":
                print(f"  ⚠️ Stalled too long ({timeout_s}s). Retrying...")
            elif err:
                print(f"  ❌ API Error: {err}")
                time.sleep(10) # Backoff
            else:
                print(f"  ✨ Success in {duration:.1f}s")
                try:
                    data = json.loads(response.text)
                    data["file_name"] = pdf_path.name
                    data["process_time"] = f"{duration:.1f}s"
                    return data
                except:
                    break
        
        return {"file_name": pdf_path.name, "error": "Failed after retries"}

# --- 3. MAIN EXECUTION BLOCK ---
if __name__ == "__main__":
    # SETTINGS
    TARGET_MODEL = "gemini-3-flash-preview"
    # Choose: 'minimal', 'low', 'medium', 'high'
    THINKING_LEVEL_VAR = "low" 
    
    PROMPT_YAML = Path(r"prompt/full_text_screening_pr_pdf.yaml")
    INPUT_DIR = Path(r"data\test_data\gemini_screening\input")
    OUTPUT_CSV = Path(r"data\test_data\gemini_screening\output_gemini_3\gemini_3_low_screening_v012.csv")

    # Initialize with the variable
    runner = Gemini3ScreeningRunner(PROMPT_YAML, TARGET_MODEL, THINKING_LEVEL_VAR)
    
    # Run loop...
    all_pdfs = list(INPUT_DIR.glob("*.pdf"))
    for pdf in all_pdfs:
        result = runner.process_pdf(pdf)
        # Append to CSV logic here...