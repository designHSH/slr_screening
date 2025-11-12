import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from openai import OpenAI, APIError
from dotenv import load_dotenv  # <-- Added this import

class BarrierExtractor:
    """
    Processes text files to extract HCD barriers using an LLM.
    
    Reads prompt configuration from a YAML file, processes each .txt file
    in an input directory, and saves the LLM's JSON output to an
    output directory.
    """
    def __init__(self, prompt_yaml_path: Path, input_dir: Path, output_dir: Path):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.prompt_yaml_path = Path(prompt_yaml_path)

        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.log_file = self.output_dir / 'processing_log.txt'
        self._setup_logging()

        # Load prompt and config
        try:
            self._load_prompt_config()
            logging.info(f"Successfully loaded prompt YAML: {self.prompt_yaml_path.name}")
        except Exception as e:
            logging.error(f"Failed to load prompt YAML from {self.prompt_yaml_path}: {e}")
            raise

        # Setup API client
        try:
            # The OpenAI() client, by default, reads the OPENAI_API_KEY
            # from the environment, which load_dotenv() populated.
            self.client = OpenAI()
            
            # Explicitly check if the key was actually loaded.
            if self.client.api_key is None:
                raise ValueError("OPENAI_API_KEY not found. Ensure it is set in your .env file.")
                
            logging.info("OpenAI client initialized.")
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client. Is OPENAI_API_KEY set in your .env file? Error: {e}")
            raise

        # Log the initial setup details
        logging.info("--- Processing Run Started ---")
        logging.info(f"Model: {self.model_name}")
        logging.info(f"Prompt Version: {self.prompt_version} (Task: {self.task_name})")
        logging.info(f"Input Directory: {self.input_dir}")
        logging.info(f"Output Directory: {self.output_dir}")

    def _setup_logging(self):
        """Configures logging to both file and console."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                # Append to the log file in the output directory
                logging.FileHandler(self.log_file, mode='a', encoding='utf-8'),
                # Also print logs to the console
                logging.StreamHandler()
            ]
        )

    def _load_prompt_config(self):
        """Loads task, model, and prompt templates from the YAML file."""
        with open(self.prompt_yaml_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        self.prompt_data = data
        self.task_name = data.get('task', 'unknown_task')
        self.prompt_version = str(data.get('version', 'unknown_version'))
        
        config = data.get('config', {})
        # Use the model from YAML, with 'gpt-5-mini' as a fallback
        self.model_name = config.get('model', 'gpt-5-mini')
        self.top_p = config.get('top_p', 1.0)
        
        self.system_prompt = data.get('system_command', '')
        self.user_prompt_template = data.get('user_command', '')

        if not self.system_prompt or not self.user_prompt_template:
            logging.error("YAML file is missing 'system_command' or 'user_command'.")
            raise ValueError("Invalid prompt YAML: system_command or user_command missing.")

    def _call_model(self, text_content: str) -> str | None:
        """Calls the OpenAI API with the formatted prompt and text."""
        
        # Format the user message with the input text
        user_message_content = self.user_prompt_template.format(input_text=text_content)
        
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_message_content}
        ]

        try:
            logging.info(f"Sending request to model ({self.model_name})...")
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                top_p=self.top_p,
                # This enforces the model to output a valid JSON object
                response_format={"type": "json_object"} 
            )
            
            response_content = completion.choices[0].message.content
            logging.info("Received valid JSON response from model.")
            return response_content

        except APIError as e:
            logging.error(f"OpenAI API error: {e}")
            # Note: gpt-5-mini may not be a valid model name.
            if "does not exist" in str(e):
                logging.error(f"Model '{self.model_name}' may not exist. Check your YAML config.")
        except Exception as e:
            logging.error(f"An unexpected error occurred during API call: {e}")
        
        return None # Return None on failure

    def process_file(self, txt_file_path: Path):
        """Reads a single text file, calls the model, and saves the JSON output."""
        file_name = txt_file_path.name
        output_json_path = self.output_dir / f"{txt_file_path.stem}.json"

        logging.info(f"--- Processing file: {file_name} ---")

        try:
            text_content = txt_file_path.read_text(encoding='utf-8')
            if not text_content.strip():
                logging.warning(f"File is empty, skipping: {file_name}")
                return
        except Exception as e:
            logging.error(f"Failed to read file {txt_file_path}: {e}")
            return # Skip this file

        # Get the JSON response string from the model
        model_response_str = self._call_model(text_content)

        if not model_response_str:
            logging.error(f"No response from model for file: {file_name}. Skipping.")
            return

        # Parse the JSON string and inject the correct filename
        try:
            # The model *should* return valid JSON because of `response_format`
            data = json.loads(model_response_str)
            
            # **Robustness:**
            # Override the filename with the *actual* source filename.
            # This ensures 100% accuracy, even if the model hallucinates
            # or omits the 'file_name' key.
            data['file_name'] = file_name
            
            # Save the validated and corrected JSON
            with open(output_json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Successfully processed and saved: {output_json_path.name}")

        except json.JSONDecodeError:
            logging.error(f"CRITICAL: Failed to decode JSON from model for file: {file_name}")
            logging.error(f"Raw model output: {model_response_str[:500]}...")
        except Exception as e:
            logging.error(f"Failed to save JSON file {output_json_path}: {e}")

    def run_processing(self):
        """Loops through all .txt files in the input directory and processes them."""
        logging.info(f"Starting batch processing from: {self.input_dir}")
        
        text_files = list(self.input_dir.glob('*.txt'))
        if not text_files:
            logging.warning(f"No .txt files found in {self.input_dir}. Nothing to process.")
            return

        logging.info(f"Found {len(text_files)} .txt file(s) to process.")

        for txt_file_path in text_files:
            self.process_file(txt_file_path)

        logging.info("--- Batch processing complete ---")

# --- Main execution ---
if __name__ == "__main__":
    
    # Load environment variables from .env file at the script's start
    load_dotenv() # <-- Added this line
    
    # Define paths relative to this script's location
    SCRIPT_DIR = Path(__file__).parent.resolve()
    PROMPT_YAML_FILE = SCRIPT_DIR / "prompt/barrier_identification_pr.yaml"
    INPUT_FOLDER = SCRIPT_DIR / "data/test_data/barrier_identification_test/input_sample"
    OUTPUT_FOLDER = SCRIPT_DIR / "data/test_data/barrier_identification_test/output/1st_run"

    # --- Setup Directories and Files ---
    INPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    PROMPT_YAML_FILE.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if prompt file exists before starting
        if not PROMPT_YAML_FILE.exists():
            print(f"Error: {PROMPT_YAML_FILE.name} not found.")
            print(f"Please create {PROMPT_YAML_FILE.name} in the script directory.")
            print("You can copy the content from the documentation.")
            exit(1) # Exit with an error code
            
        # Create and run the extractor
        extractor = BarrierExtractor(
            pprompt_yaml_path=PROMPT_YAML_FILE,
            input_dir=INPUT_FOLDER,
            output_dir=OUTPUT_FOLDER
        )
        extractor.run_processing()

    except Exception as e:
        # This will catch setup errors (e.g., bad YAML path or missing API key)
        logging.error(f"A critical error occurred: {e}", exc_info=False)
        # Set exc_info=False to avoid a full stack trace for common errors
        # like the missing API key, which is handled gracefully.