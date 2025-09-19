# src/prompt_runner.py
import yaml
from dotenv import load_dotenv  # Import dotenv
from pathlib import Path
from openai import OpenAI
import json

load_dotenv()

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
        system_msg = {"role": "system", "content": self.prompt_cfg["system_command"]}
        user_msg = {
            "role": "user",
            "content": self.prompt_cfg["user_command"].format(**kwargs),
        }
        return [system_msg, user_msg]

    def run(self, **kwargs):
        """Run the prompt with given paper content and return JSON result."""
        messages = self.build_messages(**kwargs)
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0,
            top_p=1,
            max_tokens=1000,
        )
        return response.choices[0].message.content.strip()


if __name__ == "__main__":
    # Example usage for quick test
    runner = PromptRunner("prompt/full_text_screening_pr.yaml")

    paper_data = {
      "title": "Context-Sensitive Ecological Momentary Assessment: Application of User-Centered Design for Improving User Satisfaction and Engagement During Self-Report",
      "authors": "Preethi Srinivas et al.",
      "year": "2019",
      "text": "Methods: An iterative, user-centered design process with obese, middle-aged women seeking care in a safety-net health system was used to identify the preferred format of self-report measures and the look, feel, and interaction of the mobile EMA tool. A single-arm feasibility field trial with 21 participants receiving 12 prompts each day for momentary self-reports over a 4-week period (336 total prompts per participant) was used to determine user satisfaction with interface quality and user engagement, operationalized as response rate. A second trial among 38 different participants randomized to receive or not to receive a feature designed to improve engagement was conducted.  Results: The feasibility trial results showed high interface satisfaction and engagement, with an average response rate of 50% over 4 weeks. Qualitative feedback pointed to the need for auditory alerts. We settled on 3 alerts at 10-min intervals to accompany each EMA-reporting prompt. The second trial testing this feature showed a statistically significant increase in the response rate between participants randomized to receive repeat auditory alerts versus those who were not (60% vs 40%).  Conclusions: This paper reviews the design research and a set of design constraints that may be considered in the creation of mobile EMA interfaces personalized to users’ preferences. Novel aspects of the study include the involvement of low health-literacy adults in design research, the capture of data on time, place, and social context of eating and sedentary behavior, and reporting prompts tailored to an individual’s location and schedule."
    }

    result = runner.run(**paper_data)

    try:
        parsed = json.loads(result)
        print("✅ Parsed JSON:", parsed)
    except json.JSONDecodeError:
        print("⚠️ Raw output (not valid JSON):", result)
