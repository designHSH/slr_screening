# 📂 prompts/

This folder contains all **prompt templates and schemas** used in the full-text screening process powered by Large Language Models (LLMs), such as GPT or Gemini.

Prompts stored here are carefully engineered to:
- Automate inclusion/exclusion screening decisions
- Incorporate domain knowledge and criteria from `/configs/`
- Ensure reliable, explainable, and reproducible model behavior

---

## 📄 Typical Contents

### `slr_screening.yaml`

This file defines the **main prompt configuration** used for LLM-based screening. It typically includes:

- `user_command`: Instructions to the model, referencing a paper's metadata and full text
- `system_command`: System-level context (role, tone, constraints)
- `output_format`: Expected response structure (usually JSON schema)
- Optional `examples`: Few-shot examples to guide the LLM's behavior

> ✅ The YAML format makes the prompt **readable**, **editable**, and **modular** for testing and reuse.

---

## 🧠 Purpose in the Pipeline

Prompts are dynamically loaded and built by the screening engine (`prompt_builder.py`) to:

- Convert each paper’s content into a structured LLM input
- Include screening criteria from `/configs/screening.yml`
- Request decisions in structured format for automation and parsing

---

## 🧪 Prompt Engineering Best Practices

- Prompts should be:
  - Clear and specific
  - Structured for reproducible output
  - Aligned with JSON schema (for automatic validation)
- Use `{input_text}` or `{paper}` as placeholders for dynamic substitution
- Keep `system_command` minimal but directive (e.g., “You are a screening assistant for systematic reviews…”)

---

## 📌 Versioning

Prompt behavior affects model output — even small changes in phrasing can change decisions. To ensure reproducibility:

- Track prompt changes using Git
- Save the prompt version (or copy) used in each run under `reports/configs_used/`
- Link the prompt file version in your `run_YYYY-MM-DD.md` reports

---

## 📎 Linked Modules

Prompts in this folder are consumed by:

- `prompt_builder.py` — builds the final prompt with paper data
- `llm_client.py` — sends the prompt to OpenAI/Gemini
- `screen_llm.py` — processes and interprets LLM responses

---

## ✅ Example Structure

```yaml
user_command: |
  You are reviewing a scientific article to determine whether it should be included in a systematic literature review.

  TASK:
  Based on the following abstract and metadata, decide if the article meets the inclusion criteria:
  {input_text}

system_command: |
  You are an expert assistant for systematic literature reviews in the field of design studies. Return only structured JSON.

output_format: |
  {
    "decision": "include" | "exclude" | "maybe",
    "reason": "...",
    "confidence": 0.0–1.0
  }
