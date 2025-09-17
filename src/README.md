# 📂 src/

This folder contains the **source code** for the full-text screening pipeline used in this systematic literature review (SLR) project.

The code is organized under the `slr_screening` package and will include all the core functions required to:

- Read and process paper metadata
- Build and send prompts to Large Language Models (LLMs)
- Interpret and store model responses
- Apply screening rules and criteria
- Prepare outputs for human review and final inclusion decisions

---

## 🧱 Suggested Structure

src/
└── slr_screening/
├── init.py
├── screen_llm.py # Entry point for LLM screening
├── llm_client.py # Wrapper for GPT or Gemini API
├── prompt_builder.py # Builds prompt messages from config and data
├── io_utils.py # File reading and writing utilities
├── cache.py # Local caching of LLM responses
├── thresholding.py # Applies screening decision logic
├── adjudication.py # Merges LLM and human reviews
├── human_pack.py # Prepares Excel files for reviewers
├── eval_metrics.py # Evaluation tools (to be developed)
├── id_utils.py # Utilities for managing paper IDs
└── schema.py # Data models (e.g., Pydantic)


> 📝 *Note: File content and logic are under development and subject to change.*

---

## 🔧 How to Use

Once completed, this codebase will support a semi-automated screening workflow. It will be run using Python scripts or CLI commands, and rely on external configuration files stored in `/configs/`.

---

## 📌 Reminder

If you make structural changes or add new modules, update this file to reflect the latest architecture.

---

