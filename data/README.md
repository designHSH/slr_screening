# 📂 data/

This folder stores all **paper-related data** used and generated throughout the full-text screening process.

Data is organized into three stages of transformation:

data/
├── raw/ # Original Rayyan export (CSV)
├── interim/ # Cleaned input, LLM outputs, human reviews
└── processed/ # Final adjudicated screening results


---

## 📁 Folder Structure

### 🟩 `/raw/`

- Source data exported from Rayyan or reference manager (e.g., `screening_input.csv`)
- Contains titles, authors, year, DOI, abstract, and other metadata
- **Do not modify manually** — treat as read-only reference

### 🟨 `/interim/`

Intermediate files used during screening:

- `clean_input.parquet` — Cleaned and preprocessed input from raw CSV
- `llm_screening.parquet` — LLM-generated decisions and metadata
- `human_reviews.parquet` — Human annotations for validation/adjudication

These files are updated frequently and should be versioned carefully.

### 🟦 `/processed/`

Final, stable outputs after merging LLM and human decisions:

- `final_screening.csv` — Contains inclusion/exclusion decisions, labels, notes
- Used for downstream analysis and publication

---

## 🧠 Naming Convention

Files are named clearly by content and transformation stage. You can add version numbers or timestamps if needed (e.g., `final_screening_v2.csv`, `clean_input_2025-09-17.parquet`).

---

## 🔐 Sensitive Data

This folder is expected to contain **non-sensitive, published academic metadata**. However:

- Do not include PDFs or full-text content here
- Store those in a separate `/pdf/` or project folder, outside of Git

---

## 🔁 Reproducibility Tip

Every screening run should be linked to the exact input data version used. Consider:

- Copying the input CSV to a `/reports/configs_used/` folder
- Logging hashes of the `.parquet` files in run reports

---

## 🛑 Git Ignore Notice

The following folders are excluded from version control (see `.gitignore`):

- `data/interim/`
- `data/processed/`

This avoids committing large or frequently changing files. Use manual versioning or tagging for critical files.

---

## 📎 Linked Scripts

These files are read or written by:

- `src/slr_screening/io_utils.py`
- `src/slr_screening/screen_llm.py`
- `src/slr_screening/adjudication.py`

---

