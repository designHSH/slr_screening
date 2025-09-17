# SLR_SCREENING

**Semi-Automated Full-Text Screening Pipeline for Systematic Literature Reviews Using GPT and  PDF Retrieval**

## 📘 Overview

This repository supports a structured, reproducible pipeline for the **full-text screening phase** of a systematic literature review (SLR), integrating:

- ✅ Automated PDF retrieval based on DOIs
- ✅ Structured file and metadata management
- ✅ Prompt-engineered LLM (e.g., GPT) screening
- ✅ Human-in-the-loop validation
- ✅ Modular and extensible Python framework

The pipeline is designed to **maximize efficiency**, **ensure traceability**, and **support rigorous screening criteria**, with clear separation of data, config, and code.

---

## 🔍 Use Case

This project is tailored for researchers conducting SLRs who:
- Have already completed **title/abstract screening** (e.g., via Rayyan)
- Need to collect and organize **420+ full-text PDFs**
- Aim to **triage papers using LLMs** based on inclusion/exclusion criteria
- Want a **repeatable and transparent** screening process

---

## 📁 Repository Structure

```bash
slr_screening/
├─ README.md                  # 🔹 This file
├─ .gitignore
├─ .env.example               # Template for API keys
├─ requirements.txt           # Python dependencies
├─ configs/                   # Screening criteria & thresholds
├─ prompts/                   # LLM prompts and schemas
├─ data/                      # All paper data: input, interim, final
├─ outputs/                   # Generated Excel files for review
├─ cache/                     # JSON logs of LLM runs per paper
├─ logs/                      # Logs from LLM and script execution
├─ reports/                   # Run reports for reproducibility
└─ src/slr_screening/         # Python source code for pipeline
