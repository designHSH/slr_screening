# 📂 configs/

This folder contains all configuration files that define the **screening logic, inclusion/exclusion criteria**, and **domain-specific knowledge** used by the screening pipeline.

These configurations guide both the **LLM-based triage process** and **human validation workflows**, ensuring consistent, explainable, and reproducible screening decisions.

---

## 📌 Contents

Typical files stored in this folder:

- `incl_excl_criteria.yml`  
  Defines the **inclusion and exclusion criteria**, screening categories, thresholds, and possible label mappings (e.g., `include`, `exclude`, `maybe`).

- `domain_knowledge.yml` *(optional)*  
  Contains **structured knowledge**, such as key definitions, keywords, or concept descriptions to be embedded into LLM prompts or used for rule-based pre-filtering.

- `label_schema.yml` *(optional)*  
  Specifies a **controlled vocabulary** or schema for classification outputs (e.g., reasons for exclusion, relevance dimensions, or TASKS framework labels).

---

## 🧠 Role in the Pipeline

The configuration files in this folder are:

- **Loaded by the screening engine**  to dynamically build prompts and control behavior.
- Used to **embed domain-specific logic** into the prompt structure.
- Referenced during **thresholding** and **post-processing** .
- Serve as **single source of truth** for all decision criteria, enabling transparent updates without modifying the code.

---

## 🔁 Versioning and Reproducibility

Any change to these files **may affect screening results**. Please:

- Track changes using Git commits.
- Reference the config version used in your `reports/run_YYYY-MM-DD.md` files.
- Store copies of used configs alongside each run for full reproducibility.

---

## ✅ Best Practices

- Use **clear YAML formatting** and comments to document each parameter.
- Keep criteria concise, specific, and operationalizable.
- Use separate files if configurations grow large (e.g., split criteria from knowledge base).

---

## 🔒 Example

```yaml
# screening.yml

criteria:
  include_if:
    - mentions human-centered design AND implementation context
    - discusses challenges, barriers, or enablers

  exclude_if:
    - not peer-reviewed
    - unrelated domain (e.g., architecture, art)
    - discusses only user testing without design phase

thresholds:
  gpt_confidence_include: 0.75
  gpt_confidence_exclude: 0.75
