# Instruction Document  
## Quote ID Generation & JSON Schema  
### (Step 2 – Barrier Candidate Extraction)
revision: 0.1.0
---

## 1. Purpose of This Document

This document defines the **standardized rules** for:

- Generating **Quote IDs** for extracted barrier candidates  
- Computing a **quote anchor** for auditability and deduplication  
- Producing a **consistent JSON schema** for Step 2 outputs  

These rules ensure:

- Full traceability from quote → paper  
- Stability across pipeline reruns  
- Compatibility with downstream coding, merging, and dependency modeling  

This specification applies **only to Step 2 (Extraction)**.  
No coding, labeling, or interpretation is permitted at this stage.

---

## 2. Scope (Step 2 Only)

Step 2 performs **barrier candidate extraction** by identifying verbatim text excerpts (meaning units) that explicitly describe barriers to applying Human-Centered Design (HCD).

At this step:

- Quotes are **identified and recorded**
- No categorization or abstraction is applied
- Each quote receives a deterministic identifier

---

## 3. Quote ID Design

### 3.1 Quote ID Format

Each extracted quote MUST be assigned a human-readable `quote_id` using the following format:

```
{first_author_lastname}-{publication_year}-{section_code}-{counter}
```

### Example

```
abu-haydar-2021-D-007
smith-2019-M-002
```

---

### 3.2 Components

#### a) `first_author_lastname`

- Use the **first author’s last name only**
- Lowercase
- Replace spaces with hyphens
- Remove punctuation

**Examples:**
- `Abu-Haydar` → `abu-haydar`
- `van der Waals` → `van-der-waals`

---

#### b) `publication_year`

- Four-digit year of publication
- Must be derived from reliable metadata

---

#### c) `section_code`

Each quote must be associated with **exactly one** section code:

| Code | Section |
|----|--------|
| M | Methods |
| R | Results |
| D | Discussion |
| C | Conclusion |
| L | Limitations |
| O | Other (eligible content not fitting above categories) |

**Rules for `O` (Other):**

- Use only if the text is within the eligible scope
- Examples: *Findings*, *Evaluation*, *Case Study Reflections*
- Do not use `O` for Introduction, Abstract, or References

---

#### d) `counter` (Per Paper)

- Numeric counter starting at `001`
- Increments **per paper**, not per section
- Assigned in extraction order across eligible sections
- Zero-padded to three digits (`001`, `002`, …)

---

## 4. Quote Anchor (Deterministic Fingerprint)

### 4.1 Purpose of `quote_anchor`

The `quote_anchor` provides a **stable fingerprint** of the quote text to support:

- Duplicate detection
- Auditability across reruns
- Traceability when counters shift
- Validation during merging and normalization

The anchor is **not** meant to be human-readable.

---

### 4.2 Anchor Generation Rules

1. Normalize the quote text:
   - Convert to lowercase
   - Trim leading and trailing whitespace
   - Collapse multiple spaces/newlines into a single space

2. Compute a hash:
   - Apply `sha1` to the normalized text
   - Extract the **first 6 hexadecimal characters**

**Example:**

```
quote_anchor = "8f3a1c"
```

The anchor must be deterministic: the same quote text must always produce the same anchor.

---

## 5. Paper Identifier (`paper_id`)

Each quote record MUST include a stable `paper_id`.

### Requirements

- Must uniquely identify the paper
- Must remain stable across pipeline runs

**Examples:**
- Internal `doc_id`
- Controlled filename slug
- Database primary key

The `paper_id` is the **true unique identifier**.  
The `quote_id` is human-readable and paper-scoped.

---

## 6. JSON Output Schema (Step 2 – Final)

Each extracted quote must be represented as one JSON object.  
The output must be a **JSON array**.

### Required Fields

```json
[
  {
    "paper_id": "string (stable unique paper identifier)",
    "quote_id": "string (author-year-section-counter)",
    "quote_anchor": "string (deterministic hash fingerprint)",
    "section": "Methods | Results | Discussion | Conclusion | Limitations | Other",
    "section_code": "M | R | D | C | L | O",
    "quote": "string (verbatim text excerpt)"
  }
]
```

---

## 7. Field Rules & Constraints

- `quote` must be **verbatim** (no paraphrasing)
- `quote_id` must be unique **within the paper**
- `quote_anchor` must be derived **only** from the quote text
- `section` and `section_code` must correspond correctly
- If no qualifying barriers are found, return an **empty array**: `[]`
- No additional fields may be added at Step 2

---

## 8. Handling Later Modifications (Forward Compatibility)

### Step 3 – Splitting Quotes

If a quote is split later:

- Preserve the original `quote_id` as a parent
- Append suffix letters for children:
  - `abu-haydar-2021-D-007a`
  - `abu-haydar-2021-D-007b`
- Recompute `quote_anchor` for each new quote
- Store `parent_quote_id` in Step 3 (not Step 2)

---

## 9. What This Step Explicitly Does NOT Do

At Step 2, the system must NOT:

- Assign categories
- Create barrier labels
- Merge quotes
- Interpret meaning
- Infer causes or consequences

Step 2 is **identification only**, not analysis.

---

## 10. Methodological Rationale

This design:

- Separates data identification from interpretation
- Ensures quote-level traceability
- Supports LLM-assisted qualitative rigor
- Enables reproducibility and auditing
- Prepares clean inputs for coding, merging, and dependency modeling
