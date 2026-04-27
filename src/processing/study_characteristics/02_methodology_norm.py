from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd


# ---------------------------------------------------------
# Short description
# ---------------------------------------------------------
# This script processes the raw design_methodology_value field
# and creates rule-based coding columns for methodology analysis.
#
# It asks the user for:
# 1) input CSV file path
# 2) output directory
#
# Then it:
# - keeps original columns unchanged
# - reads design_methodology_value
# - detects core HCD/UCD methodology
# - detects integrated methodologies
# - detects process descriptors
# - decides whether the methodology is hybrid
# - flags ambiguous cases for manual review
# - saves the coded result as a new CSV file
# ---------------------------------------------------------


# =========================================================
# Utility functions
# =========================================================
def normalize_text(text: str) -> str:
    """Normalize text for rule matching."""
    text = text.lower()
    text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    text = text.replace("–", "-").replace("—", "-").replace("â€“", "-")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def contains_any(text: str, phrases: List[str]) -> bool:
    """Return True if any phrase is found in text."""
    return any(phrase in text for phrase in phrases)


def find_all_matches(text: str, rules: Dict[str, List[str]]) -> Set[str]:
    """Return all rule labels whose phrases appear in text."""
    matches = set()
    for label, phrases in rules.items():
        if contains_any(text, phrases):
            matches.add(label)
    return matches


# =========================================================
# Rule dictionaries
# =========================================================

CORE_UCD_PHRASES = [
    "user centered design",
    "user-centered design",
    "user centred design",
    "user-centred design",
    "user centered approach",
    "user-centered approach",
    "user centred approach",
    "user-centred approach",
    "user centered design approach",
    "user-centered design approach",
    "user centred design approach",
    "user-centred design approach",
    "user centered design process",
    "user-centered design process",
    "user centred design process",
    "user-centred design process",
    "user-centered software design approach",
    "user-centred software design approach",
    "systematic, user-centered design approach",
    "systematic user-centered design approach",
    "iterative user-centered approach",
    "iterative user-centred approach",
    "iterative user-centered design",
    "iterative user-centred design",
    "user-centric design",
    "user-centric design process",
    "(ucd)",
    " ucd ",
]

CORE_HCD_PHRASES = [
    "human centered design",
    "human-centered design",
    "human centred design",
    "human-centred design",
    "human centered approach",
    "human-centered approach",
    "human centred approach",
    "human-centred approach",
    "(hcd)",
    " hcd ",
]

INTEGRATED_RULES = {
    "Participatory design (PD)": [
        "participatory design",
        "(pd)",
        " pd ",
    ],
    "Co-design": [
        "co-design",
        "codesign",
    ],
    "Cooperative design": [
        "cooperative design",
    ],
    "Service design": [
        "service design",
    ],
    "Contextual design": [
        "contextual design",
    ],
    "Scenario-based design (SBD)": [
        "scenario-based design",
        "(sbd)",
        " sbd ",
    ],
    "Agile development": [
        "agile development",
        "agile software development",
        " agile ",
    ],
    "Software engineering methods": [
        "software engineering methods",
        "iterative software development",
        "software development techniques",
    ],
    "Community-based participatory research (CBPR)": [
        "community-based participatory research",
        "(cbpr)",
        " cbpr ",
    ],
    "Community-based research": [
        "community-based research",
    ],
    "Participatory action research (PAR)": [
        "participatory action research",
        "(par)",
        " par ",
    ],
    "Participatory research": [
        "participatory research approach",
        "participatory research",
    ],
    "Participatory ergonomics (PE)": [
        "participatory ergonomics",
        "(pe)",
        " pe ",
    ],
    "Patient and public involvement (PPI)": [
        "patient and public involvement",
        "(ppi)",
        " ppi ",
    ],
    "Intervention Mapping (IM)": [
        "intervention mapping",
        "(im)",
        " im ",
    ],
    "Design science research": [
        "design science research methodology",
        "design science research",
        "(dsrm)",
        " dsrm ",
    ],
    "CeHRes Roadmap": [
        "cehres roadmap",
    ],
    "Rapid Contextual Design (RCD)": [
        "rapid contextual design",
        "(rcd)",
        " rcd ",
    ],
    "Situated Cognitive Engineering (sCE)": [
        "situated cognitive engineering",
        "(sce)",
        " sce ",
    ],
    "HCD+": [
        "hcd+",
        "human centered design for aging",
        "human centred design for aging",
    ],
    "Grounded theory (GT)": [
        "grounded theory",
        "(gt)",
        " gt ",
    ],
    "Evolutionary design": [
        "evolutionary design",
    ],
    "Design thinking": [
        "design thinking",
    ],
    "Human–information interaction (HII)": [
        "human-information interaction",
        "human information interaction",
        "(hii)",
        " hii ",
    ],
    "Person-based approach": [
        "person-based approach",
    ],
    "Theory-based design": [
        "theory-based design",
    ],
    "SDM framework": [
        "sdm framework",
    ],
    "Medical Research Council (MRC) framework": [
        "medical research council",
        "mrc framework",
    ],
    "Cooperative Method Development (CMD)": [
        "cooperative method development",
        "(cmd)",
        " cmd ",
    ],
    "Iterative Development in the Field (IDF)": [
        "iterative development in the field",
        "(idf)",
        " idf ",
    ],
    "InterMod": [
        "intermod",
    ],
    "Human-centered distributed information design (HCDID)": [
        "human-centered distributed information design",
        "(hcdid)",
        " hcdid ",
    ],
    "Hybrid User Centered Development Methodology (HUCDM)": [
        "hybrid user centered development methodology",
        "(hucdm)",
        " hucdm ",
    ],
    "Annotator-centered design": [
        "annotator-centered design",
    ],
}

DESCRIPTOR_RULES = {
    "Iterative": [
        "iterative",
    ],
    "Incremental": [
        "incremental",
    ],
    "Qualitative": [
        "qualitative",
    ],
    "Mixed methods": [
        "mixed methods",
        "mixed-methods",
    ],
    "Theory-driven": [
        "theory-driven",
    ],
    "Systematic": [
        "systematic",
    ],
    "Interdisciplinary": [
        "interdisciplinary",
    ],
    "Interactive": [
        "interactive",
    ],
    "Rapid prototyping": [
        "rapid prototyping",
        "rapid-prototyping",
    ],
    "Divergent/convergent": [
        "divergent and convergent",
        "divergent",
        "convergent",
    ],
    "Engineering-oriented": [
        "engineering design approach",
        "engineering design",
    ],
    "Participatory": [
        "participatory",
    ],
}

MANUAL_REVIEW_ONLY_PHRASES = [
    "participatory design",
    "co-design",
    "mixed methods",
    "contextual design",
    "rapid contextual design",
    "cehres roadmap",
    "intermod",
    "design study methodology",
    "design science research methodology",
    "intervention mapping",
    "situated cognitive engineering",
    "participatory research approach",
    "annotator-centered design",
    "human-centered distributed information design",
    "hcd+",
]


# =========================================================
# Classification logic
# =========================================================
def detect_core_methodology(text: str) -> Tuple[str, List[str]]:
    """
    Detect core HCD/UCD methodology.
    Returns (core_label, notes)
    """
    notes = []
    has_ucd = contains_any(text, CORE_UCD_PHRASES)
    has_hcd = contains_any(text, CORE_HCD_PHRASES)

    if has_ucd and not has_hcd:
        return "User-centered design (UCD)", notes
    if has_hcd and not has_ucd:
        return "Human-centered design (HCD)", notes
    if has_ucd and has_hcd:
        notes.append("Both HCD and UCD terms detected")
        return "HCD/UCD not distinguishable", notes

    notes.append("No explicit HCD/UCD core detected")
    return "", notes


def detect_integrated_methods(text: str) -> Set[str]:
    """Detect integrated methodologies."""
    return find_all_matches(text, INTEGRATED_RULES)


def detect_descriptors(text: str) -> Set[str]:
    """Detect process descriptors."""
    return find_all_matches(text, DESCRIPTOR_RULES)


def needs_manual_review(
    text: str,
    core: str,
    integrated: Set[str],
    descriptor: Set[str],
) -> Tuple[bool, List[str]]:
    """
    Decide if manual review is needed.
    """
    reasons = []

    if not core:
        reasons.append("No explicit HCD/UCD core detected")
        return True, reasons

    if core == "HCD/UCD not distinguishable":
        reasons.append("Core is ambiguous between HCD and UCD")
        return True, reasons

    # Ambiguous participatory phrasing without explicit distinct method structure
    if "participatory" in text and "participatory design" not in text:
        reasons.append("Participatory wording may be descriptive rather than a distinct methodology")

    # Manual-review-only phrases if they appear in difficult structure
    if contains_any(text, MANUAL_REVIEW_ONLY_PHRASES) and not integrated:
        reasons.append("Contains named methodology that may require inclusion check")

    # If methodology is only descriptors beyond core, that's okay, not necessarily review
    # But if unusual or very vague approach wording occurs:
    vague_patterns = [
        "approach",
        "method",
        "framework",
    ]
    if not integrated and descriptor and any(v in text for v in vague_patterns):
        # Mild caution only, not enough alone to trigger mandatory review
        pass

    return (len(reasons) > 0), reasons


def remove_false_integrated_matches(core: str, integrated: Set[str]) -> Set[str]:
    """
    Remove methodology labels from integrated if they duplicate the core.
    """
    cleaned = set(integrated)

    # Core UCD/HCD are not part of integrated anyway in our rules.
    # This function is mostly defensive.
    return cleaned


def finalize_multilabel(values: Set[str]) -> str:
    """Return semicolon-separated sorted string or 'None'."""
    if not values:
        return "None"
    return "; ".join(sorted(values))


def classify_methodology(raw_value: object) -> dict:
    """Classify one design_methodology_value cell."""
    if pd.isna(raw_value) or str(raw_value).strip() == "":
        return {
            "core_design_methodology_value": "",
            "integrated_design_methodology_value": "None",
            "hybrid_design_methodology_flag": "",
            "design_process_descriptor_value": "None",
            "manual_review_needed": "Yes",
            "coding_note": "Missing methodology text",
        }

    original_text = str(raw_value).strip()
    text = normalize_text(original_text)

    notes: List[str] = []

    core, core_notes = detect_core_methodology(text)
    notes.extend(core_notes)

    integrated = detect_integrated_methods(text)
    integrated = remove_false_integrated_matches(core, integrated)

    descriptors = detect_descriptors(text)

    # Special participatory ambiguity rule:
    # If participatory appears only as adjectival wording and no explicit
    # participatory design phrase is present, keep it as descriptor only.
    if "participatory" in text and "participatory design" not in text:
        descriptors.add("Participatory")

    # If participatory design is explicitly present together with core,
    # keep it as integrated methodology.
    # If only descriptor words are present, integrated remains empty.

    # Hybrid rule
    if core:
        hybrid = "Yes" if len(integrated) > 0 else "No"
    else:
        hybrid = ""

    manual_review, review_reasons = needs_manual_review(text, core, integrated, descriptors)
    notes.extend(review_reasons)

    if not integrated:
        integrated_str = "None"
    else:
        integrated_str = finalize_multilabel(integrated)

    # Remove descriptor "Participatory" if explicit Participatory design exists
    if "Participatory design (PD)" in integrated and "Participatory" in descriptors:
        descriptors.remove("Participatory")

    descriptor_str = finalize_multilabel(descriptors)

    if not notes:
        if hybrid == "Yes":
            notes.append("Clear hybrid case")
        elif hybrid == "No":
            notes.append("Core-only or descriptor-only case")

    return {
        "core_design_methodology_value": core,
        "integrated_design_methodology_value": integrated_str,
        "hybrid_design_methodology_flag": hybrid,
        "design_process_descriptor_value": descriptor_str,
        "manual_review_needed": "Yes" if manual_review else "No",
        "coding_note": "; ".join(dict.fromkeys(notes)),  # preserve order, remove duplicates
    }


# =========================================================
# File handling
# =========================================================
def validate_required_columns(df: pd.DataFrame, required_columns: List[str]) -> None:
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s): {', '.join(missing)}")


def build_output_path(input_file: Path, output_dir: Path) -> Path:
    stem = input_file.stem
    suffix = input_file.suffix if input_file.suffix else ".csv"
    return output_dir / f"{stem}_coded{suffix}"


# =========================================================
# Main
# =========================================================
def main() -> None:
    print("Methodology coding script")
    print("-" * 40)

    input_path_str = input("Enter the full path to the input CSV file: ").strip().strip('"')
    output_dir_str = input("Enter the full path to the output directory: ").strip().strip('"')

    input_path = Path(input_path_str)
    output_dir = Path(output_dir_str)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    if input_path.suffix.lower() != ".csv":
        raise ValueError("Input file must be a CSV file.")

    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nReading CSV...")
    df = pd.read_csv(input_path)

    required_columns = ["design_methodology_value"]
    validate_required_columns(df, required_columns)

    print("Processing methodology coding...")

    coded_rows = df["design_methodology_value"].apply(classify_methodology)
    coded_df = pd.DataFrame(list(coded_rows))

    # Add coded columns to original dataframe
    result_df = pd.concat([df.copy(), coded_df], axis=1)

    output_path = build_output_path(input_path, output_dir)

    print("Saving coded CSV...")
    result_df.to_csv(output_path, index=False, encoding="utf-8-sig")

    print("\nDone.")
    print(f"Output file saved to:\n{output_path}")

    # Simple summary
    print("\nQuick summary:")
    if "hybrid_design_methodology_flag" in result_df.columns:
        print(result_df["hybrid_design_methodology_flag"].value_counts(dropna=False))
    if "manual_review_needed" in result_df.columns:
        print("\nManual review:")
        print(result_df["manual_review_needed"].value_counts(dropna=False))


if __name__ == "__main__":
    main()