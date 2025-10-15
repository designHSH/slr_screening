# src/clean_texts_regex.py
import os
import re
from pathlib import Path

# === Paths ===
RAW_DIR = Path("data/test_data/section_extraction/input")
CLEAN_DIR = Path("data/test_data/section_extraction/output")

# Create output folder if it doesn't exist
CLEAN_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(text: str) -> str:
    """
    Removes back-matter sections such as references, acknowledgments, funding, etc.
    Keeps all content before them intact.
    """
    patterns = [
        r"(?i)\bReferences\b.*",          # References
        r"(?i)\bBibliography\b.*",        # Bibliography
        r"(?i)\bAcknowledg(e)?ments?\b.*",# Acknowledgements / Acknowledgments
        r"(?i)\bFunding\b.*",             # Funding
        r"(?i)\bAppendix\b.*",            # Appendix
        r"(?i)\bAppendices\b.*",          # Appendices
        r"(?i)\bSupplementary Material\b.*" # Supplementary materials
    ]
    for pat in patterns:
        text = re.sub(pat, "", text, flags=re.DOTALL)
    return text.strip()


def process_all_texts():
    print(f"🚀 Cleaning all .txt files in: {RAW_DIR}")
    files = list(RAW_DIR.glob("*.txt"))

    if not files:
        print("⚠️ No text files found in input directory.")
        return

    for file_path in files:
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned_text = clean_text(raw_text)

            output_path = CLEAN_DIR / file_path.name
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(cleaned_text)

            print(f"✅ Cleaned: {file_path.name}")
        except Exception as e:
            print(f"❌ Error processing {file_path.name}: {e}")

    print(f"\n🎯 Cleaning complete! Results saved to: {CLEAN_DIR}")


if __name__ == "__main__":
    process_all_texts()
