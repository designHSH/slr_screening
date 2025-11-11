import fitz  # PyMuPDF
from pathlib import Path
import json

PDF_DIR = Path(r"data\test_data\files_pdf_to_txt\input")
TEXT_DIR = Path(r"data\test_data\files_pdf_to_txt\output")
LOG_PATH = Path(r"data\test_data\files_pdf_to_txt\log\logextraction_log.jsonl")

TEXT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

def extract_text_from_pdf(pdf_path: Path):
    """Extract plain text from PDF using PyMuPDF."""
    text = ""
    try:
        with fitz.open(pdf_path) as doc:
            for page in doc:
                text += page.get_text("text") + "\n"
        return text.strip()
    except Exception as e:
        return f"[ERROR: {e}]"

def main():
    results = []
    pdf_files = list(PDF_DIR.glob("*.pdf"))
    print(f"📄 Found {len(pdf_files)} PDF files.")

    for pdf in pdf_files:
        txt_path = TEXT_DIR / f"{pdf.stem}.txt"
        if txt_path.exists():
            continue

        text = extract_text_from_pdf(pdf)
        if text.startswith("[ERROR"):
            print(f"❌ Failed: {pdf.name}")
        else:
            print(f"✅ Extracted: {pdf.name}")
            txt_path.write_text(text, encoding="utf-8")

        results.append({"file": pdf.name, "length": len(text), "status": "ok" if not text.startswith("[ERROR") else "error"})

    LOG_PATH.write_text("\n".join(json.dumps(r) for r in results), encoding="utf-8")
    print(f"✅ Extraction complete. Log saved to {LOG_PATH}")

if __name__ == "__main__":
    main()
