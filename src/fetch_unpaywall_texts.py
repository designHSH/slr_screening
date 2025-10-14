import os
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime
import csv
import time
from bs4 import BeautifulSoup  # ✅ added for HTML text extraction

# =========================================================
# CONFIGURATION
# =========================================================
INPUT_CSV = Path(r"data\test_data\articles_420_papers.csv")
OUTPUT_DIR = Path(r"data\test_data\test_output\unpaywall_test_420")
EMAIL = "designer1358@gmai.com"  # Required by Unpaywall API
SLEEP_TIME = 1.0  # Seconds between API calls (to avoid rate limits)

# =========================================================
# SETUP OUTPUT FOLDERS
# =========================================================
TEXT_DIR = OUTPUT_DIR / "text"
PDF_DIR = OUTPUT_DIR / "pdf"
LOG_DIR = OUTPUT_DIR / "logs" 
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
ERROR_LOG = LOG_DIR / "errors.log"

for d in [TEXT_DIR, PDF_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# =========================================================
# HELPER FUNCTIONS
# =========================================================
def safe_filename(doi: str) -> str:
    """Convert DOI to a filesystem-safe filename."""
    return doi.replace("/", "_").replace(":", "_").replace("?", "_")

def query_unpaywall(doi: str, email: str) -> dict:
    """Query Unpaywall API and return JSON response."""
    url = f"https://api.unpaywall.org/v2/{doi}?email={email}"
    r = requests.get(url, timeout=15)
    if r.status_code == 200:
        return r.json()
    else:
        raise Exception(f"Unpaywall returned {r.status_code} for DOI {doi}")

def download_file(url: str, out_path: Path):
    """Download file from a URL and save locally."""
    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 200:
            with open(out_path, "wb") as f:
                f.write(r.content)
            return True
        else:
            return False
    except Exception as e:
        raise Exception(f"Download failed: {e}")

def append_metadata_row(row_data: dict):
    """Append one row to metadata.csv (create header if not exists)."""
    header = [
        "doi",
        "is_oa",
        "oa_status",
        "best_url_html",
        "best_url_pdf",
        "file_type",
        "downloaded_file",
        "text_file",
        "status",
        "error_message"
    ]
    file_exists = METADATA_FILE.exists()
    with open(METADATA_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

def log_error(msg: str):
    """Write an error message to logs/errors.log."""
    with open(ERROR_LOG, "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now()}] {msg}\n")

def extract_text_from_html(html_path: Path, txt_path: Path):
    """Extract readable text from saved HTML and save to .txt."""
    try:
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        soup = BeautifulSoup(html, "html.parser")

        # remove unwanted tags
        for tag in soup(["script", "style", "nav", "header", "footer", "form", "noscript"]):
            tag.decompose()

        text = " ".join(soup.stripped_strings)
        txt_path.write_text(text, encoding="utf-8")
        print(f"🧾 Extracted text saved: {txt_path.name}")
        return True
    except Exception as e:
        print(f"⚠️ Text extraction failed for {html_path.name}: {e}")
        log_error(f"Text extraction failed for {html_path.name}: {e}")
        return False

# =========================================================
# MAIN WORKFLOW
# =========================================================
def main():
    print(f"🚀 Starting Unpaywall fetcher using input: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)
    dois = df["doi"].dropna().unique()

    for doi in dois:
        doi = doi.strip()
        print(f"\n🔍 Processing DOI: {doi}")
        safe_name = safe_filename(doi)

        # Skip already processed
        if  (TEXT_DIR / f"{safe_name}.html").exists() or (PDF_DIR / f"{safe_name}.pdf").exists():
            print("⚠️ Already downloaded. Skipping.")
            continue

        try:
            data = query_unpaywall(doi, EMAIL)
            oa = data.get("best_oa_location") or {}
            is_oa = data.get("is_oa", False)
            oa_status = data.get("oa_status", "")

            if not is_oa or not oa:
                append_metadata_row({
                    "doi": doi,
                    "is_oa": is_oa,
                    "oa_status": oa_status,
                    "best_url_html": None,
                    "best_url_pdf": None,
                    "file_type": None,
                    "downloaded_file": None,
                    "text_file": None,
                    "status": "no_oa",
                    "error_message": ""
                })
                print("🚫 No open-access version found.")
                continue

            html_url = oa.get("url_for_landing_page")
            pdf_url = oa.get("url_for_pdf")

            text_file_path = None
            if html_url:
                out_path = TEXT_DIR / f"{safe_name}.html"
                success = download_file(html_url, out_path)
                file_type = "text"
                if success:
                    text_file_path = TEXT_DIR / f"{safe_name}.txt"
                    extract_text_from_html(out_path, text_file_path)
            elif pdf_url:
                out_path = PDF_DIR / f"{safe_name}.pdf"
                success = download_file(pdf_url, out_path)
                file_type = "pdf"
            else:
                success = False
                out_path = None
                file_type = None

            status = "success" if success else "failed"

            append_metadata_row({
                "doi": doi,
                "is_oa": is_oa,
                "oa_status": oa_status,
                "best_url_html": html_url,
                "best_url_pdf": pdf_url,
                "file_type": file_type,
                "downloaded_file": str(out_path) if out_path else None,
                "text_file": str(text_file_path) if text_file_path else None,
                "status": status,
                "error_message": ""
            })

            print(f"✅ {file_type.upper()} saved to: {out_path}" if success else "❌ Download failed")

        except Exception as e:
            msg = f"Error for DOI {doi}: {e}"
            print(msg)
            log_error(msg)
            append_metadata_row({
                "doi": doi,
                "is_oa": None,
                "oa_status": None,
                "best_url_html": None,
                "best_url_pdf": None,
                "file_type": None,
                "downloaded_file": None,
                "text_file": None,
                "status": "error",
                "error_message": str(e)
            })

        time.sleep(SLEEP_TIME)  # Respect API rate limits

    print("\n✅ Fetching complete. Results saved to:")
    print(f" - {METADATA_FILE}")
    print(f" - {TEXT_DIR}")
    print(f" - {PDF_DIR}")


if __name__ == "__main__":
    main()
