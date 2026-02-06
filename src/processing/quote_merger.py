import json
import csv
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = Path(r"data\test_data\gpt_test\barrier_identification_test\Quote_extraction\revised011")
OUTPUT_CSV = INPUT_DIR / "quotes.csv"

# -----------------------------
# CSV column structure
# -----------------------------
CSV_COLUMNS = [
    "source_json_file",
    "paper_id",
    "quote_id",
    "quote_anchor",
    "section",
    "section_code",
    "quote",
    "quote_with_context"
    
]

# -----------------------------
# Process all JSON files
# -----------------------------
rows = []

for json_file in INPUT_DIR.glob("*.json"):
    try:
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        source_json = json_file.name
        if isinstance(data, dict):
            quotes = data.get("quote", [])
        elif isinstance(data, list):
            quotes = data
        else:
            raise ValueError(f"Unexpected JSON root type: {type(data).__name__}")

        for quote in quotes:
            rows.append({
                "source_json_file": source_json,
                "paper_id": quote.get("paper_id", ""),
                "quote_id": quote.get("quote_id", ""),
                "quote_anchor": quote.get("quote_anchor", ""),
                "section": quote.get("section", ""),
                "section_code": quote.get("section_code", ""),
                "quote": quote.get("quote", ""),
                "quote_with_context": quote.get("quote_with_context", "")
                
            })

        print(f"Processed: {source_json} -> {len(quotes)} quotes")

    except Exception as e:
        print(f"Error processing {json_file.name}: {e}")


# -----------------------------
# Write CSV
# -----------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV created successfully: {OUTPUT_CSV}")
print(f"   Total barriers written: {len(rows)}")
