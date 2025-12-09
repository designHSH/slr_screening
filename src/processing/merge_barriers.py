import json
import csv
from pathlib import Path

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = Path(r"data\test_data\barrier_identification_test\2nd_run\output_2nd")
OUTPUT_CSV = INPUT_DIR / "barriers_merged.csv"

# -----------------------------
# CSV column structure
# -----------------------------
CSV_COLUMNS = [
    "source_json_file",
    "source_text_file",
    "barrier_id",
    "actor",
    "barrier_type",
    "description",
    "justification",
    "reference_excerpt",
    "certainty",
    "note"
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
        source_text = data.get("file_name", "")
        note = data.get("note", "")

        for barrier in data.get("barriers", []):
            rows.append({
                "source_json_file": source_json,
                "source_text_file": source_text,
                "barrier_id": barrier.get("barrier_id", ""),
                "actor": barrier.get("actor", ""),
                "barrier_type": barrier.get("barrier_type", ""),
                "description": barrier.get("description", ""),
                "justification": barrier.get("justification", ""),
                "reference_excerpt": barrier.get("reference_excerpt", ""),
                "certainty": barrier.get("certainty", ""),
                "note": note
            })

        print(f"Processed: {source_json}  → {len(data.get('barriers', []))} barriers")

    except Exception as e:
        print(f"❌ Error processing {json_file.name}: {e}")


# -----------------------------
# Write CSV
# -----------------------------
with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=CSV_COLUMNS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\n✅ CSV created successfully: {OUTPUT_CSV}")
print(f"   Total barriers written: {len(rows)}")
