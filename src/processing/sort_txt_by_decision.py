from pathlib import Path
import sys
import re
import shutil
from typing import Optional, Tuple
import pandas as pd

# --- Configuration (optional): set defaults here for quick runs in VS Code ---
# If you set these, the script will use them as defaults when prompting.
DEFAULT_CSV_PATH = ""
DEFAULT_TXT_DIR = ""
DEFAULT_OUTDIR = ""   # leave "" to use <txt_dir>/sorted_txt
DEFAULT_FILENAME_COL = ""  # e.g., "file_name" (leave "" to auto-detect)
DEFAULT_DECISION_COL = ""  # e.g., "decision" (leave "" to auto-detect)

INCLUDE_TOKENS = {"include", "included", "keep", "accept", "accepted", "yes", "y", "true", "t", "1"}
EXCLUDE_TOKENS = {"exclude", "excluded", "reject", "rejected", "remove", "no", "n", "false", "f", "0"}

FILENAME_HINTS = ["file_name", "filename", "file", "name", "txt", "path", "doc", "paper"]
DECISION_HINTS = ["decision", "include", "included", "label", "status", "screen", "verdict", "class", "keep", "accept"]

def _normalize_filename(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().strip('"').strip("'")
    v = Path(v).name
    if not v.lower().endswith(".txt"):
        v = f"{v}.txt"
    return v

def _normalize_decision(value) -> Optional[str]:
    if value is None:
        return None
    v = str(value).strip().lower()
    v = re.sub(r"[^a-z0-9\s]", "", v).strip()

    if v.isdigit():
        if v == "1":
            return "include"
        if v == "0":
            return "exclude"

    if v in INCLUDE_TOKENS:
        return "include"
    if v in EXCLUDE_TOKENS:
        return "exclude"

    words = set(v.split())
    if words & INCLUDE_TOKENS and not words & EXCLUDE_TOKENS:
        return "include"
    if words & EXCLUDE_TOKENS and not words & INCLUDE_TOKENS:
        return "exclude"
    return None

def _auto_detect_columns(df: pd.DataFrame) -> Tuple[Optional[str], Optional[str]]:
    cols = [c for c in df.columns if isinstance(c, str)]

    fname_cands = [c for c in cols if any(h in c.lower() for h in FILENAME_HINTS)]
    if not fname_cands:
        for c in cols:
            sample = df[c].dropna().astype(str).head(50).str.lower()
            hits = int(sample.str.endswith(".txt").sum())
            if hits >= max(3, len(sample)//5):
                fname_cands.append(c)

    decision_cands = [c for c in cols if any(h in c.lower() for h in DECISION_HINTS)]

    def fname_score(col):
        series = df[col].dropna().astype(str).str.lower()
        return int(series.str.endswith(".txt").sum())

    def decision_score(col):
        series = df[col].dropna().astype(str)
        return int(series.map(_normalize_decision).notna().sum())

    if fname_cands:
        fname_cands = sorted(fname_cands, key=fname_score, reverse=True)
    if decision_cands:
        decision_cands = sorted(decision_cands, key=decision_score, reverse=True)

    return (fname_cands[0] if fname_cands else None,
            decision_cands[0] if decision_cands else None)

def _copy_files(csv_path: Path,
                txt_dir: Path,
                out_base: Path,
                filename_col: Optional[str] = None,
                decision_col: Optional[str] = None) -> Path:
    df = pd.read_csv(csv_path)

    if not filename_col or not decision_col:
        auto_fname, auto_decision = _auto_detect_columns(df)
        filename_col = filename_col or auto_fname
        decision_col = decision_col or auto_decision

    if not filename_col:
        raise ValueError(f"Could not detect filename column. CSV columns: {list(df.columns)}")
    if not decision_col:
        raise ValueError(f"Could not detect decision column. CSV columns: {list(df.columns)}")

    df["_normalized_filename"] = df[filename_col].map(_normalize_filename)
    df["_normalized_decision"] = df[decision_col].map(_normalize_decision)

    out_included = out_base / "included"
    out_excluded = out_base / "excluded"
    out_included.mkdir(parents=True, exist_ok=True)
    out_excluded.mkdir(parents=True, exist_ok=True)

    txt_files = {p.name.lower(): p for p in txt_dir.glob("*.txt")}

    copied = []
    missing = []
    undecided = []

    for _, row in df.iterrows():
        fname = row["_normalized_filename"]
        decision = row["_normalized_decision"]
        if not fname:
            missing.append((row, "empty filename"))
            continue

        src = txt_files.get(fname.lower())
        if src is None:
            stem = Path(fname).stem.lower()
            candidates = [p for name, p in txt_files.items() if Path(name).stem.lower() == stem]
            if candidates:
                src = candidates[0]

        if decision not in {"include", "exclude"}:
            undecided.append((row, decision))
            continue
        if src is None:
            missing.append((row, "not found"))
            continue

        dest = (out_included if decision == "include" else out_excluded) / src.name
        shutil.copy2(src, dest)
        copied.append((src.name, decision, str(dest)))

    # Logs
    log_df = pd.DataFrame(copied, columns=["filename", "decision", "destination"])
    log_df.to_csv(out_base / "copy_log.csv", index=False)

    with open(out_base / "report.txt", "w", encoding="utf-8") as f:
        f.write(f"CSV: {csv_path}\nTXT folder: {txt_dir}\nOutput: {out_base}\n\n")
        f.write(f"Copied: {len(copied)}\nMissing: {len(missing)}\nUndecided: {len(undecided)}\n\n")
        if missing:
            f.write("Missing examples (up to 10):\n")
            for r, reason in missing[:10]:
                f.write(f"  - {r.get('_normalized_filename')}: {reason}\n")
            f.write("\n")
        if undecided:
            f.write("Undecided examples (up to 10):\n")
            for r, d in undecided[:10]:
                f.write(f"  - {r.get('_normalized_filename')}: raw='{d}'\n")

    return out_base

def _prompt_or_args():
    # If CLI args present, use them; otherwise prompt.
    if len(sys.argv) >= 3:
        csv_path = Path(sys.argv[1]).expanduser()
        txt_dir = Path(sys.argv[2]).expanduser()
        outdir = Path(sys.argv[3]).expanduser() if len(sys.argv) >= 4 else None
        fname_col = sys.argv[4] if len(sys.argv) >= 5 else None
        decision_col = sys.argv[5] if len(sys.argv) >= 6 else None
        return csv_path, txt_dir, outdir, fname_col, decision_col

    print("=== Sort TXT by Inclusion (VS Code Friendly) ===")
    csv_in = input(f"CSV path [{DEFAULT_CSV_PATH or 'required'}]: ").strip() or DEFAULT_CSV_PATH
    txt_in = input(f"TXT folder [{DEFAULT_TXT_DIR or 'required'}]: ").strip() or DEFAULT_TXT_DIR
    out_in = input(f"Output folder (Enter to use <txt>/sorted_txt) [{DEFAULT_OUTDIR or ''}]: ").strip() or DEFAULT_OUTDIR
    fname_col = input(f"Filename column (Enter to auto-detect) [{DEFAULT_FILENAME_COL or ''}]: ").strip() or DEFAULT_FILENAME_COL
    decision_col = input(f"Decision column (Enter to auto-detect) [{DEFAULT_DECISION_COL or ''}]: ").strip() or DEFAULT_DECISION_COL

    if not csv_in or not txt_in:
        print("ERROR: CSV path and TXT folder are required.")
        sys.exit(2)

    csv_path = Path(csv_in).expanduser()
    txt_dir = Path(txt_in).expanduser()
    outdir = Path(out_in).expanduser() if out_in else None
    fname_col = fname_col or None
    decision_col = decision_col or None
    return csv_path, txt_dir, outdir, fname_col, decision_col

def main():
    csv_path, txt_dir, outdir, fname_col, decision_col = _prompt_or_args()

    if not csv_path.exists():
        print(f"ERROR: CSV not found: {csv_path}")
        sys.exit(2)
    if not txt_dir.exists():
        print(f"ERROR: TXT folder not found: {txt_dir}")
        sys.exit(2)

    out_base = outdir if outdir else (txt_dir / "sorted_txt")
    out_base.mkdir(parents=True, exist_ok=True)

    try:
        result_dir = _copy_files(csv_path, txt_dir, out_base, fname_col, decision_col)
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(2)

    print("\nDone.")
    print(f"Included: {result_dir / 'included'}")
    print(f"Excluded: {result_dir / 'excluded'}")
    print(f"Log:      {result_dir / 'copy_log.csv'}")
    print(f"Report:   {result_dir / 'report.txt'}")

if __name__ == "__main__":
    main()
