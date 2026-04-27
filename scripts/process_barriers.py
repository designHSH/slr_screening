import re
import json
import os
from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "03_barrier_indentification_by_askr_gap" / "02-mapping" / "Rev_01" / "4_step_c_summary_table" / "all_barriers_summary_merged_with_counter.csv"


def find_barrier_column(df):
    candidates = [c for c in df.columns if re.search(r'barrier|statement|text', c, re.I)]
    if not candidates:
        return None
    # prefer exact match
    for c in candidates:
        if re.search(r'barrier\s*statement|barrier_statement', c, re.I):
            return c
    return candidates[0]


WORKLOAD_KEYWORDS = [
    'understand context', 'understand context of use', 'understand', 'specify user requirements',
    'produce design', 'produce design solutions', 'produce design solutions', 'produce design solutions',
]


def extract_workload(s):
    # look for patterns like 'during ...' or ending phrase 'during X'
    m = re.search(r"(during|while|in|for)\s+([^\,\.;]{1,80})", s, re.I)
    if m:
        cand = m.group(2).strip()
        for kw in WORKLOAD_KEYWORDS:
            if kw in cand.lower():
                return cand
    # as fallback, look for a trailing workload like 'during XYZ' at end
    m2 = re.search(r"during\s+(.+)$", s, re.I)
    if m2:
        return m2.group(1).strip()
    return "Not explicitly stated"


def extract_source(s):
    # Look for causal phrases indicating source
    patterns = [r"because of\s+([^\,\.;]+)", r"due to\s+([^\,\.;]+)", r"caused by\s+([^\,\.;]+)",
                r"resulting from\s+([^\,\.;]+)", r"lack of\s+([^\,\.;]+)", r"no\s+([^\,\.;]+)",
                r"not enough\s+([^\,\.;]+)"]
    for p in patterns:
        m = re.search(p, s, re.I)
        if m:
            return m.group(0).strip()

    # look for adjectival causes: 'limited X', 'insufficient X', 'restricted X', 'incomplete X', 'weak X'
    m2 = re.search(r"(limited|insufficient|restricted|incomplete|weak|partial|missing|inadequate|insufficient)\s+([^,;\.]{1,80})", s, re.I)
    if m2:
        return m2.group(0).strip()

    return "Not explicitly stated"


def extract_blocked_action(s):
    # Look for common blocking verbs and extract surrounding clause
    m = re.search(r"(prevent|preventing|prevents|unable to|cannot|can not|can't|stop|hinder|limit|prevented|prevented from)\s+([^\,\.;]+)", s, re.I)
    if m:
        return m.group(0).strip()

    # look for 'from' constructions: 'from doing X' or 'from using Y'
    m2 = re.search(r"from\s+([^\,\.;]+)", s, re.I)
    if m2:
        return m2.group(0).strip()

    # fallback: find verbs like 'use', 'define', 'describe', 'apply' and nearby words
    m3 = re.search(r"\b(use|using|define|defining|describe|describing|apply|using|identify|identify|forming|gather|gathering|maintain|maintaining)\s+([^\,\.;]{1,80})", s, re.I)
    if m3:
        return m3.group(0).strip()

    return "Not explicitly stated"


STOPWORDS = set([w.strip() for w in "the a an and or of to in for on with by at from as that this these those which what when where how why during".split()])
COMMON_VERBS = set(['prevent', 'preventing', 'prevents', 'use', 'using', 'define', 'describe', 'apply', 'identify', 'gather', 'maintain', 'carry', 'treat', 'keep', 'rely', 'select', 'compare'])


def extract_nouns(s, max_n=3):
    # crude noun phrase candidates: sequences of words with letters, apostrophes, /-&
    candidates = re.findall(r"[A-Za-z0-9'/-]+(?:\s+[A-Za-z0-9'/-]+){0,4}", s)
    filtered = []
    for c in candidates:
        low = c.lower().strip().strip("'\"")
        if len(low) < 3:
            continue
        # avoid pure numeric ids
        if re.fullmatch(r"\d+", low):
            continue
        # remove trailing verbs
        if low.split()[-1] in COMMON_VERBS:
            continue
        if low in filtered:
            continue
        filtered.append(c.strip())
        if len(filtered) >= max_n:
            break
    return filtered if filtered else ["Not explicitly stated"]


def select_side_nouns(important_nouns, source_text, target_text):
    source_side = []
    target_side = []
    s_low = source_text.lower() if source_text else ""
    t_low = target_text.lower() if target_text else ""
    for n in important_nouns:
        if n == "Not explicitly stated":
            continue
        nl = n.lower()
        if nl in s_low or any(tok in s_low for tok in nl.split()):
            source_side.append(n)
        if nl in t_low or any(tok in t_low for tok in nl.split()):
            target_side.append(n)
    # If nothing matched, try to heuristically pick noun from 'source' or 'target' texts
    if not source_side and source_text and source_text != "Not explicitly stated":
        src_n = extract_nouns(source_text, max_n=1)
        if src_n:
            source_side = src_n
    if not target_side and target_text and target_text != "Not explicitly stated":
        tgt_n = extract_nouns(target_text, max_n=1)
        if tgt_n:
            target_side = tgt_n
    if not source_side:
        source_side = ["Not explicitly stated"]
    if not target_side:
        target_side = ["Not explicitly stated"]
    return source_side, target_side


def analyze_statement(s):
    if not isinstance(s, str) or not s.strip():
        return {
            "barrier_statement": "",
            "barrier_source_or_condition": "Not explicitly stated",
            "blocked_or_affected_action": "Not explicitly stated",
            "important_nouns": ["Not explicitly stated"],
            "workload_context": "Not explicitly stated",
            "source_side_nouns": ["Not explicitly stated"],
            "target_side_nouns": ["Not explicitly stated"]
        }
    orig = s.strip()
    source = extract_source(orig)
    blocked = extract_blocked_action(orig)
    workload = extract_workload(orig)
    important = extract_nouns(orig, max_n=3)
    src_side, tgt_side = select_side_nouns(important, source, blocked)

    return {
        "barrier_statement": orig,
        "barrier_source_or_condition": source,
        "blocked_or_affected_action": blocked,
        "important_nouns": important,
        "workload_context": workload,
        "source_side_nouns": src_side,
        "target_side_nouns": tgt_side
    }


def extract_shortfall_and_ideal(s):
    """Extracts shortfall and ideal condition from barrier statement.

    Pattern assumed: "[shortfall] prevents implementers from [ideal] during [workload]"
    Returns tuple (shortfall, ideal) or ("Not explicitly stated", "Not explicitly stated").
    """
    if not isinstance(s, str) or not s.strip():
        return ("Not explicitly stated", "Not explicitly stated")
    s = s.strip()
    # try to capture shortfall and ideal up to 'during' (preferred)
    m = re.search(r"^(?P<shortfall>.+?)\s+prevent(?:s|ing)?\s+implementers\s+from\s+(?P<ideal>.+?)(?:\s+during\s+|\s+in\s+|\s+while\s+|$)", s, re.I)
    if m:
        short = m.group('shortfall').strip().strip(' ,.;')
        ideal = m.group('ideal').strip().strip(' ,.;')
        return (short if short else "Not explicitly stated", ideal if ideal else "Not explicitly stated")
    # fallback: split on ' prevents ' and ' from '
    try:
        left, right = s.split(' prevent', 1)
        short = left.strip().strip(' ,.;')
        # find 'from' in right
        if ' from ' in right:
            _, after = right.split(' from ', 1)
            # remove trailing ' during ...' if present
            after = re.split(r"\s+during\s+|\s+in\s+|\s+while\s+", after, flags=re.I)[0]
            ideal = after.strip().strip(' ,.;')
            return (short if short else "Not explicitly stated", ideal if ideal else "Not explicitly stated")
    except Exception:
        pass
    return ("Not explicitly stated", "Not explicitly stated")


def main():
    if not CSV_PATH.exists():
        print("CSV not found:", CSV_PATH)
        return
    df = pd.read_csv(CSV_PATH, dtype=str)
    col = find_barrier_column(df)
    if not col:
        print("Could not find a barrier statement column. Available columns:", list(df.columns))
        return
    # prepare per-column outputs
    bs_list = []
    src_cond_list = []
    blocked_list = []
    important_list = []
    workload_list = []
    source_side_list = []
    target_side_list = []
    shortfall_list = []
    ideal_list = []

    for i, val in enumerate(df[col].fillna('')):
        res = analyze_statement(val)
        bs_list.append(res.get('barrier_statement',''))
        src_cond_list.append(res.get('barrier_source_or_condition','Not explicitly stated'))
        blocked_list.append(res.get('blocked_or_affected_action','Not explicitly stated'))
        # join noun lists with pipe to avoid JSON objects in cells
        imp = res.get('important_nouns', [])
        important_list.append(' | '.join(imp) if isinstance(imp, list) else imp)
        workload_list.append(res.get('workload_context','Not explicitly stated'))
        sside = res.get('source_side_nouns', [])
        source_side_list.append(' | '.join(sside) if isinstance(sside, list) else sside)
        tside = res.get('target_side_nouns', [])
        target_side_list.append(' | '.join(tside) if isinstance(tside, list) else tside)
        # extract shortfall and ideal condition
        short, ideal = extract_shortfall_and_ideal(val)
        shortfall_list.append(short)
        ideal_list.append(ideal)

    # backup safely (avoid replace if file is open)
    import shutil
    bak = CSV_PATH.with_suffix('.csv.bak')
    try:
        if not bak.exists():
            shutil.copy2(CSV_PATH, bak)
    except Exception:
        pass

    # add columns to DataFrame (use the exact column names requested)
    df['barrier_statement'] = bs_list
    df['barrier_source_or_condition'] = src_cond_list
    df['blocked_or_affected_action'] = blocked_list
    df['important_nouns'] = important_list
    df['workload_context'] = workload_list
    df['source_side_nouns'] = source_side_list
    df['target_side_nouns'] = target_side_list
    # add requested shortfall and ideal condition columns
    df['shortfall'] = shortfall_list
    df['ideal condition'] = ideal_list

    df.to_csv(CSV_PATH, index=False)
    print(f"Processed {len(df)} rows. Updated file written to {CSV_PATH.name}. Backup: {bak.name if bak.exists() else 'not created'}")


if __name__ == '__main__':
    main()
