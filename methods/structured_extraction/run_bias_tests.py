"""
Bias robustness tests for Method 2b (Structured Extraction + Rules).

Tests:
  1. Formatting bias  — 4 layout variants per CV (single-col, two-col, table, section-order)
  2. Name bias        — 8 demographic name variants per CV (WH/AA/TR/IN × M/F) + NEUTRAL control

For each variant, only CV extraction changes — the JD is fixed per pair.
JD extractions are cached to avoid redundant API calls.

Each variant result is stored in the same format as structured_extraction_results.json
so Ipek's eval harness can consume them directly.

Output:
  annotation/results/bias_format_results.json
  annotation/results/bias_name_results.json
  annotation/results/bias_format_summary.csv
  annotation/results/bias_name_summary.csv

Run from repo root or methods/structured_extraction/:
  export LLM_PROVIDER=gemini
  export GEMINI_API_KEY=your_key
  python methods/structured_extraction/run_bias_tests.py
"""

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extraction import extract_profile
from rule_scorer import score_match
from llm_client import get_last_usage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIAS_DIR = REPO_ROOT / "formatting&name bias inputs"
FORMAT_DIR = BIAS_DIR / "format_variants"
NAME_DIR = BIAS_DIR / "name_variants"
JD_DIR = BIAS_DIR / "original_controls" / "jds"
RESULTS_DIR = REPO_ROOT / "annotation" / "results"

BASELINE_PATH = RESULTS_DIR / "structured_extraction_results.json"

PAIRS = [
    "PAIR_01", "PAIR_09", "PAIR_10", "PAIR_11", "PAIR_13", "PAIR_14",
    "PAIR_15", "PAIR_17", "PAIR_20", "PAIR_25", "PAIR_27", "PAIR_29",
]

FORMAT_VARIANTS = {
    "single_column": "_single_column.pdf",
    "two_column":    "_two_column.pdf",
    "table":         "_table.pdf",
    "section_order": "_section_order.pdf",
}

NAME_VARIANTS = ["NEUTRAL", "WH_F", "WH_M", "AA_F", "AA_M", "TR_F", "TR_M", "IN_F", "IN_M"]


def extract_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()


def score_cv_vs_jd(cv_text: str, jd_profile) -> dict:
    """Returns a result dict in the same format as structured_extraction_results.json."""
    start = time.time()
    prompt_tokens, completion_tokens = 0, 0

    cv_profile = extract_profile(cv_text, source_type="cv")
    u = get_last_usage()
    prompt_tokens += u["prompt_tokens"] or 0
    completion_tokens += u["completion_tokens"] or 0

    result = score_match(cv_profile, jd_profile)
    latency = round(time.time() - start, 2)

    return {
        "match_score": result.match_score,
        "sub_scores": {
            "skills": result.sub_scores.skills_score,
            "experience": result.sub_scores.experience_score,
            "education": result.sub_scores.education_score,
        },
        "source": "structured_extraction",
        "latency_seconds": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def run_variants(pairs, variant_map, pdf_dir, jd_profiles, baseline, save_path, label):
    results = {}
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)

    print(f"\n=== {label} ===")
    for pair_id in pairs:
        # Resume at variant level — retry any null variants within saved pairs
        if pair_id not in results:
            results[pair_id] = {
                "baseline": baseline.get(pair_id, {}).get("match_score"),
                "variants": {},
            }

        all_done = all(
            results[pair_id]["variants"].get(v) is not None
            for v in variant_map
        )
        if all_done:
            print(f"  {pair_id} — all variants done, skipping")
            continue

        for variant_name, suffix in variant_map.items():
            if results[pair_id]["variants"].get(variant_name) is not None:
                continue  # already succeeded
            pdf_path = pdf_dir / f"{pair_id}{suffix}"
            if not pdf_path.exists():
                print(f"  {pair_id}/{variant_name} — file missing, skipping")
                continue
            try:
                cv_text = extract_pdf(pdf_path)
                res = score_cv_vs_jd(cv_text, jd_profiles[pair_id])
                results[pair_id]["variants"][variant_name] = res
                print(f"  {pair_id}/{variant_name}: match_score={res['match_score']}")
                time.sleep(5)
            except Exception as e:
                print(f"  {pair_id}/{variant_name} FAILED: {e}")
                results[pair_id]["variants"][variant_name] = None

        with open(save_path, "w") as f:
            json.dump(results, f, indent=2)

    return results


def write_summary_csv(results, csv_path, variant_keys):
    rows = []
    for pair_id, data in results.items():
        scores = [
            data["variants"][k]["match_score"]
            for k in variant_keys
            if data["variants"].get(k) is not None
        ]
        row = {"pair_id": pair_id, "baseline": data["baseline"]}
        for k in variant_keys:
            v = data["variants"].get(k)
            row[k] = v["match_score"] if v else None
        row["score_range"] = round(max(scores) - min(scores), 1) if scores else None
        row["score_std"] = round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0
        rows.append(row)

    fields = ["pair_id", "baseline"] + list(variant_keys) + ["score_range", "score_std"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    return rows


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(BASELINE_PATH) as f:
        baseline = json.load(f)

    # Cache JD extractions — each JD is reused up to 13 times
    print("Extracting JD profiles (cached per pair)...")
    jd_profiles = {}
    for pair_id in PAIRS:
        jd_path = JD_DIR / f"{pair_id}_JD.pdf"
        jd_text = extract_pdf(jd_path)
        jd_profiles[pair_id] = extract_profile(jd_text, source_type="jd")
        print(f"  {pair_id} JD extracted")
        time.sleep(5)

    # Format bias
    fmt_results = run_variants(
        PAIRS, FORMAT_VARIANTS, FORMAT_DIR, jd_profiles, baseline,
        RESULTS_DIR / "bias_format_results.json", "FORMAT BIAS",
    )
    fmt_rows = write_summary_csv(
        fmt_results, RESULTS_DIR / "bias_format_summary.csv", list(FORMAT_VARIANTS.keys())
    )

    # Name bias — variant_map: {"NEUTRAL": "_NEUTRAL.pdf", ...}
    name_variant_map = {c: f"_{c}.pdf" for c in NAME_VARIANTS}
    name_results = run_variants(
        PAIRS, name_variant_map, NAME_DIR, jd_profiles, baseline,
        RESULTS_DIR / "bias_name_results.json", "NAME BIAS",
    )
    name_rows = write_summary_csv(
        name_results, RESULTS_DIR / "bias_name_summary.csv", NAME_VARIANTS
    )

    # Print report
    print("\n==============================")
    print("FORMAT BIAS SUMMARY")
    print("==============================")
    print(f"{'Pair':<10} {'Baseline':>9} {'single':>8} {'two_col':>8} {'table':>8} {'sec_ord':>8} {'Range':>7} {'Std':>6}")
    for r in fmt_rows:
        print(f"{r['pair_id']:<10} {str(r['baseline']):>9} "
              f"{str(r.get('single_column','?')):>8} {str(r.get('two_column','?')):>8} "
              f"{str(r.get('table','?')):>8} {str(r.get('section_order','?')):>8} "
              f"{str(r.get('score_range','?')):>7} {str(r.get('score_std','?')):>6}")

    print("\n==============================")
    print("NAME BIAS SUMMARY")
    print("==============================")
    print(f"{'Pair':<10} {'Baseline':>9} {'NEUTRAL':>8} {'WH_F':>6} {'WH_M':>6} "
          f"{'AA_F':>6} {'AA_M':>6} {'TR_F':>6} {'TR_M':>6} {'IN_F':>6} {'IN_M':>6} {'Range':>7} {'Std':>6}")
    for r in name_rows:
        print(f"{r['pair_id']:<10} {str(r['baseline']):>9} "
              f"{str(r.get('NEUTRAL','?')):>8} {str(r.get('WH_F','?')):>6} {str(r.get('WH_M','?')):>6} "
              f"{str(r.get('AA_F','?')):>6} {str(r.get('AA_M','?')):>6} {str(r.get('TR_F','?')):>6} "
              f"{str(r.get('TR_M','?')):>6} {str(r.get('IN_F','?')):>6} {str(r.get('IN_M','?')):>6} "
              f"{str(r.get('score_range','?')):>7} {str(r.get('score_std','?')):>6}")

    print(f"\nSaved to {RESULTS_DIR}/")


if __name__ == "__main__":
    main()
