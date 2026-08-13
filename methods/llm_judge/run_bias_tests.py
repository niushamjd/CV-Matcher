"""
Bias robustness tests for Method 2c (LLM-as-Judge).

Tests:
  1. Formatting bias  — 4 layout variants per CV
  2. Name bias        — 8 demographic name variants per CV + NEUTRAL control

The LLM sees raw CV text directly, so name bias is especially meaningful here.

Output:
  results/bias_format_results_2c.json
  results/bias_name_results_2c.json
  results/bias_format_summary_2c.csv
  results/bias_name_summary_2c.csv

Run from repo root:
  export GEMINI_API_KEY=your_key
  python methods/llm_judge/run_bias_tests.py
"""

import csv
import json
import statistics
import sys
import time
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_judge_starter import llm_as_judge

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
BIAS_DIR = REPO_ROOT / "data" / "week2_variants"
FORMAT_DIR = BIAS_DIR / "format"
NAME_DIR = BIAS_DIR / "names"
JD_DIR = BIAS_DIR / "original_controls" / "jds"
RESULTS_DIR = REPO_ROOT / "results"
BASELINE_PATH = RESULTS_DIR / "judge_results_cache.json"

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


def run_variants(pairs, variant_map, pdf_dir, jd_texts, baseline, save_path, label):
    results = {}
    if save_path.exists():
        with open(save_path) as f:
            results = json.load(f)

    print(f"\n=== {label} ===")
    for pair_id in pairs:
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
                continue
            pdf_path = pdf_dir / f"{pair_id}{suffix}"
            if not pdf_path.exists():
                print(f"  {pair_id}/{variant_name} — file missing, skipping")
                continue
            try:
                cv_text = extract_pdf(pdf_path)
                res = llm_as_judge(cv_text, jd_texts[pair_id])
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

    print("Loading JD texts...")
    jd_texts = {}
    for pair_id in PAIRS:
        jd_path = JD_DIR / f"{pair_id}_JD.pdf"
        jd_texts[pair_id] = extract_pdf(jd_path)
        print(f"  {pair_id} JD loaded")

    fmt_results = run_variants(
        PAIRS, FORMAT_VARIANTS, FORMAT_DIR, jd_texts, baseline,
        RESULTS_DIR / "bias_format_results_2c.json", "FORMAT BIAS (2c)",
    )
    fmt_rows = write_summary_csv(
        fmt_results, RESULTS_DIR / "bias_format_summary_2c.csv", list(FORMAT_VARIANTS.keys())
    )

    name_variant_map = {c: f"_{c}.pdf" for c in NAME_VARIANTS}
    name_results = run_variants(
        PAIRS, name_variant_map, NAME_DIR, jd_texts, baseline,
        RESULTS_DIR / "bias_name_results_2c.json", "NAME BIAS (2c)",
    )
    name_rows = write_summary_csv(
        name_results, RESULTS_DIR / "bias_name_summary_2c.csv", NAME_VARIANTS
    )

    print("\n==============================")
    print("FORMAT BIAS SUMMARY (2c)")
    print("==============================")
    print(f"{'Pair':<10} {'Baseline':>9} {'single':>8} {'two_col':>8} {'table':>8} {'sec_ord':>8} {'Range':>7} {'Std':>6}")
    for r in fmt_rows:
        print(f"{r['pair_id']:<10} {str(r['baseline']):>9} "
              f"{str(r.get('single_column','?')):>8} {str(r.get('two_column','?')):>8} "
              f"{str(r.get('table','?')):>8} {str(r.get('section_order','?')):>8} "
              f"{str(r.get('score_range','?')):>7} {str(r.get('score_std','?')):>6}")

    print("\n==============================")
    print("NAME BIAS SUMMARY (2c)")
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
