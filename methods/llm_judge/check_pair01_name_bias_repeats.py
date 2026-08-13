import json
import statistics
import sys
from pathlib import Path

import pdfplumber

sys.path.insert(0, str(Path(__file__).resolve().parent))
from llm_judge_starter import llm_as_judge

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
NAME_DIR = REPO_ROOT / "data" / "week2_variants" / "names"
JD_PATH = REPO_ROOT / "data" / "week2_variants" / "original_controls" / "jds" / "PAIR_01_JD.pdf"
RESULTS_PATH = REPO_ROOT / "results" / "pair01_name_bias_repeats.json"

NAME_VARIANTS = ["NEUTRAL", "WH_F", "WH_M", "AA_F", "AA_M", "TR_F", "TR_M", "IN_F", "IN_M"]
N_RUNS = 3


def extract_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()


if __name__ == "__main__":
    jd_text = extract_pdf(JD_PATH)

    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH) as f:
            results = json.load(f)

    for variant in NAME_VARIANTS:
        if variant in results and len(results[variant]["scores"]) >= N_RUNS:
            print(f"{variant} — already done, skipping")
            continue

        pdf_path = NAME_DIR / f"PAIR_01_{variant}.pdf"
        cv_text = extract_pdf(pdf_path)

        scores = results.get(variant, {}).get("scores", [])
        while len(scores) < N_RUNS:
            res = llm_as_judge(cv_text, jd_text)
            scores.append(res["match_score"])
            print(f"{variant} run {len(scores)}/{N_RUNS}: match_score={res['match_score']}")

        results[variant] = {
            "scores": scores,
            "mean": round(statistics.mean(scores), 1),
            "stdev": round(statistics.stdev(scores), 2) if len(scores) > 1 else 0.0,
        }

        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)

    print("\n=== Summary ===")
    print(f"{'Variant':<10} {'Scores':<20} {'Mean':>6} {'Stdev':>6}")
    for variant, data in results.items():
        print(f"{variant:<10} {str(data['scores']):<20} {data['mean']:>6} {data['stdev']:>6}")

    print(f"\nSaved to {RESULTS_PATH}")