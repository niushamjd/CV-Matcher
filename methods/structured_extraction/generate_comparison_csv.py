"""
Generate structured_extraction_vs_human.csv — mirrors Niyousha's judge_vs_human_comparison.csv.

Run from repo root:
    python methods/structured_extraction/generate_comparison_csv.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "structured_extraction_results.json"
EXCEL_PATH = REPO_ROOT / "data" / "annotation" / "annotated_pairs.xlsx"
OUT_PATH = REPO_ROOT / "results" / "structured_extraction_vs_human.csv"


def rescale_1to5(v):
    """Convert 1-5 human score to 0-100."""
    if pd.isna(v):
        return None
    return round((float(v) - 1) / 4 * 100, 1)


def main():
    with open(RESULTS_PATH, encoding="utf-8") as f:
        results = json.load(f)

    df_ann = pd.read_excel(EXCEL_PATH, sheet_name="Sheet1")
    df_ann = df_ann.set_index("pair_id")

    rows = []
    for pair_id, res in results.items():
        if pair_id not in df_ann.index:
            continue
        ann = df_ann.loc[pair_id]

        human_1to5 = ann.get("overall_fit_1to5")
        rows.append({
            "pair_id": pair_id,
            "match_score": res["match_score"],
            "human_overall_fit_1to5": human_1to5,
            "human_overall_fit_rescaled": rescale_1to5(human_1to5),
            "skills_pred": res["sub_scores"].get("skills"),
            "skills_human_1to5": ann.get("subscore_skills_1to5"),
            "experience_pred": res["sub_scores"].get("experience"),
            "experience_human_1to5": ann.get("subscore_experience_1to5"),
            "education_pred": res["sub_scores"].get("education"),
            "education_human_1to5": ann.get("subscore_education_1to5"),
            "writing_quality_pred": res["sub_scores"].get("writing_quality"),
            "shortlist_decision": ann.get("shortlist_decision"),
            "hard_requirement_flag_met": ann.get("hard_requirement_flag_met"),
            "human_rationale": ann.get("rationale"),
            "explanation": res.get("explanation"),
        })

    df = pd.DataFrame(rows).sort_values("pair_id").reset_index(drop=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved {len(df)} rows to {OUT_PATH}")

    # Quick metrics
    valid = df.dropna(subset=["match_score", "human_overall_fit_rescaled"])
    mae = (valid["match_score"] - valid["human_overall_fit_rescaled"]).abs().mean()
    r, p = spearmanr(valid["match_score"], valid["human_overall_fit_rescaled"])
    print(f"\nMetrics vs human (n={len(valid)}):")
    print(f"  MAE      : {mae:.1f} pts")
    print(f"  Spearman : r={r:.3f}  p={p:.4f}")

    # Comparison vs Niyousha's judge
    judge_path = REPO_ROOT / "results" / "judge_vs_human_comparison.csv"
    if judge_path.exists():
        df_j = pd.read_csv(judge_path).set_index("pair_id")
        common = df.set_index("pair_id").join(df_j[["match_score"]], rsuffix="_judge").dropna(
            subset=["match_score", "match_score_judge", "human_overall_fit_rescaled"]
        )
        mae_j = (common["match_score_judge"] - common["human_overall_fit_rescaled"]).abs().mean()
        r_j, _ = spearmanr(common["match_score_judge"], common["human_overall_fit_rescaled"])
        print(f"\nSide-by-side on {len(common)} shared pairs:")
        print(f"  Structured extraction — MAE: {mae:.1f}  Spearman: {r:.3f}")
        print(f"  LLM judge             — MAE: {mae_j:.1f}  Spearman: {r_j:.3f}")


if __name__ == "__main__":
    main()
