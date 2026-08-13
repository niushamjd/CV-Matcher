"""
Run the structured extraction + rules pipeline (method 2b) against all 30
gold-standard CV–JD pairs from Buse's annotation Excel.

Mirrors Niyousha's llm_judge_starter.py so results are directly comparable.
Outputs are saved to results/structured_extraction_results.json.

Run from the repo root or from methods/structured_extraction/:
    python methods/structured_extraction/run_on_gold_standard.py
"""

import json
import sys
import time
from pathlib import Path

# Allow imports from this directory regardless of where the script is called from
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_loader import load_pairs_from_excel
from extraction import extract_profile
from rule_scorer import score_match
from gap_explainer import explain_llm, explain_template
from llm_client import writing_quality_signal, get_last_usage

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "structured_extraction_results.json"


def run_pair(pair_id: str, resume_text: str, jd_text: str) -> dict:
    start = time.time()
    prompt_tokens = 0
    completion_tokens = 0

    def _accum():
        u = get_last_usage()
        nonlocal prompt_tokens, completion_tokens
        prompt_tokens += u["prompt_tokens"] or 0
        completion_tokens += u["completion_tokens"] or 0

    cv_profile = extract_profile(resume_text, source_type="cv")
    _accum()
    jd_profile = extract_profile(jd_text, source_type="jd")
    _accum()
    result = score_match(cv_profile, jd_profile)

    # Writing-quality signal (CV only, mirrors Niyousha's implementation)
    wq = writing_quality_signal(resume_text)
    _accum()
    result.sub_scores.writing_quality_score = wq["writing_quality_score"]

    # Gap Explainer Option B — LLM-based natural explanation
    try:
        result.explanation = explain_llm(result.structured_diff, wq["writing_quality_score"])
        _accum()
    except Exception:
        result.explanation = explain_template(result.structured_diff)  # fallback to template

    latency = round(time.time() - start, 2)

    return {
        "match_score": result.match_score,
        "sub_scores": {
            "skills": result.sub_scores.skills_score,
            "experience": result.sub_scores.experience_score,
            "education": result.sub_scores.education_score,
            "writing_quality": result.sub_scores.writing_quality_score,
        },
        "explanation": result.explanation,
        "writing_quality_explanation": wq["explanation"],
        "structured_diff": result.structured_diff.model_dump(),
        "source": "structured_extraction",
        "latency_seconds": latency,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


if __name__ == "__main__":
    print("Loading gold-standard pairs...")
    pairs = load_pairs_from_excel()
    print(f"Loaded {len(pairs)} pairs.\n")

    # Resume: load any results already saved so we don't redo completed pairs
    results = {}
    if RESULTS_PATH.exists():
        with open(RESULTS_PATH, encoding="utf-8") as f:
            results = json.load(f)
        print(f"Resuming — {len(results)} pairs already done, skipping them.\n")

    failed = []

    for pair in pairs:
        if pair.pair_id in results:
            print(f"=== {pair.pair_id} — already done, skipping ===")
            continue

        print(f"=== {pair.pair_id} ===")
        try:
            out = run_pair(pair.pair_id, pair.resume_text, pair.jd_text)
            results[pair.pair_id] = out
            print(f"  match_score : {out['match_score']}")
            print(f"  sub_scores  : skills={out['sub_scores']['skills']}  "
                  f"exp={out['sub_scores']['experience']}  "
                  f"edu={out['sub_scores']['education']}")
            print(f"  explanation : {out['explanation'][:120]}...")
            print(f"  latency     : {out['latency_seconds']}s")
            # Save after each pair so progress is never lost
            with open(RESULTS_PATH, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            print(f"  [FAILED] {e}")
            failed.append(pair.pair_id)

        # 15-second pause between pairs to stay under Groq's 6k TPM free-tier limit
        time.sleep(15)
        print()

    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"Saved {len(results)} results to {RESULTS_PATH}")
    if failed:
        print(f"Failed pairs: {failed}")
