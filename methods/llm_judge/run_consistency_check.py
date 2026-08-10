import json
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

from llm_judge_starter import load_pairs_from_excel, check_consistency

N_PAIRS = 4
N_RUNS = 4

pairs = load_pairs_from_excel("data/cv_matcher_candidate_pool.xlsx")
subset = pairs[:N_PAIRS]

results = {}
for pair in subset:
    pair_id = pair.get("pair_id") or pair.get("resume_id")
    print(f"Running {pair_id} x{N_RUNS}...")
    result = check_consistency(pair["resume_text"], pair["jd_text"], n_runs=N_RUNS)
    results[pair_id] = result
    print(json.dumps(result, indent=2))
    print()

print("=== Summary ===")
for pair_id, r in results.items():
    print(f"{pair_id}: scores={r['scores']} spread={r['spread']}")

avg_spread = sum(r["spread"] for r in results.values()) / len(results)
print(f"\nAverage spread across {len(results)} pairs, {N_RUNS} runs each: {avg_spread:.1f}")