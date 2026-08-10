import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from llm_judge_starter import load_pairs_from_excel, llm_as_judge

DATA_PATH = "data/cv_matcher_candidate_pool.xlsx"
LABELS_PATH = "data/annotation-buse.csv"
CACHE_PATH = "data/judge_results_cache.json"
COMPARISON_CSV_PATH = "data/judge_vs_human_comparison.csv"


def load_labels(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {row["pair_id"]: row for row in rows}


def to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def rank(values):
    indexed = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman(x, y):
    n = len(x)
    if n < 2:
        return None
    rx = rank(x)
    ry = rank(y)
    mean_rx = sum(rx) / n
    mean_ry = sum(ry) / n
    cov = sum((rx[i] - mean_rx) * (ry[i] - mean_ry) for i in range(n))
    var_x = sum((rx[i] - mean_rx) ** 2 for i in range(n))
    var_y = sum((ry[i] - mean_ry) ** 2 for i in range(n))
    if var_x == 0 or var_y == 0:
        return None
    return cov / (var_x ** 0.5 * var_y ** 0.5)


def load_cache():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)
    return {}


def save_cache(cache):
    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(cache, f, indent=2)


def get_judge_result(pair_id, cv_text, jd_text, cache):
    if pair_id in cache:
        return cache[pair_id]
    result = llm_as_judge(cv_text, jd_text)
    cache[pair_id] = result
    save_cache(cache)
    return result


def classification_metrics(match_scores, shortlist_labels, threshold=60):
    tp = fp = tn = fn = 0
    total = 0
    positives = 0
    for score, label in zip(match_scores, shortlist_labels):
        if label is None:
            continue
        predicted = score >= threshold
        actual = str(label).strip().lower() == "shortlist"
        total += 1
        positives += int(actual)
        if predicted and actual:
            tp += 1
        elif predicted and not actual:
            fp += 1
        elif not predicted and actual:
            fn += 1
        else:
            tn += 1

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (2 * precision * recall / (precision + recall)) if precision and recall else None
    majority_baseline_accuracy = max(positives, total - positives) / total if total else None

    return {
        "n": total,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "majority_baseline_accuracy": majority_baseline_accuracy,
    }


if __name__ == "__main__":
    pairs = load_pairs_from_excel(DATA_PATH)
    labels_by_id = load_labels(LABELS_PATH)

    labeled_pairs = []
    for pair in pairs:
        pair_id = pair.get("pair_id") or pair.get("resume_id")
        label = labels_by_id.get(pair_id)
        if label is not None and to_float(label.get("overall_fit_1to5")) is not None:
            labeled_pairs.append((pair, label))

    print(f"{len(pairs)} total pairs, {len(labeled_pairs)} with gold labels")

    if not labeled_pairs:
        print("No gold labels found yet. Nothing to calibrate against.")
        sys.exit(0)

    cache = load_cache()

    rows = []
    for pair, label in labeled_pairs:
        pair_id = pair.get("pair_id") or pair.get("resume_id")
        result = get_judge_result(pair_id, pair["resume_text"], pair["jd_text"], cache)

        human_overall = to_float(label.get("overall_fit_1to5"))
        skills_human = to_float(label.get("subscore_skills_1to5"))
        experience_human = to_float(label.get("subscore_experience_1to5"))
        education_human = to_float(label.get("subscore_education_1to5"))

        rows.append({
            "pair_id": pair_id,
            "match_score": result["match_score"],
            "human_overall_fit_1to5": human_overall,
            "human_overall_fit_rescaled": human_overall * 20,
            "skills_pred": result["sub_scores"]["skills"],
            "skills_human_1to5": skills_human,
            "experience_pred": result["sub_scores"]["experience"],
            "experience_human_1to5": experience_human,
            "education_pred": result["sub_scores"]["education"],
            "education_human_1to5": education_human,
            "shortlist_decision": label.get("shortlist_decision"),
            "hard_requirement_flag_met": label.get("hard_requirement_flag_met"),
            "human_rationale": label.get("rationale"),
            "judge_explanation": result["explanation"],
        })

    os.makedirs(os.path.dirname(COMPARISON_CSV_PATH), exist_ok=True)
    with open(COMPARISON_CSV_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote comparison table to {COMPARISON_CSV_PATH}")

    match_scores = [r["match_score"] for r in rows]
    human_overall = [r["human_overall_fit_rescaled"] for r in rows]

    skills_pred = [r["skills_pred"] for r in rows if r["skills_human_1to5"] is not None]
    skills_human = [r["skills_human_1to5"] * 20 for r in rows if r["skills_human_1to5"] is not None]

    experience_pred = [r["experience_pred"] for r in rows if r["experience_human_1to5"] is not None]
    experience_human = [r["experience_human_1to5"] * 20 for r in rows if r["experience_human_1to5"] is not None]

    education_pred = [r["education_pred"] for r in rows if r["education_human_1to5"] is not None]
    education_human = [r["education_human_1to5"] * 20 for r in rows if r["education_human_1to5"] is not None]

    shortlist_labels = [r["shortlist_decision"] for r in rows]

    print()
    print("=== Spearman correlation (judge vs. human, rescaled to 0-100) ===")
    print(f"Overall match_score vs overall_fit_1to5: {spearman(match_scores, human_overall)}")
    print(f"Skills sub-score vs subscore_skills_1to5: {spearman(skills_pred, skills_human)}")
    print(f"Experience sub-score vs subscore_experience_1to5: {spearman(experience_pred, experience_human)}")
    print(f"Education sub-score vs subscore_education_1to5: {spearman(education_pred, education_human)}")

    metrics = classification_metrics(match_scores, shortlist_labels, threshold=60)
    print()
    print(f"=== Shortlisting classification at threshold=60 (n={metrics['n']}) ===")
    print(f"Accuracy: {metrics['accuracy']}")
    print(f"Majority-class baseline accuracy: {metrics['majority_baseline_accuracy']}")
    print(f"Precision: {metrics['precision']}")
    print(f"Recall: {metrics['recall']}")
    print(f"F1: {metrics['f1']}")