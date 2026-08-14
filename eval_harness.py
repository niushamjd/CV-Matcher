import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

class EvaluationHarness:
    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold

    def _extract_score(self, pred_item) -> dict:
        """Extracts the score, latency (seconds/milliseconds), and token total from the incoming JSON object."""
        if isinstance(pred_item, str):
            try:
                pred_item = json.loads(pred_item)
            except Exception:
                return {"match_score": 0.0, "latency_ms": 0.0, "total_tokens": 0}

        if isinstance(pred_item, dict):
            # 1. Match Score
            score = pred_item.get("match_score", pred_item.get("score", pred_item.get("overall_score", 0.0)))

            # 2. Latency (seconds-to-milliseconds conversion logic added)
            if "latency_seconds" in pred_item:
                latency = float(pred_item["latency_seconds"]) * 1000.0
            elif "latency_ms" in pred_item:
                latency = float(pred_item["latency_ms"])
            else:
                latency = float(pred_item.get("latency", 0.0))

            # 3. Token count (prompt_tokens + completion_tokens)
            if "prompt_tokens" in pred_item or "completion_tokens" in pred_item:
                p_tok = pred_item.get("prompt_tokens", 0) or 0
                c_tok = pred_item.get("completion_tokens", 0) or 0
                tokens = p_tok + c_tok
            else:
                tokens = pred_item.get("total_tokens", pred_item.get("tokens", 0))

            return {
                "match_score": float(score) if score is not None else 0.0,
                "latency_ms": round(latency, 2),
                "total_tokens": int(tokens) if tokens is not None else 0
            }

        return {"match_score": 0.0, "latency_ms": 0.0, "total_tokens": 0}

    def evaluate_method(self, predictions: dict, gold_scores: dict) -> dict:
        """predictions and gold_scores must both be {pair_id: value} dicts.
        Only pair_ids present in both are compared, joined explicitly by key —
        never by list position, since prediction files and the gold CSV are
        not guaranteed to share the same row order."""
        shared_ids = sorted(set(predictions.keys()) & set(gold_scores.keys()))
        if not shared_ids:
            raise ValueError("No overlapping pair_ids between predictions and gold_scores.")
        missing_in_preds = set(gold_scores.keys()) - set(predictions.keys())
        missing_in_gold = set(predictions.keys()) - set(gold_scores.keys())
        if missing_in_preds:
            print(f"Warning: {len(missing_in_preds)} gold pair_ids have no prediction: {sorted(missing_in_preds)}")
        if missing_in_gold:
            print(f"Warning: {len(missing_in_gold)} predicted pair_ids have no gold label: {sorted(missing_in_gold)}")

        parsed_preds = [self._extract_score(predictions[pid]) for pid in shared_ids]
        gold_list = [float(gold_scores[pid]) for pid in shared_ids]

        pred_scores = [p["match_score"] for p in parsed_preds]
        latencies = [p["latency_ms"] for p in parsed_preds]
        tokens = [p["total_tokens"] for p in parsed_preds]

        # 1. Spearman Rank Correlation
        if len(set(pred_scores)) == 1 or len(set(gold_list)) == 1:
            spearman_corr = 0.0
        else:
            corr, _ = spearmanr(pred_scores, gold_list)
            spearman_corr = 0.0 if np.isnan(corr) else float(corr)

        # 2. Shortlisting Classification Accuracy @ Threshold
        pred_binary = [1 if s >= self.threshold else 0 for s in pred_scores]
        gold_binary = [1 if g >= self.threshold else 0 for g in gold_list]
        acc = float(accuracy_score(gold_binary, pred_binary))

        # 3. Aggregated Latency & Token Usage
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        total_tokens_count = int(np.sum(tokens))
        avg_tokens = float(np.mean(tokens)) if tokens else 0.0

        return {
            "n_pairs": len(shared_ids),
            "spearman_correlation": round(spearman_corr, 4),
            "accuracy_at_threshold": round(acc, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens_count,
            "avg_tokens_per_pair": round(avg_tokens, 1)
        }

    def compare_methods(self, methods_predictions: dict[str, dict], gold_scores: dict) -> dict:
        results = {}
        for method_name, preds in methods_predictions.items():
            results[method_name] = self.evaluate_method(preds, gold_scores)
        return results


def generate_plots(methods_preds, gold_scores, report):
    import csv
    import matplotlib.pyplot as plt
    import numpy as np

    method_meta = {
        "2a_embedding_similarity":  ("Method 2a\nDense Embedding",  "#4C72B0"),
        "2b_structured_extraction": ("Method 2b\nStructured Extraction", "#55A868"),
        "2c_llm_as_judge":          ("Method 2c\nLLM-as-Judge",     "#C44E52"),
    }

    # ── Figure 1: Predicted vs Human scatter (3 panels) ──────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    harness = EvaluationHarness()

    for ax, (method, preds) in zip(axes, methods_preds.items()):
        label, color = method_meta[method]
        shared_ids = sorted(set(preds.keys()) & set(gold_scores.keys()))
        parsed = [harness._extract_score(preds[pid]) for pid in shared_ids]
        pred_scores = [p["match_score"] for p in parsed]
        gold_list = [gold_scores[pid] for pid in shared_ids]
        rho = report[method]["spearman_correlation"]

        ax.scatter(gold_list, pred_scores, color=color, alpha=0.75, s=55,
                   edgecolors="white", linewidth=0.5)
        ax.plot([0, 100], [0, 100], "k--", linewidth=0.8, alpha=0.35)
        ax.set_xlabel("Human Score (0–100)", fontsize=10)
        ax.set_ylabel("Predicted Score (0–100)", fontsize=10)
        ax.set_title(f"{label}\n$\\rho = {rho:.3f}$", fontsize=10)
        ax.set_xlim(-5, 105)
        ax.set_ylim(-5, 105)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Predicted vs Human Scores", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results/fig1_scatter.pdf", bbox_inches="tight")
    plt.savefig("results/fig1_scatter.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved results/fig1_scatter.pdf/.png")

    # ── Figure 2: Bias robustness grouped bar chart ───────────────────────────
    def avg_range_from_csv(path):
        try:
            with open(path) as f:
                rows = list(csv.DictReader(f))
            vals = [float(r["score_range"]) for r in rows if r.get("score_range")]
            return round(sum(vals) / len(vals), 1) if vals else 0.0
        except FileNotFoundError:
            return 0.0

    fmt_avgs  = [avg_range_from_csv(f"results/bias_format_summary_2a.csv"),
                 avg_range_from_csv(f"results/bias_format_summary.csv"),
                 avg_range_from_csv(f"results/bias_format_summary_2c.csv")]
    name_avgs = [avg_range_from_csv(f"results/bias_name_summary_2a.csv"),
                 avg_range_from_csv(f"results/bias_name_summary.csv"),
                 avg_range_from_csv(f"results/bias_name_summary_2c.csv")]

    x = np.arange(3)
    w = 0.35
    xlabels = ["2a: Embedding", "2b: Structured", "2c: LLM Judge"]

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    b1 = ax2.bar(x - w/2, fmt_avgs,  w, label="Format bias", color="#4C72B0", alpha=0.85)
    b2 = ax2.bar(x + w/2, name_avgs, w, label="Name bias",   color="#C44E52", alpha=0.85)

    for bar in list(b1) + list(b2):
        ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                 f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=9)

    ax2.set_ylabel("Avg score range (pts) — lower is more robust", fontsize=10)
    ax2.set_title("Bias Robustness Across Methods (n=12 pairs)", fontsize=11, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(xlabels, fontsize=10)
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", alpha=0.3)
    ax2.set_ylim(0, max(max(fmt_avgs), max(name_avgs)) * 1.3)

    plt.tight_layout()
    plt.savefig("results/fig2_bias.pdf", bbox_inches="tight")
    plt.savefig("results/fig2_bias.png", bbox_inches="tight", dpi=150)
    plt.close()
    print("Saved results/fig2_bias.pdf/.png")


def load_pair_id_dict(filepath):
    """Loads a predictions JSON and returns it as a {pair_id: prediction} dict.
    Does NOT sort or flatten into a list — callers must join by pair_id
    explicitly against gold_scores, never by position, since row/key order
    across files is not guaranteed to match."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        return data

    if isinstance(data, list):
        result = {}
        for item in data:
            pid = item.get("pair_id") if isinstance(item, dict) else None
            if pid is None:
                raise ValueError(
                    f"{filepath} is a list but an item has no 'pair_id' key; "
                    "cannot safely join to gold scores without one."
                )
            result[pid] = item
        return result

    raise ValueError(f"{filepath}: unsupported JSON shape {type(data)}")


if __name__ == "__main__":

    df_human = pd.read_csv("results/judge_vs_human_comparison.csv")

    if "human_overall_fit_rescaled" in df_human.columns:
        gold_scores = dict(zip(df_human["pair_id"], df_human["human_overall_fit_rescaled"].astype(float)))
    else:
        df_key = pd.read_csv("data/annotation/annotation_packets/pair_key.csv")
        bucket_mapping = {"strong": 90.0, "partial": 50.0, "weak": 10.0}
        gold_scores = dict(zip(df_key["pair_id"], df_key["fit_bucket_prescreen"].map(bucket_mapping)))

    preds_2a = load_pair_id_dict("results/embedding_results.json")
    preds_2b = load_pair_id_dict("results/structured_extraction_results.json")
    preds_2c = load_pair_id_dict("results/judge_results_cache.json")

    harness = EvaluationHarness(threshold=70.0)
    methods_dict = {
        "2a_embedding_similarity": preds_2a,
        "2b_structured_extraction": preds_2b,
        "2c_llm_as_judge": preds_2c
    }

    report = harness.compare_methods(methods_dict, gold_scores)

    with open("results/final_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_report = pd.DataFrame(report).T
    df_report.to_csv("results/final_summary_metrics.csv")

    print("Final Scores")
    print(df_report.to_string())

    generate_plots(methods_dict, gold_scores, report)