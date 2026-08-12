import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import accuracy_score

class EvaluationHarness:
    def __init__(self, threshold: float = 70.0):
        self.threshold = threshold

    def _extract_score(self, pred_item) -> dict:
        """Gelen JSON objesinden skor, saniye/milisaniye latency ve token toplamını çeker."""
        if isinstance(pred_item, str):
            try:
                pred_item = json.loads(pred_item)
            except Exception:
                return {"match_score": 0.0, "latency_ms": 0.0, "total_tokens": 0}

        if isinstance(pred_item, dict):
            # 1. Match Score
            score = pred_item.get("match_score", pred_item.get("score", pred_item.get("overall_score", 0.0)))
            
            # 2. Latency (Saniyeyi Milisaniyeye Çevirme Mantığı eklendi)
            if "latency_seconds" in pred_item:
                latency = float(pred_item["latency_seconds"]) * 1000.0
            elif "latency_ms" in pred_item:
                latency = float(pred_item["latency_ms"])
            else:
                latency = float(pred_item.get("latency", 0.0))

            # 3. Token Hesabı (prompt_tokens + completion_tokens)
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

    def evaluate_method(self, predictions: list, gold_scores: list[float]) -> dict:
        if len(predictions) != len(gold_scores):
            raise ValueError(f"Length mismatch: {len(predictions)} predictions vs {len(gold_scores)} gold labels.")
        
        parsed_preds = [self._extract_score(p) for p in parsed_preds_list if p is not None] if 'parsed_preds_list' in locals() else [self._extract_score(p) for p in predictions]
        
        pred_scores = [p["match_score"] for p in parsed_preds]
        latencies = [p["latency_ms"] for p in parsed_preds]
        tokens = [p["total_tokens"] for p in parsed_preds]

        # 1. Spearman Rank Correlation
        if len(set(pred_scores)) == 1 or len(set(gold_scores)) == 1:
            spearman_corr = 0.0
        else:
            corr, _ = spearmanr(pred_scores, gold_scores)
            spearman_corr = 0.0 if np.isnan(corr) else float(corr)

        # 2. Shortlisting Classification Accuracy @ Threshold
        pred_binary = [1 if s >= self.threshold else 0 for s in pred_scores]
        gold_binary = [1 if g >= self.threshold else 0 for g in gold_scores]
        acc = float(accuracy_score(gold_binary, pred_binary))

        # 3. Aggregated Latency & Token Usage
        avg_latency = float(np.mean(latencies)) if latencies else 0.0
        total_tokens_count = int(np.sum(tokens))
        avg_tokens = float(np.mean(tokens)) if tokens else 0.0

        return {
            "spearman_correlation": round(spearman_corr, 4),
            "accuracy_at_threshold": round(acc, 4),
            "avg_latency_ms": round(avg_latency, 2),
            "total_tokens": total_tokens_count,
            "avg_tokens_per_pair": round(avg_tokens, 1)
        }

    def compare_methods(self, methods_predictions: dict[str, list], gold_scores: list[float]) -> dict:
        results = {}
        for method_name, preds in methods_predictions.items():
            results[method_name] = self.evaluate_method(preds, gold_scores)
        return results


def load_dict_or_list_json(filepath):
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    if isinstance(data, dict):
       
        sorted_keys = sorted(data.keys(), key=lambda x: int(x.split('_')[1]) if '_' in x and x.split('_')[1].isdigit() else x)
        return [data[k] for k in sorted_keys]
    return data


if __name__ == "__main__":
    
    df_human = pd.read_csv("judge_vs_human_comparison.csv")
    
    if "human_overall_fit_rescaled" in df_human.columns:
        gold_scores = df_human["human_overall_fit_rescaled"].astype(float).tolist()
    else:
        df_key = pd.read_csv("pair_key.csv")
        bucket_mapping = {"strong": 90.0, "partial": 50.0, "weak": 10.0}
        gold_scores = df_key["fit_bucket_prescreen"].map(bucket_mapping).tolist()

    
    preds_2a = load_dict_or_list_json("embedding_results.json")
    preds_2b = load_dict_or_list_json("structured_extraction_results.json")
    preds_2c = load_dict_or_list_json("judge_results_cache.json")

    
    harness = EvaluationHarness(threshold=70.0)
    methods_dict = {
        "2a_embedding_similarity": preds_2a,
        "2b_structured_extraction": preds_2b,
        "2c_llm_as_judge": preds_2c
    }

    report = harness.compare_methods(methods_dict, gold_scores)

    
    with open("final_summary_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    df_report = pd.DataFrame(report).T
    df_report.to_csv("final_summary_metrics.csv")
    
    print("Final Scores")
    print(df_report.to_string())
