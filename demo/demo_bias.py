"""
Live bias-robustness demo for the project video: runs all three
CV-Matcher methods on PAIR_13's baseline CV plus every formatting and
name variant, live (not from the cached results/bias_*.csv files),
prints the biased results at the end of each method's run, and closes
with a one-sentence takeaway.

PAIR_13 (Senior Human Resources Specialist) is one of the 12 pairs in
the bias-robustness subset -- see data/week2_variants/. Only match_score
is computed per variant (the deterministic quantity the report's bias
tables are built from); Method 2b's writing-quality and gap-explainer
calls are skipped here since neither one feeds match_score.

Setup (see README.md "API Keys"):
    GEMINI_API_KEY   required (method 2c, and method 2b if LLM_PROVIDER=gemini)
    GROQ_API_KEY     required for method 2b's default provider (LLM_PROVIDER=groq)

Run from the repo root:
    python demo/demo_bias.py
"""

import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import pdfplumber

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "methods" / "structured_extraction"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "embedding_similarity"))
sys.path.insert(0, str(REPO_ROOT / "methods" / "llm_judge"))

PAIR_ID = "PAIR_13"
VARIANTS_DIR = REPO_ROOT / "data" / "week2_variants"
BASELINE_CV = VARIANTS_DIR / "original_controls" / "cvs" / f"{PAIR_ID}_CV.pdf"
JD_PDF = VARIANTS_DIR / "original_controls" / "jds" / f"{PAIR_ID}_JD.pdf"

FORMAT_VARIANTS = {
    "single_column": VARIANTS_DIR / "format" / f"{PAIR_ID}_single_column.pdf",
    "two_column":    VARIANTS_DIR / "format" / f"{PAIR_ID}_two_column.pdf",
    "table":         VARIANTS_DIR / "format" / f"{PAIR_ID}_table.pdf",
    "section_order": VARIANTS_DIR / "format" / f"{PAIR_ID}_section_order.pdf",
}
NAME_VARIANTS = {
    name: VARIANTS_DIR / "names" / f"{PAIR_ID}_{name}.pdf"
    for name in ["NEUTRAL", "WH_F", "WH_M", "AA_F", "AA_M", "TR_F", "TR_M", "IN_F", "IN_M"]
}


def extract_pdf(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(p.extract_text() or "" for p in pdf.pages).strip()


def section(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def step(msg: str):
    print(f"  -> {msg}")


def print_results_table(label: str, baseline_score: float, format_scores: dict, name_scores: dict):
    def stats(scores: dict) -> tuple[float, float, str]:
        vals = list(scores.values())
        rng = round(max(vals) - min(vals), 2)
        worst = max(scores, key=lambda k: abs(scores[k] - baseline_score))
        return rng, scores[worst], worst

    print(f"\n--- {label}: biased results (baseline match_score = {baseline_score}) ---")
    print("Format variants:")
    for k, v in format_scores.items():
        print(f"  {k:<16}: {v:>6.2f}   (Δ {v - baseline_score:+.2f})")
    fmt_range, _, fmt_worst = stats(format_scores)
    print(f"  format range    : {fmt_range} pts (largest shift: {fmt_worst})")

    print("Name variants:")
    for k, v in name_scores.items():
        print(f"  {k:<16}: {v:>6.2f}   (Δ {v - baseline_score:+.2f})")
    name_range, _, name_worst = stats(name_scores)
    print(f"  name range      : {name_range} pts (largest shift: {name_worst})")

    return fmt_range, name_range


def run_2a(jd_text: str, cv_texts: dict) -> dict:
    from embedding_matcher import EmbeddingMatcher

    step("Loading all-MiniLM-L6-v2 locally and scoring every variant (no API calls)...")
    matcher = EmbeddingMatcher()
    scores = {}
    for name, cv_text in cv_texts.items():
        scores[name] = matcher.match(cv_text, jd_text, section_aware=True)["match_score"]
    return scores


def run_2b(jd_text: str, cv_texts: dict) -> dict:
    from extraction import extract_profile
    from rule_scorer import score_match

    step("Calling LLM once to extract the fixed JD profile (reused across all variants)...")
    jd_profile = extract_profile(jd_text, source_type="jd")

    scores = {}
    for name, cv_text in cv_texts.items():
        step(f"Calling LLM to extract structured profile for CV variant '{name}'...")
        cv_profile = extract_profile(cv_text, source_type="cv")
        result = score_match(cv_profile, jd_profile)
        scores[name] = result.match_score
        time.sleep(2)  # light pacing to stay under free-tier rate limits
    return scores


def run_2c(jd_text: str, cv_texts: dict) -> dict:
    from llm_judge_starter import llm_as_judge

    scores = {}
    for name, cv_text in cv_texts.items():
        step(f"Calling Gemini judge on CV variant '{name}'...")
        result = llm_as_judge(cv_text, jd_text)
        scores[name] = result["match_score"]
    return scores


def main():
    section(f"CV Matcher bias demo -- {PAIR_ID}")
    step("Extracting text from baseline CV, 4 format variants, and 9 name variants (pdfplumber)...")
    jd_text = extract_pdf(JD_PDF)
    baseline_text = extract_pdf(BASELINE_CV)
    format_texts = {name: extract_pdf(path) for name, path in FORMAT_VARIANTS.items()}
    name_texts = {name: extract_pdf(path) for name, path in NAME_VARIANTS.items()}
    all_cv_texts = {"baseline": baseline_text, **format_texts, **name_texts}

    method_ranges = {}

    section("Method 2a -- Dense Embedding Similarity")
    try:
        scores = run_2a(jd_text, all_cv_texts)
        fmt_range, name_range = print_results_table(
            "2a Dense Embedding", scores["baseline"],
            {k: scores[k] for k in FORMAT_VARIANTS}, {k: scores[k] for k in NAME_VARIANTS},
        )
        method_ranges["2a"] = (fmt_range, name_range)
    except Exception as e:
        print(f"[skipped] {e}")

    section("Method 2b -- Structured Extraction + Rules")
    try:
        scores = run_2b(jd_text, all_cv_texts)
        fmt_range, name_range = print_results_table(
            "2b Structured Extraction", scores["baseline"],
            {k: scores[k] for k in FORMAT_VARIANTS}, {k: scores[k] for k in NAME_VARIANTS},
        )
        method_ranges["2b"] = (fmt_range, name_range)
    except Exception as e:
        print(f"[skipped] {e}")

    section("Method 2c -- LLM-as-Judge")
    try:
        scores = run_2c(jd_text, all_cv_texts)
        fmt_range, name_range = print_results_table(
            "2c LLM-as-Judge", scores["baseline"],
            {k: scores[k] for k in FORMAT_VARIANTS}, {k: scores[k] for k in NAME_VARIANTS},
        )
        method_ranges["2c"] = (fmt_range, name_range)
    except Exception as e:
        print(f"[skipped] {e}")

    section("Result")
    if method_ranges:
        most_format_sensitive = max(method_ranges, key=lambda m: method_ranges[m][0])
        most_name_sensitive = max(method_ranges, key=lambda m: method_ranges[m][1])
        parts = ", ".join(
            f"{m} (format {r[0]} pts / name {r[1]} pts)" for m, r in method_ranges.items()
        )
        print(
            f"For {PAIR_ID}, score ranges across variants were {parts} -- "
            f"{most_format_sensitive} was the most sensitive to formatting changes and "
            f"{most_name_sensitive} was the most sensitive to name changes, "
            f"consistent with the project's bias-robustness findings."
        )
    else:
        print("No method completed successfully -- check API keys and try again.")


if __name__ == "__main__":
    main()
