# CV Matcher — Explainable CV–Job Matching System (ECMS)

CVMatcher evaluates how well a candidate CV matches a job description using three complementary methods, ranging from fast local embeddings to a full LLM-as-judge pipeline.

## Demo Video

[Watch the demo on YouTube](https://www.youtube.com/watch?v=cViJXYbMVmU)

---

## Project Report

Project report is added to the repository named "NLP_in_Industry_CV_Matcher.pdf".

---

## Project Structure

```
CVMatcher/
├── methods/
│   ├── embedding_similarity/      # Method 2a — dense embedding similarity
│   ├── structured_extraction/     # Method 2b — LLM extraction + rule scoring
│   └── llm_judge/                 # Method 2c — LLM-as-judge
├── data/
│   ├── cv_matcher_candidate_pool.xlsx   # 30 gold-standard CV/JD pairs
│   ├── annotation/                      # Human annotations and pair packets
│   └── week2_variants/                  # Bias robustness test PDFs
├── results/                       # All output JSON/CSV files
├── demo/
│   ├── demo.py                    # Live single-pair demo, all three methods
│   ├── demo_bias.py               # Live bias-robustness demo, all three methods
│   ├── demo_output.txt            # Terminal output from demo.py
│   └── demo_bias_output.txt       # Terminal output from demo_bias.py
├── eval_harness.py                # Unified evaluation across all methods
└── requirements.txt
```

---

## Setup

```bash
git clone https://github.com/niushamjd/CV-Matcher.git
cd CV-Matcher

conda create -n matchlens python=3.11
conda activate matchlens

pip install -r requirements.txt
```

## API Keys

Create a `.env` file in the repo root (never commit this):

```
GEMINI_API_KEY=your-key-here
GROQ_API_KEY=your-key-here
```

Method 2c always uses Gemini — get a free key at [aistudio.google.com](https://aistudio.google.com), no credit card needed.

Method 2b (`llm_client.py`) defaults to **Groq** (`LLM_PROVIDER=groq`), not Gemini — get a free key at [console.groq.com](https://console.groq.com). To run 2b on Gemini instead, set `LLM_PROVIDER=gemini`; to run it fully offline, set `LLM_PROVIDER=ollama` (requires a local Ollama install, no API key needed).

---

## Method 2a — Dense Embedding Similarity

Runs fully locally, no API key needed.

```bash
python methods/embedding_similarity/embedding_matcher.py
```

Reads the 30 gold-standard pairs from `data/cv_matcher_candidate_pool.xlsx` and writes scores to `results/embedding_results.json`.

To run the bias robustness tests (format + name variants):

```bash
python methods/embedding_similarity/run_bias_tests.py
```

**How it works:** Uses `all-MiniLM-L6-v2` (SentenceTransformers) with section-aware cosine similarity — Skills at 40%, Experience at 40%, overall document at 20%.

---

## Method 2b — Structured Extraction + Rules

```bash
cd methods/structured_extraction
python run_on_gold_standard.py
```

Reads the 30 gold-standard pairs and writes scores to `results/structured_extraction_results.json`.

To run the bias robustness tests:

```bash
python methods/structured_extraction/run_bias_tests.py
```

**How it works:** An LLM call parses each CV and JD into a shared schema (skills, experience, education), then rule-based scoring computes sub-scores deterministically. A gap explainer turns the structured diff into a natural-language rationale.

| File | Role |
|---|---|
| `schema.py` | Shared Pydantic contracts: `ExtractedProfile`, `StructuredDiff`, `MatchResult` |
| `llm_client.py` | LLM abstraction using Gemini via OpenAI-compatible API |
| `extraction.py` | Parses raw CV/JD text into `ExtractedProfile` |
| `rule_scorer.py` | Skill overlap + experience/education threshold logic |
| `gap_explainer.py` | Turns `StructuredDiff` into a human-readable explanation |

---

## Method 2c — LLM-as-Judge

```bash
python methods/llm_judge/llm_judge_starter.py
```

Reads the 30 gold-standard pairs and caches results in `results/judge_results_cache.json`.

To run the bias robustness tests:

```bash
python methods/llm_judge/run_bias_tests.py
```

**How it works:** The judge receives raw CV and JD text in a single prompt grounded in a two-tier must-have vs. nice-to-have recruiting rubric, returning a match score (0–100), sub-scores, and a natural-language explanation.

---

## Evaluation Harness

Compares all three methods against the 30-pair gold standard and generates plots:

```bash
python eval_harness.py
```

Outputs:
- `results/final_summary_metrics.json` — Spearman ρ, accuracy @70, latency, token usage
- `results/fig1_scatter.png` — predicted vs. human score scatter plots
- `results/fig2_bias.png` — bias robustness bar chart

---

## Demo

Two standalone scripts, built for the project video, that run all three
methods live end-to-end on a single gold-standard pair — **PAIR_13**, a
Senior HR Specialist the human annotator rated 5/5 (shortlist). No
batching, no cached results: every score you see is computed on the spot.

```bash
python demo/demo.py
```
Runs Methods 2a, 2b, and 2c on PAIR_13, narrating each step as it happens
(model loads, each LLM call), then prints a final summary table comparing
all three match scores against the human label.
Terminal output: [`demo/demo_output.txt`](demo/demo_output.txt).

```bash
python demo/demo_bias.py
```
Runs all three methods on PAIR_13's baseline CV plus all 4 formatting
variants and all 9 demographic name variants (13 live re-scorings per
method), prints each method's full bias breakdown against baseline, and
closes with a one-sentence takeaway on which method was most sensitive
to formatting vs. name changes.
Terminal output: [`demo/demo_bias_output.txt`](demo/demo_bias_output.txt).

Both scripts need the same `.env` setup as Methods 2b/2c above
(`GEMINI_API_KEY`, and `GROQ_API_KEY` unless you set `LLM_PROVIDER=gemini`).

---

## Results Summary

| Method | Spearman ρ | Acc @70 | Avg Latency | Tokens/Pair |
|---|---|---|---|---|
| 2a: Embedding Similarity | 0.517 | 66.7% | 83 ms | 0 |
| 2b: Structured Extraction | 0.177 | 63.3% | 4825 ms | 5215 |
| 2c: LLM-as-Judge | **0.788** | **76.7%** | 2886 ms | 2964 |

(n=30 gold pairs; see `results/final_summary_metrics.json`.)
