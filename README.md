# CV-Matcher

CVmatcher helps job seekers and recruiters quickly assess how well a CV matches a job description, using LLM-based methods instead of keyword-only ATS filtering.

## Setup

```bash
git clone https://github.com/niushamjd/CV-Matcher.git
cd CV-Matcher

python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

## API keys

### Gemini (for method 2c — LLM-as-judge)

1. Get a free key at [aistudio.google.com](https://aistudio.google.com) (API Keys → Create API key).
2. Create a `.env` file in the repo root (never commit this file):

```
GEMINI_API_KEY=your-key-here
```

### Groq (for method 2b — Structured Extraction + Rules)

1. Get a free key at [console.groq.com](https://console.groq.com) — no credit card needed.
2. Add it to the same `.env` file:

```
GROQ_API_KEY=your-key-here
```

## Gold-standard data

The script reads real CV/JD pairs from `data/cv_matcher_candidate_pool.xlsx` (not tracked in git — download it from the shared Drive and place it in `data/` yourself). It reads the `Candidate Pool` sheet, not `Final Selection` — the text lives in the `resume_text` and `jd_text` columns of `Candidate Pool`, filtered to the 30 rows where `FINAL_SELECTION (y/n) == "y"`.

```bash
mkdir -p data
mv ~/Downloads/cv_matcher_candidate_pool.xlsx data/
```

## Running the LLM-as-judge (2c)

```bash
python3 methods/llm_judge/llm_judge_starter.py
```

By default this runs the judge and writing-quality signal against all 30 real gold-standard CV/JD pairs from `data/cv_matcher_candidate_pool.xlsx`. Pair order matches the sheet's row order, which is intentionally randomized per the labeling rubric.

To run against the six synthetic edge-case samples instead (strong match, clear mismatch, negation, poorly-written-but-qualified, hallucination, synonym/implied-experience) — useful for quick sanity checks without burning API quota on 30 pairs — swap the loop in `__main__` to iterate `SAMPLES` instead of `load_pairs_from_excel(...)`.

### Output schema

```json
{
  "match_score": 0-100,
  "sub_scores": {
    "skills": 0-100,
    "experience": 0-100,
    "education": 0-100
  },
  "explanation": "text",
  "source": "llm_judge"
}
```

### Notes

- Default model is `gemini-3.5-flash-lite`. Free tier has daily request limits — check [aistudio.google.com/usage](https://aistudio.google.com/usage) if you hit a 429 error. A full 30-pair run uses ~60 calls (judge + writing-quality per pair).
- `check_consistency()` in the script re-runs a pair multiple times to check score stability; call it manually.
- `load_pairs_from_excel()` reads the real gold-standard dataset. `load_pairs_from_csv()` is a fallback stub if the data ever comes as CSV instead.

---

## Running Structured Extraction + Rules (2b)

```bash
cd methods/structured_extraction
python test_pipeline.py
```

This runs the full pipeline (LLM extraction → rule scoring → gap explanation) against sample CVs from the Kaggle Resume dataset.

The pipeline has two data source options — edit the `USE_PDF` flag at the top of `test_pipeline.py`:

- `USE_PDF = False` — reads from `Resume/Resume.csv` (Kaggle CSV format)
- `USE_PDF = True` — reads from `data/data/<CATEGORY>/*.pdf`

### What each file does

| File | Role |
|---|---|
| `schema.py` | Shared Pydantic contracts: `ExtractedProfile`, `StructuredDiff`, `MatchResult` |
| `llm_client.py` | LLM abstraction — defaults to Groq (`llama-3.3-70b-versatile`), Ollama fallback |
| `extraction.py` | LLM prompt that parses raw CV/JD text into a structured `ExtractedProfile` |
| `rule_scorer.py` | Skill overlap + experience/education threshold logic → `MatchResult` |
| `gap_explainer.py` | Turns `StructuredDiff` into a human-readable explanation |
| `data_loader.py` | Loads CVs from CSV or PDF, returns `list[ResumeRecord]` |

### Output schema

```json
{
  "method": "structured_extraction",
  "match_score": 0-100,
  "sub_scores": {
    "skills_score": 0-100,
    "experience_score": 0-100,
    "education_score": 0-100
  },
  "explanation": "text",
  "structured_diff": {
    "matched_skills": ["..."],
    "missing_skills": ["..."],
    "extra_skills": ["..."],
    "years_required": 3.0,
    "years_candidate": 5.0,
    "years_gap": -2.0,
    "education_match": true
  }
}
```

### Notes

- The rule scorer is intentionally simple (set intersection + threshold) — it is a baseline, not the star of the show.
- Skill normalization aliases (e.g. `ms excel` → `microsoft excel`) live in `rule_scorer.py`'s `_SKILL_ALIASES` dict — add entries there if you spot mismatches.
- `gap_explainer.py` has a deterministic template (Option A, default) and a stubbed LLM version (Option B) for nicer phrasing if time allows.
