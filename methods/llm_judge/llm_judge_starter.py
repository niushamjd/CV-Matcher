import os
import json
import re
import time
import csv

from dotenv import load_dotenv
from openai import OpenAI, RateLimitError

load_dotenv()

client = OpenAI(
    api_key=os.environ.get("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)
MODEL = "gemini-3.5-flash-lite"


def call_with_retry(**kwargs):
    for attempt in range(5):
        try:
            return client.chat.completions.create(**kwargs)
        except RateLimitError:
            time.sleep(2 ** attempt)
    return client.chat.completions.create(**kwargs)


JUDGE_SYSTEM_PROMPT = """You are an experienced technical recruiter scoring how well a
candidate's CV matches a job description. Follow a two-tier evaluation, the
way a real screening rubric works:

  1. Must-haves first: required skills, minimum years of experience, and
     required education/certifications. A missing must-have should weigh
     heavily on the score, even if everything else looks strong.
  2. Nice-to-haves second: preferred skills, domain relevance, and career
     progression. These are bonuses, not filters — don't let a missing
     nice-to-have sink an otherwise strong match the way a missing
     must-have should.

When judging experience, weigh accomplishments over responsibilities. "Led a
team of six that reduced time-to-fill by 30%" is stronger evidence than
"managed a team" — vague, responsibility-only phrasing (common in
AI-generated resumes optimized for keyword matching) should not score as
well as specific, measurable claims, even if the keywords match.

Be careful with:
  - Negation ("no experience with X" is NOT a match for X)
  - Synonyms/implied experience (e.g. "led a team of engineers" implies
    people-management even if the JD says "management experience")
  - Do not invent skills or experience that are not stated or clearly implied.
  - Keyword-only matches: a resume that repeats JD phrasing without concrete
    evidence should not automatically score high.

Score skills, experience, and education separately, then give an overall
match_score. The overall score should reflect your holistic judgment (e.g.
a required-skill gap can matter more than the average of the three parts) —
don't just average the sub-scores mechanically.

Return ONLY valid JSON, no other text:
{
  "match_score": <int 0-100>,
  "sub_scores": {
    "skills": <int 0-100>,
    "experience": <int 0-100>,
    "education": <int 0-100>
  },
  "explanation": "<2-3 sentence rationale citing specific matches and gaps>"
}
"""

JUDGE_USER_TEMPLATE = """Job Description:
---
{jd_text}
---

Candidate CV:
---
{cv_text}
---

Score this match."""


def llm_as_judge(cv_text: str, jd_text: str) -> dict:
    start = time.time()
    response = call_with_retry(
        model=MODEL,
        max_tokens=2000,
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            {"role": "user", "content": JUDGE_USER_TEMPLATE.format(jd_text=jd_text, cv_text=cv_text)},
        ],
    )
    latency = time.time() - start
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError(f"Empty response from model. Full response object for debugging:\n{response}")
    raw = raw.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output was:\n{raw}") from e
    return {
        "match_score": parsed["match_score"],
        "sub_scores": parsed["sub_scores"],
        "explanation": parsed["explanation"],
        "source": "llm_judge",
        "latency_seconds": round(latency, 2),
        "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
        "completion_tokens": response.usage.completion_tokens if response.usage else None,
    }


WRITING_QUALITY_SYSTEM_PROMPT = """Rate the writing quality of this CV on clarity,
grammar, and professionalism only — NOT on the candidate's actual skills or
experience level. A junior candidate with clean, clear writing should score
as well as a senior candidate with clean writing.

Return ONLY valid JSON, no other text:
{
  "match_score": <int 0-100, where 100 = very clear/professional writing>,
  "explanation": "<1 sentence reason>"
}
"""


def writing_quality_signal(cv_text: str) -> dict:
    response = call_with_retry(
        model=MODEL,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": WRITING_QUALITY_SYSTEM_PROMPT},
            {"role": "user", "content": cv_text},
        ],
    )
    raw = response.choices[0].message.content
    if not raw:
        raise ValueError(f"Empty response from model. Full response object for debugging:\n{response}")
    raw = raw.strip()
    raw = re.sub(r"^```json|```$", "", raw, flags=re.MULTILINE).strip()
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model did not return valid JSON. Raw output was:\n{raw}") from e
    return {
        "match_score": parsed["match_score"],
        "explanation": parsed["explanation"],
        "source": "writing_quality",
    }


SAMPLE_JD = """Senior Data Analyst
We are looking for a Senior Data Analyst with 3+ years of experience in SQL
and Python. Experience with Tableau or similar BI tools required. Kubernetes
and cloud infrastructure experience (AWS/GCP) is a strong plus. Bachelor's
degree in a quantitative field required."""

SAMPLES = {
    "strong_match": """Jane Doe
5 years of experience as a Data Analyst at a mid-size fintech company.
Proficient in Python, SQL, and Tableau for dashboarding and reporting.
Built automated ETL pipelines and led ad-hoc analyses for the finance team.
B.Sc. in Statistics.
No experience with cloud infrastructure or container orchestration tools.""",

    "clear_mismatch": """John Smith
1 year of experience as a marketing coordinator. Managed social media
campaigns and created content calendars. Familiar with Excel and Canva.
B.A. in Communications.""",

    "negation_edge_case": """Alex Kim
4 years of experience as a Data Analyst. Strong in SQL and Python, built
several Tableau dashboards for the sales team. Explicitly has not worked
with Kubernetes or any cloud infrastructure — team used on-prem servers
exclusively. B.Sc. in Applied Mathematics.""",

    "poorly_written_strong_skills": """maria garcia
i work with data for 5 year at company doing analyst job. i use python sql
and also tableu alot for make dashboard and reportings for teem. i also
did build pipeline automatic for etl and help finance teem with analysis
when they need. i have degre in statistic from university.
dont have expereince with cloud or kubernentes but willing to lern quick""",

    "hallucination_test": """Sam Rivera
5 years of experience as a Data Analyst. Strong in Python for data
processing and statistical modeling. Built several internal dashboards
using a custom in-house tool. B.Sc. in Statistics.""",

    "synonym_test": """Priya Nair
4 years turning raw data into decisions. Owned the reporting pipeline
end-to-end — writing complex queries against our warehouse, scripting
automation in Python, and building exec-facing dashboards in Tableau.
B.Sc. in Statistics.""",
}


def check_consistency(cv_text: str, jd_text: str, n_runs: int = 3) -> dict:
    scores = [llm_as_judge(cv_text, jd_text)["match_score"] for _ in range(n_runs)]
    return {
        "scores": scores,
        "min": min(scores),
        "max": max(scores),
        "spread": max(scores) - min(scores),
    }


def load_pairs_from_csv(path: str) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


if __name__ == "__main__":
    for label, cv_text in SAMPLES.items():
        print(f"=== {label} ===")

        judge_result = llm_as_judge(cv_text, SAMPLE_JD)
        print("LLM-as-Judge result:")
        print(json.dumps(judge_result, indent=2))

        wq_result = writing_quality_signal(cv_text)
        print("Writing-quality result:")
        print(json.dumps(wq_result, indent=2))
        print()