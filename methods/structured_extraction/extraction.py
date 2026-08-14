"""
Extraction step for method 2b (Structured Extraction + Rules).

Takes raw CV or JD text -> returns an ExtractedProfile (see schema.py).
Uses the free LLM client (llm_client.py, defaults to Groq -- no cost, but
requires a free GROQ_API_KEY) with a strict "return JSON only" instruction.

Usage:
    from extraction import extract_profile
    profile = extract_profile(cv_text, source_type="cv")
"""

import json
from schema import ExtractedProfile
from llm_client import call_llm

EXTRACTION_SYSTEM_PROMPT = """You extract structured information from a CV or a job description.

Return ONLY valid JSON matching exactly this schema, nothing else -- no preamble, no markdown fences:

{
  "source_type": "cv" | "jd",
  "skills": [{"name": "<lowercase normalized skill>", "raw_mention": "<original text or null>"}],
  "experience": {
    "total_years": <float or null>,
    "roles": ["<role title>", ...],
    "seniority": "junior" | "mid" | "senior" | "unspecified"
  },
  "education": {
    "highest_degree": "<e.g. BSc, MSc, PhD, or null>",
    "field_of_study": "<field, or null>"
  }
}

Rules:
- Normalize skill names to lowercase, singular/canonical form (e.g. "Python programming" -> "python").
- For a job description, "skills" means REQUIRED or PREFERRED skills, and "experience.total_years"
  means the years of experience REQUIRED (not the years the writer of the JD has).
- If information is not present, use null (for scalars) or an empty list (for lists). Do not guess.
- Do not include soft skills like "communication" or "teamwork" -- only concrete technical/domain skills.
"""


def extract_profile(text: str, source_type: str) -> ExtractedProfile:
    """
    source_type: "cv" or "jd"
    """
    user_prompt = f"Document type: {source_type}\n\nText:\n{text}"

    raw = call_llm(
        system_prompt=EXTRACTION_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        json_mode=True,
    ).strip()

    # Defensive cleanup in case the model wraps in markdown fences despite instructions
    if raw.startswith("```"):
        raw = raw.strip("`")
        raw = raw.split("\n", 1)[1] if "\n" in raw else raw
        raw = raw.rsplit("```", 1)[0]

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"LLM did not return valid JSON. Raw output was:\n{raw}\n\nError: {e}"
        )

    data["source_type"] = source_type  # pin it — don't trust the model's echo
    return ExtractedProfile(**data)


if __name__ == "__main__":
    # Quick manual test -- run this file directly to sanity check the prompt.
    # Make sure Ollama is running first (see llm_client.py docstring for setup).
    sample_cv = """
    Jane Doe. 4 years of experience as a Data Analyst and Junior Data Scientist.
    Skills: Python, SQL, pandas, Tableau, basic machine learning.
    Education: BSc in Statistics, University of Heidelberg.
    """
    profile = extract_profile(sample_cv, source_type="cv")
    print(profile.model_dump_json(indent=2))
