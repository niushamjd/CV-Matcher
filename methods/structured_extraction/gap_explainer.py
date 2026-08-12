"""
Gap Explainer -- Week 2.

Takes the StructuredDiff produced by rule_scorer.py and turns it into a short,
human-readable explanation of strengths/gaps for the candidate.

Two options, pick based on time budget:
  (A) Template-based (no LLM call, instant, fully deterministic) -- good fallback.
  (B) Lightweight LLM call for more natural phrasing -- nicer output, adds latency/cost.

Start with (A) since it's free and instant; swap to (B) if time allows and you
want the explanation to read less robotically. Either way, output goes into
MatchResult.explanation so the shared schema is unaffected.
"""

from schema import StructuredDiff


def explain_template(diff: StructuredDiff) -> str:
    """Option A: deterministic template, no LLM call needed."""
    parts = []

    if diff.matched_skills:
        parts.append(f"Matches required skills: {', '.join(diff.matched_skills)}.")
    if diff.missing_skills:
        parts.append(f"Missing required skills: {', '.join(diff.missing_skills)}.")

    if diff.years_gap is not None:
        if diff.years_gap >= 0:
            parts.append(
                f"Meets experience requirement ({diff.years_candidate} yrs vs "
                f"{diff.years_required} yrs required)."
            )
        else:
            parts.append(
                f"Below required experience by {abs(diff.years_gap):.1f} years "
                f"({diff.years_candidate} yrs vs {diff.years_required} yrs required)."
            )

    if diff.education_match is not None:
        edu_parts = []
        if diff.degree_level_match is not None:
            edu_parts.append(
                "degree level meets requirement" if diff.degree_level_match
                else "degree level below requirement"
            )
        if diff.field_of_study_match is not None:
            edu_parts.append(
                "field of study matches" if diff.field_of_study_match
                else "field of study does not match"
            )
        if edu_parts:
            parts.append("Education: " + "; ".join(edu_parts) + ".")

    return " ".join(parts) if parts else "Not enough information to generate an explanation."


# --- Option B (LLM-based): natural, encouraging explanation grounded in the diff ---
def explain_llm(diff: StructuredDiff, writing_quality_score: float = None) -> str:
    """
    Turns the StructuredDiff into a 2-3 sentence human-readable explanation.
    Grounded in the diff — not free-form hallucination.
    Optionally mentions writing quality if the score is notably high or low.
    """
    from llm_client import call_llm
    import re

    wq_note = ""
    if writing_quality_score is not None:
        if writing_quality_score >= 80:
            wq_note = f" The CV is well-written and professional (writing quality: {writing_quality_score}/100)."
        elif writing_quality_score <= 40:
            wq_note = f" Note: the CV writing quality is weak ({writing_quality_score}/100), which may affect readability."

    prompt = (
        f"Turn this structured CV–JD comparison into a 2-3 sentence, honest and encouraging "
        f"explanation for the candidate. Be specific — name the matched and missing skills. "
        f"Do not invent information beyond what is in the diff.\n\n"
        f"Structured diff:\n{diff.model_dump_json(indent=2)}"
        f"{wq_note}"
    )

    raw = call_llm(
        system_prompt="You write short, honest, encouraging candidate feedback grounded strictly in the provided data.",
        user_prompt=prompt,
        json_mode=False,
    ).strip()

    # Strip any accidental markdown
    raw = re.sub(r"^#+\s*", "", raw, flags=re.MULTILINE).strip()
    return raw


if __name__ == "__main__":
    from extraction import extract_profile
    from rule_scorer import score_match

    cv_text = """
    Jane Doe. 4 years of experience as a Data Analyst and Junior Data Scientist.
    Skills: Python, SQL, pandas, Tableau, basic machine learning.
    Education: BSc in Statistics, University of Heidelberg.
    """
    jd_text = """
    We are looking for a Data Scientist with 5+ years of experience.
    Required skills: Python, SQL, machine learning, deep learning, AWS.
    Education: MSc in Computer Science, Statistics, or related field.
    """

    cv_profile = extract_profile(cv_text, source_type="cv")
    jd_profile = extract_profile(jd_text, source_type="jd")
    result = score_match(cv_profile, jd_profile)
    result.explanation = explain_template(result.structured_diff)
    print(result.model_dump_json(indent=2))
