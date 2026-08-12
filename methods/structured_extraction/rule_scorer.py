"""
Rule-based scorer for method 2b.

DELIBERATELY SIMPLE -- the instructor flagged this piece not to overengineer.
This is a baseline, not the star of the show. Plain overlap + threshold logic only.

Takes two ExtractedProfile objects (cv_profile, jd_profile) -> returns a MatchResult
with a structured_diff, following the shared schema.
"""

from typing import Optional
from schema import ExtractedProfile, MatchResult, SubScores, StructuredDiff

# Common aliases that the LLM may normalize differently across CV vs JD.
# Maps every variant to a single canonical form.
_SKILL_ALIASES: dict[str, str] = {
    "ms excel": "microsoft excel",
    "excel": "microsoft excel",
    "ms office": "microsoft office",
    "ms word": "microsoft word",
    "ml": "machine learning",
    "dl": "deep learning",
    "nlp": "natural language processing",
    "cv": "computer vision",
    "oop": "object-oriented programming",
    "js": "javascript",
    "ts": "typescript",
    "postgres": "postgresql",
    "k8s": "kubernetes",
    "crm": "customer relationship management",
    "kyc": "kyc compliance",
}


def _normalize(skill: str) -> str:
    s = skill.lower().strip()
    return _SKILL_ALIASES.get(s, s)


def score_match(cv_profile: ExtractedProfile, jd_profile: ExtractedProfile) -> MatchResult:
    # --- Skills overlap (simple set intersection) ---
    cv_skills = {_normalize(s.name) for s in cv_profile.skills}
    jd_skills = {_normalize(s.name) for s in jd_profile.skills}

    matched = sorted(cv_skills & jd_skills)
    missing = sorted(jd_skills - cv_skills)
    extra = sorted(cv_skills - jd_skills)

    skills_score = 100.0 if not jd_skills else 100.0 * len(matched) / len(jd_skills)

    # --- Experience gap (simple numeric threshold) ---
    years_required = jd_profile.experience.total_years
    years_candidate = cv_profile.experience.total_years
    years_gap = None
    if years_required is not None and years_candidate is not None:
        years_gap = years_candidate - years_required
        if years_gap >= 0:
            experience_score = 100.0
        else:
            # linear penalty: missing 1 full year below requirement = -25 points, floored at 0
            experience_score = max(0.0, 100.0 + years_gap * 25)
    else:
        experience_score = None  # not enough info to score

    # --- Education: two simple components -- degree level, and field of study ---
    # Still intentionally simple (no ML/embedding-based field similarity), but now
    # actually checks the field, not just the degree string.
    DEGREE_RANK = {
        "bsc": 1, "ba": 1, "bachelor": 1, "bachelors": 1,
        "msc": 2, "ma": 2, "master": 2, "masters": 2,
        "phd": 3, "doctorate": 3,
    }

    def _degree_rank(degree_str: Optional[str]) -> Optional[int]:
        if not degree_str:
            return None
        d = degree_str.lower().strip()
        for key, rank in DEGREE_RANK.items():
            if key in d:
                return rank
        return None  # unrecognized degree string -- treat as unknown, not a mismatch

    def _field_match(cv_field: Optional[str], jd_field: Optional[str]) -> Optional[bool]:
        if not jd_field:
            return None  # JD didn't specify a field -- nothing to check
        if not cv_field:
            return False  # JD wants a field, CV has none listed
        # simple token overlap: e.g. "statistics" in "computer science, statistics, or related field"
        cv_tokens = set(cv_field.lower().replace(",", " ").split())
        jd_tokens = set(jd_field.lower().replace(",", " ").split())
        # also check substring containment for multi-word fields
        substring_hit = cv_field.lower() in jd_field.lower() or jd_field.lower() in cv_field.lower()
        return substring_hit or bool(cv_tokens & jd_tokens - {"or", "related", "field", "and"})

    degree_level_match = None
    jd_rank = _degree_rank(jd_profile.education.highest_degree)
    cv_rank = _degree_rank(cv_profile.education.highest_degree)
    if jd_rank is not None:
        degree_level_match = (cv_rank is not None) and (cv_rank >= jd_rank)

    field_of_study_match = _field_match(
        cv_profile.education.field_of_study, jd_profile.education.field_of_study
    )

    # Combine: overall education_match is True only if both required components are satisfied
    # (components that weren't specified in the JD are treated as "not required", i.e. ignored)
    components = [c for c in [degree_level_match, field_of_study_match] if c is not None]
    education_match = all(components) if components else None
    education_score = None
    if components:
        education_score = 100.0 * sum(components) / len(components)

    # --- Overall score: simple average of available sub-scores ---
    available_scores = [s for s in [skills_score, experience_score, education_score] if s is not None]
    overall_score = sum(available_scores) / len(available_scores) if available_scores else 0.0

    diff = StructuredDiff(
        matched_skills=matched,
        missing_skills=missing,
        extra_skills=extra,
        years_required=years_required,
        years_candidate=years_candidate,
        years_gap=years_gap,
        education_match=education_match,
        degree_level_match=degree_level_match,
        field_of_study_match=field_of_study_match,
    )

    return MatchResult(
        method="structured_extraction",
        match_score=round(overall_score, 1),
        sub_scores=SubScores(
            skills_score=round(skills_score, 1) if skills_score is not None else None,
            experience_score=round(experience_score, 1) if experience_score is not None else None,
            education_score=education_score,
        ),
        explanation=None,  # filled in by Gap Explainer, week 2
        structured_diff=diff,
    )


if __name__ == "__main__":
    from extraction import extract_profile

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
    print(result.model_dump_json(indent=2))
