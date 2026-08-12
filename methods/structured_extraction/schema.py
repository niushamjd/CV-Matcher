"""
Shared schemas for MatchLens.

IMPORTANT: These are the contracts the whole team agreed on in the sync.
Every matching method (embedding similarity / structured extraction / LLM-as-judge)
must return a MatchResult. This is what lets Ipek's evaluation harness plug in
all three methods without custom code per method.

If you (Ertugrul) need to change something here, ping the group first --
Ipek's harness and Niyousha's judge prompt both depend on this shape.
"""

from __future__ import annotations
from typing import Optional, Literal
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# 1. Structured profile schema (used by YOUR extraction step, method 2b)
#    Also referenced by Niyousha's LLM-as-judge prompt for consistency.
# ---------------------------------------------------------------------------

class ExtractedSkill(BaseModel):
    name: str                      # normalized, lowercase, e.g. "python"
    raw_mention: Optional[str] = None  # original text span, for traceability


class ExtractedExperience(BaseModel):
    total_years: Optional[float] = None
    roles: list[str] = Field(default_factory=list)   # e.g. ["Software Engineer", "Data Analyst"]
    seniority: Optional[Literal["junior", "mid", "senior", "unspecified"]] = "unspecified"


class ExtractedEducation(BaseModel):
    highest_degree: Optional[str] = None     # e.g. "MSc", "BSc", "PhD"
    field_of_study: Optional[str] = None      # e.g. "Computer Science"


class ExtractedProfile(BaseModel):
    """Output of the extraction step, for EITHER a CV or a JD."""
    source_type: Literal["cv", "jd"]
    skills: list[ExtractedSkill] = Field(default_factory=list)
    experience: ExtractedExperience = Field(default_factory=ExtractedExperience)
    education: ExtractedEducation = Field(default_factory=ExtractedEducation)


# ---------------------------------------------------------------------------
# 2. Structured diff (matched/missing skills, experience gap)
#    This feeds YOUR Gap Explainer in week 2.
# ---------------------------------------------------------------------------

class StructuredDiff(BaseModel):
    matched_skills: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)
    extra_skills: list[str] = Field(default_factory=list)  # candidate has, JD doesn't ask for
    years_required: Optional[float] = None
    years_candidate: Optional[float] = None
    years_gap: Optional[float] = None            # negative = candidate exceeds requirement
    education_match: Optional[bool] = None        # overall: degree level AND field combined
    degree_level_match: Optional[bool] = None      # e.g. candidate has >= required degree level
    field_of_study_match: Optional[bool] = None    # e.g. "Statistics" vs "Statistics or related field"


# ---------------------------------------------------------------------------
# 3. THE shared output contract every method must return.
#    sub_scores are optional but recommended (per the summary: "sub-scores
#    to show job seeker their strong and weak areas").
# ---------------------------------------------------------------------------

class SubScores(BaseModel):
    skills_score: Optional[float] = None       # 0-100
    experience_score: Optional[float] = None   # 0-100
    education_score: Optional[float] = None    # 0-100


class MatchResult(BaseModel):
    method: Literal["embedding_similarity", "structured_extraction", "llm_judge"]
    match_score: float = Field(..., ge=0, le=100)
    sub_scores: Optional[SubScores] = None
    explanation: Optional[str] = None
    structured_diff: Optional[StructuredDiff] = None   # populated by 2b, optional for others
    latency_ms: Optional[float] = None
    cost_usd: Optional[float] = None
