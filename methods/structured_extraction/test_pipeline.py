"""
Integration test for the 2b pipeline (extraction -> rule scoring -> gap explanation)
using REAL CVs from Resume.csv.

We don't have a job-description dataset yet, so this uses a couple of hand-written
sample JDs matching your CSV's categories -- good enough to sanity-check the
pipeline on real, messy CV text before the team finalizes a real JD source.

Run:
    python test_pipeline.py
"""

from data_loader import load_resumes_csv, load_resumes_pdf
from extraction import extract_profile
from rule_scorer import score_match
from gap_explainer import explain_template

# No path needed — data_loader.py resolves Resume/Resume.csv and data/data/ relative to the repo root.
# Set USE_PDF=True to load from the PDF folder instead of the CSV.
USE_PDF = False

# --- hand-written sample JDs, one per category you want to test ---
SAMPLE_JDS = {
    "BANKING": """
        We are hiring a Banking Operations Associate with 3+ years of experience
        in retail or commercial banking. Required skills: risk assessment, KYC
        compliance, customer relationship management, Microsoft Excel, financial
        reporting. Education: BSc or higher in Finance, Economics, or a related field.
    """,
    "IT": """
        We are looking for a Software Engineer with 4+ years of experience.
        Required skills: Python, SQL, cloud infrastructure (AWS or Azure), CI/CD,
        REST APIs. Education: BSc in Computer Science or a related field.
    """,
}


def run_test(category: str, n_cvs: int = 3):
    print(f"\n{'='*70}\nTesting category: {category}\n{'='*70}")

    if category not in SAMPLE_JDS:
        print(f"No sample JD defined for '{category}'. Add one to SAMPLE_JDS.")
        return

    records = (load_resumes_pdf(category=category, n=n_cvs) if USE_PDF
               else load_resumes_csv(category=category, n=n_cvs))
    if not records:
        print(f"No CVs found for category '{category}' -- check the Category "
              f"value in your CSV matches exactly (case-insensitive is handled).")
        return

    # Extract the JD once, reuse for all CVs in this category
    print("Extracting JD profile...")
    jd_profile = extract_profile(SAMPLE_JDS[category], source_type="jd")
    print(jd_profile.model_dump_json(indent=2))

    for record in records:
        print(f"\n--- CV {record.id} ---")
        try:
            cv_profile = extract_profile(record.text, source_type="cv")
        except ValueError as e:
            print(f"[EXTRACTION FAILED] {e}")
            continue

        # Debug: show what skills were actually extracted from the CV,
        # so we can tell "extraction found nothing" apart from
        # "extraction found skills, they just don't string-match the JD's wording"
        cv_skill_names = [s.name for s in cv_profile.skills]
        print(f"Extracted CV skills ({len(cv_skill_names)}): {cv_skill_names}")

        result = score_match(cv_profile, jd_profile)
        result.explanation = explain_template(result.structured_diff)

        print(f"Match score: {result.match_score}")
        print(f"Sub-scores: {result.sub_scores.model_dump()}")
        print(f"Explanation: {result.explanation}")


if __name__ == "__main__":
    # Test as many categories as you like -- just make sure SAMPLE_JDS has an entry for each
    run_test("BANKING", n_cvs=3)
    # run_test("IT", n_cvs=3)
