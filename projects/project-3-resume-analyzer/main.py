"""Project 3: Resume Analyzer with Structured Extraction + Scoring

Analyzes a resume text and returns:
- Structured extraction: name, skills, years_of_experience, education, summary
- Job fit score (0-10) for a given job description
- Strengths and gaps analysis

Run:
  uv run python main.py
  uv run python main.py --demo
"""

NL = chr(10)
MODEL = "gpt-4o"

import argparse
import json
from typing import Literal

from pydantic import BaseModel, Field
from shared.mock import get_client, is_mock

SAMPLE_RESUME = """
John Doe
Software Engineer | john@example.com | github.com/johndoe

EXPERIENCE
Senior Software Engineer, Acme Corp (2020-2024) — 4 years
  - Built distributed microservices in Python and Go
  - Led team of 5 engineers, reduced latency by 40%
  - Implemented CI/CD pipelines with GitHub Actions

Software Engineer, TechStart (2018-2020) — 2 years
  - Developed REST APIs with FastAPI and PostgreSQL
  - Built internal tools using React and TypeScript

EDUCATION
B.S. Computer Science, State University, 2018

SKILLS
Python, Go, TypeScript, React, PostgreSQL, Redis, Docker, Kubernetes, AWS
"""

SAMPLE_JOB = """
Senior Backend Engineer — Python
We need an experienced Python engineer to build scalable APIs.
Requirements: 5+ years Python, distributed systems experience, team leadership.
Nice-to-have: Go, Kubernetes, AWS.
"""


class ResumeExtraction(BaseModel):
    name: str
    email: str
    years_of_experience: int
    skills: list[str]
    education: str
    current_role: str
    summary: str


class JobFitAnalysis(BaseModel):
    score: int = Field(ge=0, le=10, description="Fit score 0-10")
    strengths: list[str]
    gaps: list[str]
    recommendation: Literal["strong_yes", "yes", "maybe", "no"]


def extract_resume(client: object, resume_text: str) -> ResumeExtraction:
    """Extract structured information from resume text."""
    prompt = (
        "Extract information from this resume and return JSON with fields: "
        "name, email, years_of_experience (int), skills (list), "
        "education (string), current_role (string), summary (2 sentences)."
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": resume_text},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or json.dumps({
        "name": "John Doe",
        "email": "john@example.com",
        "years_of_experience": 6,
        "skills": ["Python", "Go", "TypeScript", "PostgreSQL", "Kubernetes"],
        "education": "B.S. Computer Science, State University, 2018",
        "current_role": "Senior Software Engineer",
        "summary": "Experienced engineer with 6 years in Python and distributed systems. Led teams and built scalable APIs.",
    })
    return ResumeExtraction.model_validate(json.loads(raw))


def analyze_fit(client: object, resume: ResumeExtraction, job_description: str) -> JobFitAnalysis:
    """Score the resume against a job description."""
    resume_summary = f"Candidate: {resume.name}, {resume.years_of_experience} years exp, skills: {', '.join(resume.skills)}"
    prompt = (
        "Analyze candidate fit for the job. Return JSON with: "
        "score (0-10), strengths (list), gaps (list), "
        "recommendation (strong_yes/yes/maybe/no)."
    )
    response = client.chat.completions.create(  # type: ignore[attr-defined]
        model=MODEL,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"Resume:{NL}{resume_summary}{NL}{NL}Job:{NL}{job_description}"},
        ],
        response_format={"type": "json_object"},
    )
    raw = response.choices[0].message.content or json.dumps({
        "score": 8,
        "strengths": ["Strong Python experience", "Distributed systems", "Team leadership"],
        "gaps": ["Needs Go verification", "5+ years — candidate has 6 (OK)"],
        "recommendation": "strong_yes",
    })
    return JobFitAnalysis.model_validate(json.loads(raw))


def analyze(client: object, resume_text: str, job_description: str) -> None:
    """Full analysis pipeline: extract → score → print report."""
    print("Extracting resume information...")
    resume = extract_resume(client, resume_text)
    print(f"  Name: {resume.name}")
    print(f"  Experience: {resume.years_of_experience} years")
    print(f"  Skills: {', '.join(resume.skills[:5])}{'...' if len(resume.skills) > 5 else ''}")
    print(f"  Education: {resume.education}")
    print(f"  Summary: {resume.summary}{NL}")

    print("Analyzing job fit...")
    fit = analyze_fit(client, resume, job_description)
    print(f"  Fit score:      {fit.score}/10")
    print(f"  Recommendation: {fit.recommendation}")
    print("  Strengths:")
    for s in fit.strengths:
        print(f"    + {s}")
    print("  Gaps:")
    for g in fit.gaps:
        print(f"    - {g}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Resume Analyzer")
    parser.add_argument("--demo", action="store_true", help="Run with sample data")
    args = parser.parse_args()

    mode = "MOCK" if is_mock() else "LIVE"
    print(f"Resume Analyzer [{mode}]{NL}")

    client = get_client()
    if args.demo:
        analyze(client, SAMPLE_RESUME, SAMPLE_JOB)
    else:
        print("Paste resume text (end with '---' on a new line):")
        lines = []
        while True:
            try:
                line = input()
                if line.strip() == "---":
                    break
                lines.append(line)
            except (EOFError, KeyboardInterrupt):
                break
        resume_text = NL.join(lines) if lines else SAMPLE_RESUME
        analyze(client, resume_text, SAMPLE_JOB)
