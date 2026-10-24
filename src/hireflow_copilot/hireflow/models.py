"""Typed view of the HireFlow domain, mirroring the GraphQL schema of AutumnTrace/hireflow."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field
from pydantic.alias_generators import to_camel


class Stage(StrEnum):
    APPLIED = "APPLIED"
    SCREENING = "SCREENING"
    INTERVIEW = "INTERVIEW"
    OFFER = "OFFER"
    HIRED = "HIRED"
    REJECTED = "REJECTED"
    WITHDRAWN = "WITHDRAWN"


class InterviewKind(StrEnum):
    PHONE_SCREEN = "PHONE_SCREEN"
    TECHNICAL = "TECHNICAL"
    SYSTEM_DESIGN = "SYSTEM_DESIGN"
    BEHAVIORAL = "BEHAVIORAL"
    ONSITE = "ONSITE"


# Same table as HireFlow.Domain.Applications.ApplicationStageRules, so fixture mode and the
# real API reject the same moves.
TRANSITIONS: dict[Stage, tuple[Stage, ...]] = {
    Stage.APPLIED: (Stage.SCREENING, Stage.REJECTED, Stage.WITHDRAWN),
    Stage.SCREENING: (Stage.INTERVIEW, Stage.REJECTED, Stage.WITHDRAWN),
    Stage.INTERVIEW: (Stage.OFFER, Stage.REJECTED, Stage.WITHDRAWN),
    Stage.OFFER: (Stage.HIRED, Stage.REJECTED, Stage.WITHDRAWN),
    Stage.HIRED: (),
    Stage.REJECTED: (),
    Stage.WITHDRAWN: (),
}


class Model(BaseModel):
    """GraphQL returns camelCase; Python code uses snake_case."""

    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class Salary(Model):
    min: int
    max: int
    currency: str


class Company(Model):
    id: str
    name: str
    industry: str | None = None
    location: str | None = None


class StageTransition(Model):
    from_: Stage | None = Field(default=None, alias="from")
    to: Stage
    note: str | None = None
    changed_at: datetime


class Interview(Model):
    id: str
    application_id: str
    kind: InterviewKind
    scheduled_at: datetime
    duration_minutes: int
    interviewer: str
    rating: int | None = None
    feedback: str | None = None


class SkillMatch(Model):
    score: float
    matched_skills: list[str] = []
    missing_skills: list[str] = []

    @classmethod
    def evaluate(cls, required: list[str], offered: list[str]) -> SkillMatch:
        """Port of HireFlow.Domain.Jobs.SkillMatch.Evaluate: case-insensitive share of required skills."""
        have = {s.lower() for s in offered}
        unique = list(dict.fromkeys(required, None))
        matched = [s for s in unique if s.lower() in have]
        missing = [s for s in unique if s.lower() not in have]
        score = 1.0 if not unique else round(len(matched) / len(unique), 3)
        return cls(score=score, matched_skills=matched, missing_skills=missing)


class Application(Model):
    id: str
    job_posting_id: str
    candidate_id: str
    stage: Stage
    cover_letter: str | None = None
    applied_at: datetime
    allowed_next_stages: list[Stage] = []
    history: list[StageTransition] = []
    interviews: list[Interview] = []
    skill_match: SkillMatch | None = None


class Job(Model):
    id: str
    company_id: str
    company_name: str = ""
    title: str
    description: str
    location: str | None = None
    work_mode: str
    employment_type: str
    salary: Salary | None = None
    required_skills: list[str] = []
    status: str


class Candidate(Model):
    id: str
    full_name: str
    email: str
    headline: str | None = None
    years_of_experience: int
    skills: list[str] = []


class Snapshot(Model):
    """Everything the copilot reads from HireFlow, fetched in one go."""

    companies: list[Company] = []
    jobs: list[Job] = []
    candidates: list[Candidate] = []
    applications: list[Application] = []

    def job(self, job_id: str) -> Job | None:
        return next((j for j in self.jobs if j.id == job_id), None)

    def candidate(self, candidate_id: str) -> Candidate | None:
        return next((c for c in self.candidates if c.id == candidate_id), None)

    def application(self, application_id: str) -> Application | None:
        return next((a for a in self.applications if a.id == application_id), None)


class HireFlowError(Exception):
    """A business error returned by HireFlow (for example ``InvalidStageTransitionError``)."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
