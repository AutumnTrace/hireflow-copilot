"""Turn a HireFlow snapshot into retrievable documents.

Each document is split into short text blocks. Claude cites whole blocks, so smaller blocks give
tighter citations (one block per cover letter, per stage note, per interview).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from hireflow_copilot.hireflow.models import Application, Snapshot

Kind = Literal["job", "candidate", "application"]


@dataclass(frozen=True)
class Document:
    id: str
    kind: Kind
    title: str
    source: str
    blocks: tuple[str, ...]
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n".join((self.title, *self.blocks))


def _salary(snapshot: Snapshot, job_id: str) -> str:
    job = snapshot.job(job_id)
    if job is None or job.salary is None:
        return "salary not listed"
    return f"{job.salary.min:,}-{job.salary.max:,} {job.salary.currency}"


def _application_doc(snapshot: Snapshot, app: Application) -> Document | None:
    job, candidate = snapshot.job(app.job_posting_id), snapshot.candidate(app.candidate_id)
    if job is None or candidate is None:
        return None

    blocks = [
        f"{candidate.full_name} applied to {job.title} at {job.company_name} on {app.applied_at:%Y-%m-%d}. "
        f"Current stage: {app.stage}."
    ]
    if app.skill_match:
        match = app.skill_match
        blocks.append(
            f"Skill match {match.score:.0%}: has {', '.join(match.matched_skills) or 'none of the required skills'}"
            + (f"; missing {', '.join(match.missing_skills)}." if match.missing_skills else ".")
        )
    if app.cover_letter:
        blocks.append(f"Cover letter: {app.cover_letter}")
    for step in app.history:
        if step.note:
            blocks.append(f"Moved {step.from_ or 'NEW'} -> {step.to} on {step.changed_at:%Y-%m-%d}: {step.note}")
    for interview in app.interviews:
        line = (
            f"{interview.kind.replace('_', ' ').title()} interview with {interview.interviewer} "
            f"on {interview.scheduled_at:%Y-%m-%d}"
        )
        if interview.feedback:
            line += f", rated {interview.rating}/5: {interview.feedback}"
        else:
            line += " (no feedback yet)."
        blocks.append(line)

    return Document(
        id=f"application:{app.id}",
        kind="application",
        title=f"Application: {candidate.full_name} for {job.title}",
        source=f"hireflow://applications/{app.id}",
        blocks=tuple(blocks),
        metadata={"application_id": app.id, "job_id": job.id, "candidate_id": candidate.id, "stage": app.stage},
    )


def build_documents(snapshot: Snapshot) -> list[Document]:
    docs: list[Document] = []

    for job in snapshot.jobs:
        where = job.location or "no fixed location"
        docs.append(
            Document(
                id=f"job:{job.id}",
                kind="job",
                title=f"Job: {job.title} at {job.company_name}",
                source=f"hireflow://jobs/{job.id}",
                blocks=(
                    f"{job.title} ({job.status}) at {job.company_name}. {job.work_mode.title()}, "
                    f"{job.employment_type.replace('_', ' ').lower()}, {where}, {_salary(snapshot, job.id)}.",
                    f"Required skills: {', '.join(job.required_skills) or 'none listed'}.",
                    job.description,
                ),
                metadata={"job_id": job.id, "status": job.status},
            )
        )

    for candidate in snapshot.candidates:
        docs.append(
            Document(
                id=f"candidate:{candidate.id}",
                kind="candidate",
                title=f"Candidate: {candidate.full_name}",
                source=f"hireflow://candidates/{candidate.id}",
                blocks=(
                    f"{candidate.full_name}: {candidate.headline or 'no headline'}. "
                    f"{candidate.years_of_experience} years of experience.",
                    f"Skills: {', '.join(candidate.skills) or 'none listed'}.",
                ),
                metadata={"candidate_id": candidate.id},
            )
        )

    for app in snapshot.applications:
        if doc := _application_doc(snapshot, app):
            docs.append(doc)

    return docs
