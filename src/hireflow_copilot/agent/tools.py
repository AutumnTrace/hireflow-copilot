"""Tools the agent can call. Reads hit the current HireFlow snapshot; writes are only *proposed*.

A proposed write is stored as a pending action and returned to the caller, who confirms it with
``POST /v1/actions/{id}/confirm``. The model can never change HireFlow on its own.
"""

from __future__ import annotations

import json
import unicodedata
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from hireflow_copilot.hireflow.models import TRANSITIONS, InterviewKind, Snapshot, Stage
from hireflow_copilot.knowledge import KnowledgeBase
from hireflow_copilot.llm.citations import search_result_block
from hireflow_copilot.rag.documents import Document


class SearchInput(BaseModel):
    query: str = Field(description="What to look for, in natural language or keywords.")
    kind: Literal["job", "candidate", "application"] | None = Field(
        default=None, description="Limit results to one document type."
    )


class ListJobsInput(BaseModel):
    status: Literal["OPEN", "DRAFT", "CLOSED"] | None = Field(default="OPEN", description="Filter by job status.")


class PipelineInput(BaseModel):
    job_id: str = Field(description="Job posting ID (UUID) from list_jobs or search results.")


class CandidateInput(BaseModel):
    name_or_email: str = Field(description="Full name, part of a name, or email address.")


class UpcomingInput(BaseModel):
    days: int = Field(default=7, ge=1, le=60, description="How many days ahead to look.")
    interviewer: str | None = Field(default=None, description="Only this interviewer's interviews.")


class MoveInput(BaseModel):
    application_id: str = Field(description="Application ID (UUID).")
    stage: Literal["SCREENING", "INTERVIEW", "OFFER", "HIRED", "REJECTED", "WITHDRAWN"]
    note: str | None = Field(default=None, description="Short reason, stored in the stage history.")


class ScheduleInput(BaseModel):
    application_id: str = Field(description="Application ID (UUID).")
    kind: Literal["PHONE_SCREEN", "TECHNICAL", "SYSTEM_DESIGN", "BEHAVIORAL", "ONSITE"]
    scheduled_at: datetime = Field(description="ISO 8601 date-time in UTC, e.g. 2026-11-03T10:00:00Z.")
    duration_minutes: int = Field(ge=15, le=480)
    interviewer: str


@dataclass
class PendingAction:
    id: str
    kind: Literal["move_application", "schedule_interview"]
    summary: str
    params: dict[str, Any]
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    status: Literal["pending", "done", "failed"] = "pending"
    result: str | None = None


class ActionStore:
    """Pending writes awaiting a human. In memory, expires after an hour."""

    ttl = timedelta(hours=1)

    def __init__(self) -> None:
        self._actions: dict[str, PendingAction] = {}

    def add(
        self, kind: Literal["move_application", "schedule_interview"], summary: str, params: dict[str, Any]
    ) -> PendingAction:
        now = datetime.now(UTC)
        self._actions = {k: a for k, a in self._actions.items() if now - a.created_at < self.ttl}
        action = PendingAction(id=uuid.uuid4().hex[:12], kind=kind, summary=summary, params=params)
        self._actions[action.id] = action
        return action

    def get(self, action_id: str) -> PendingAction | None:
        action = self._actions.get(action_id)
        if action and datetime.now(UTC) - action.created_at >= self.ttl:
            return None
        return action


ToolResult = str | list[dict[str, Any]]


@dataclass
class ToolContext:
    kb: KnowledgeBase
    actions: ActionStore
    top_k: int
    # Every search_result block sent to the model, in order. Citation indices point into this list.
    sent_documents: list[Document] = field(default_factory=list)
    proposed: list[PendingAction] = field(default_factory=list)

    @property
    def snapshot(self) -> Snapshot:
        return self.kb.snapshot


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _fold(text: str) -> str:
    """Case- and accent-insensitive form, so "stefan turcanu" finds "Ștefan Țurcanu"."""
    decomposed = unicodedata.normalize("NFKD", text.casefold())
    return "".join(ch for ch in decomposed if not unicodedata.combining(ch))


async def search_hireflow(ctx: ToolContext, args: SearchInput) -> ToolResult:
    hits = ctx.kb.index.search(args.query, k=ctx.top_k, kinds={args.kind} if args.kind else None)
    if not hits:
        return "No results found."
    ctx.sent_documents.extend(h.document for h in hits)
    return [search_result_block(h.document) for h in hits]


async def list_jobs(ctx: ToolContext, args: ListJobsInput) -> ToolResult:
    snap = ctx.snapshot
    jobs = [j for j in snap.jobs if args.status is None or j.status == args.status]
    return _json(
        [
            {
                "job_id": j.id,
                "title": j.title,
                "company": j.company_name,
                "status": j.status,
                "work_mode": j.work_mode,
                "required_skills": j.required_skills,
                "applications": sum(1 for a in snap.applications if a.job_posting_id == j.id),
            }
            for j in jobs
        ]
    )


async def get_job_pipeline(ctx: ToolContext, args: PipelineInput) -> ToolResult:
    snap = ctx.snapshot
    job = snap.job(args.job_id)
    if job is None:
        raise ToolError(f"No job with id {args.job_id}. Call list_jobs to find valid ids.")
    rows = []
    for app in (a for a in snap.applications if a.job_posting_id == job.id):
        candidate = snap.candidate(app.candidate_id)
        rows.append(
            {
                "application_id": app.id,
                "candidate": candidate.full_name if candidate else app.candidate_id,
                "stage": app.stage,
                "allowed_next_stages": app.allowed_next_stages,
                "skill_match": app.skill_match.model_dump() if app.skill_match else None,
                "last_note": next((h.note for h in reversed(app.history) if h.note), None),
                "interviews": [
                    {"kind": i.kind, "at": i.scheduled_at, "interviewer": i.interviewer, "rating": i.rating}
                    for i in app.interviews
                ],
            }
        )
    return _json({"job": job.title, "company": job.company_name, "applications": rows})


async def get_candidate(ctx: ToolContext, args: CandidateInput) -> ToolResult:
    snap = ctx.snapshot
    needle = _fold(args.name_or_email.strip())
    words = needle.split()
    if not words:
        raise ToolError("Give a candidate's name or email address.")
    # Every word must appear somewhere in the name, in any order: "popescu ana" finds "Ana Popescu".
    matches = [c for c in snap.candidates if needle == _fold(c.email) or all(w in _fold(c.full_name) for w in words)]
    if not matches:
        raise ToolError(f"No candidate matches '{args.name_or_email}'.")
    out = []
    for c in matches[:5]:
        apps = [a for a in snap.applications if a.candidate_id == c.id]
        out.append(
            {
                **c.model_dump(exclude={"email"}),
                "applications": [
                    {
                        "application_id": a.id,
                        "job": (job.title if (job := snap.job(a.job_posting_id)) else a.job_posting_id),
                        "stage": a.stage,
                        "allowed_next_stages": a.allowed_next_stages,
                    }
                    for a in apps
                ],
            }
        )
    return _json(out)


async def upcoming_interviews(ctx: ToolContext, args: UpcomingInput) -> ToolResult:
    snap = ctx.snapshot
    now = datetime.now(UTC)
    until = now + timedelta(days=args.days)
    rows = []
    for app in snap.applications:
        for i in app.interviews:
            if now <= i.scheduled_at <= until and (args.interviewer is None or i.interviewer == args.interviewer):
                candidate, job = snap.candidate(app.candidate_id), snap.job(app.job_posting_id)
                rows.append(
                    {
                        "at": i.scheduled_at,
                        "kind": i.kind,
                        "minutes": i.duration_minutes,
                        "interviewer": i.interviewer,
                        "candidate": candidate.full_name if candidate else None,
                        "job": job.title if job else None,
                    }
                )
    return _json(sorted(rows, key=lambda r: r["at"]))


async def propose_stage_change(ctx: ToolContext, args: MoveInput) -> ToolResult:
    app = ctx.snapshot.application(args.application_id)
    if app is None:
        raise ToolError(f"No application with id {args.application_id}.")
    stage = Stage(args.stage)
    if stage not in TRANSITIONS[app.stage]:
        allowed = ", ".join(TRANSITIONS[app.stage]) or "none (final stage)"
        raise ToolError(f"Cannot move from {app.stage} to {stage}. Allowed next stages: {allowed}.")
    candidate = ctx.snapshot.candidate(app.candidate_id)
    who = candidate.full_name if candidate else app.candidate_id
    action = ctx.actions.add(
        "move_application",
        f"Move {who} from {app.stage} to {stage}" + (f" ({args.note})" if args.note else ""),
        args.model_dump(mode="json"),
    )
    ctx.proposed.append(action)
    return _json({"pending_action_id": action.id, "summary": action.summary, "status": "awaiting human confirmation"})


async def propose_interview(ctx: ToolContext, args: ScheduleInput) -> ToolResult:
    app = ctx.snapshot.application(args.application_id)
    if app is None:
        raise ToolError(f"No application with id {args.application_id}.")
    if args.scheduled_at.tzinfo is None:
        args.scheduled_at = args.scheduled_at.replace(tzinfo=UTC)
    if args.scheduled_at < datetime.now(UTC):
        raise ToolError("scheduled_at is in the past.")
    candidate = ctx.snapshot.candidate(app.candidate_id)
    who = candidate.full_name if candidate else app.candidate_id
    action = ctx.actions.add(
        "schedule_interview",
        f"Schedule a {args.duration_minutes}-minute {InterviewKind(args.kind).replace('_', ' ').lower()} "
        f"with {args.interviewer} for {who} at {args.scheduled_at:%Y-%m-%d %H:%M} UTC",
        args.model_dump(mode="json"),
    )
    ctx.proposed.append(action)
    return _json({"pending_action_id": action.id, "summary": action.summary, "status": "awaiting human confirmation"})


class ToolError(Exception):
    """A problem the model can fix (bad id, illegal stage); returned to it as an ``is_error`` result."""


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    input_model: type[BaseModel]
    run: Callable[[ToolContext, Any], Awaitable[ToolResult]]

    def definition(self) -> dict[str, Any]:
        schema = self.input_model.model_json_schema()
        schema.pop("title", None)
        for prop in schema.get("properties", {}).values():
            prop.pop("title", None)
        return {"name": self.name, "description": self.description, "input_schema": schema}

    async def __call__(self, ctx: ToolContext, raw_input: dict[str, Any]) -> ToolResult:
        try:
            args = self.input_model.model_validate(raw_input)
        except ValidationError as exc:
            raise ToolError(f"Invalid input: {exc.errors(include_url=False)}") from exc
        return await self.run(ctx, args)


TOOLS: tuple[Tool, ...] = (
    Tool(
        "search_hireflow",
        "Semantic and keyword search over HireFlow job postings, candidate profiles and applications "
        "(cover letters, stage notes, interview feedback). Use it for open questions such as who has "
        "a skill or what an interviewer said. Results can be cited.",
        SearchInput,
        search_hireflow,
    ),
    Tool("list_jobs", "List job postings with their ids, skills and application counts.", ListJobsInput, list_jobs),
    Tool(
        "get_job_pipeline",
        "Every application for one job: candidate, stage, allowed next stages, skill match and interviews.",
        PipelineInput,
        get_job_pipeline,
    ),
    Tool(
        "get_candidate",
        "Look up candidates by name or email, with their applications and application ids.",
        CandidateInput,
        get_candidate,
    ),
    Tool("upcoming_interviews", "Interviews scheduled in the next N days.", UpcomingInput, upcoming_interviews),
    Tool(
        "propose_stage_change",
        "Propose moving an application to another pipeline stage. Does not change anything: it creates "
        "a pending action that a recruiter must confirm. Only stages in allowed_next_stages are valid.",
        MoveInput,
        propose_stage_change,
    ),
    Tool(
        "propose_interview",
        "Propose scheduling an interview for an application. Creates a pending action that a recruiter "
        "must confirm; nothing is booked until then.",
        ScheduleInput,
        propose_interview,
    ),
)
TOOLS_BY_NAME = {t.name: t for t in TOOLS}
