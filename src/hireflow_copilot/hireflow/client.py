"""Two ways to reach HireFlow: the bundled demo dataset, or the real GraphQL API."""

from __future__ import annotations

import json
import re
import uuid
from datetime import UTC, datetime, timedelta
from importlib import resources
from typing import Any, Protocol

import httpx

from hireflow_copilot.hireflow import queries
from hireflow_copilot.hireflow.models import (
    TRANSITIONS,
    Application,
    Candidate,
    Company,
    HireFlowError,
    Interview,
    InterviewKind,
    Job,
    SkillMatch,
    Snapshot,
    Stage,
    StageTransition,
)


class HireFlowClient(Protocol):
    async def snapshot(self) -> Snapshot: ...

    async def move_application(self, application_id: str, stage: Stage, note: str | None) -> Stage: ...

    async def schedule_interview(
        self,
        application_id: str,
        kind: InterviewKind,
        scheduled_at: datetime,
        duration_minutes: int,
        interviewer: str,
    ) -> str: ...


_RELATIVE = re.compile(r"^(?P<sign>[+-])(?P<days>\d+)d(?:@(?P<hour>\d{1,2}))?$")
FIXTURES = {"seed": "hireflow_seed.json", "web-demo": "hireflow_web_demo.json"}


def _resolve_dates(value: Any, now: datetime) -> Any:
    """Fixture dates may be relative ("+2d", "-3d@14" = 3 days ago at 14:00 UTC) so they never go stale."""
    if isinstance(value, dict):
        return {k: _resolve_dates(v, now) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_dates(v, now) for v in value]
    if isinstance(value, str) and (m := _RELATIVE.match(value)):
        days = int(m["days"]) * (1 if m["sign"] == "+" else -1)
        moment = (now + timedelta(days=days)).replace(minute=0, second=0, microsecond=0)
        if m["hour"] is not None:
            moment = moment.replace(hour=int(m["hour"]))
        return moment.isoformat()
    return value


class FixtureHireFlowClient:
    """In-memory HireFlow loaded from a bundled dataset. Writes apply the real stage rules.

    ``seed`` is HireFlow's own demo seed plus extra candidates; ``web-demo`` is the dataset the
    HireFlow Web demo runs in the browser, exported by its ``scripts/export-demo-data.ts``.
    """

    def __init__(self, data: dict[str, Any] | None = None, fixture: str = "seed") -> None:
        if data is None:
            raw = resources.files("hireflow_copilot.data").joinpath(FIXTURES[fixture]).read_text("utf-8")
            data = json.loads(raw)
        self._snapshot = Snapshot.model_validate(_resolve_dates(data, datetime.now(UTC)))
        companies = {c.id: c.name for c in self._snapshot.companies}
        for job in self._snapshot.jobs:
            job.company_name = companies.get(job.company_id, "")
        for app in self._snapshot.applications:
            app.allowed_next_stages = list(TRANSITIONS[app.stage])
            posting, person = self._snapshot.job(app.job_posting_id), self._snapshot.candidate(app.candidate_id)
            if app.skill_match is None and posting and person:
                app.skill_match = SkillMatch.evaluate(posting.required_skills, person.skills)

    async def snapshot(self) -> Snapshot:
        return self._snapshot.model_copy(deep=True)

    async def move_application(self, application_id: str, stage: Stage, note: str | None) -> Stage:
        app = self._snapshot.application(application_id)
        if app is None:
            raise HireFlowError("EntityNotFoundError", f"Application {application_id} was not found.")
        if stage not in TRANSITIONS[app.stage]:
            allowed = ", ".join(TRANSITIONS[app.stage]) or "none, it is a final stage"
            raise HireFlowError(
                "InvalidStageTransitionError",
                f"Cannot move from {app.stage} to {stage}. Allowed: {allowed}.",
            )
        app.history.append(StageTransition(from_=app.stage, to=stage, note=note, changed_at=datetime.now(UTC)))
        app.stage = stage
        app.allowed_next_stages = list(TRANSITIONS[stage])
        return stage

    async def schedule_interview(
        self,
        application_id: str,
        kind: InterviewKind,
        scheduled_at: datetime,
        duration_minutes: int,
        interviewer: str,
    ) -> str:
        app = self._snapshot.application(application_id)
        if app is None:
            raise HireFlowError("EntityNotFoundError", f"Application {application_id} was not found.")
        if not TRANSITIONS[app.stage]:
            raise HireFlowError("ApplicationClosedError", f"Application is {app.stage}; it can't get new interviews.")
        interview = Interview(
            id=str(uuid.uuid4()),
            application_id=application_id,
            kind=kind,
            scheduled_at=scheduled_at,
            duration_minutes=duration_minutes,
            interviewer=interviewer,
        )
        app.interviews.append(interview)
        return interview.id


class GraphQLHireFlowClient:
    """Talks to a running HireFlow API (``dotnet run`` or ``docker compose up``)."""

    def __init__(
        self, url: str, token: str | None = None, dev_token: bool = True, http: httpx.AsyncClient | None = None
    ):
        self._url = url
        self._token = token
        self._dev_token = dev_token
        self._http = http or httpx.AsyncClient(timeout=15.0)

    async def _auth_header(self) -> dict[str, str]:
        if self._token is None and self._dev_token:
            # HireFlow exposes POST /auth/token in development; it lives next to /graphql.
            token_url = self._url.rsplit("/graphql", 1)[0] + "/auth/token"
            response = await self._http.post(token_url, json={"name": "HireFlow Copilot", "role": "Recruiter"})
            response.raise_for_status()
            self._token = response.json()["accessToken"]
        return {"Authorization": f"Bearer {self._token}"} if self._token else {}

    async def _execute(self, query: str, variables: dict[str, Any] | None = None) -> dict[str, Any]:
        response = await self._http.post(
            self._url, json={"query": query, "variables": variables or {}}, headers=await self._auth_header()
        )
        # GraphQL servers report validation failures (such as the cost limit) as 400 with an
        # "errors" body; surface that message rather than a bare HTTP status.
        body = response.json() if response.headers.get("content-type", "").startswith("application") else {}
        if body.get("errors"):
            first = body["errors"][0]
            code = first.get("extensions", {}).get("code", "GraphQLError")
            raise HireFlowError(code, "; ".join(e.get("message", "?") for e in body["errors"]))
        response.raise_for_status()
        data: dict[str, Any] = body["data"]
        return data

    async def _paginate(self, query: str, field: str) -> list[dict[str, Any]]:
        nodes: list[dict[str, Any]] = []
        after: str | None = None
        while True:
            page = (await self._execute(query, {"after": after}))[field]
            nodes.extend(page["nodes"])
            if not page["pageInfo"]["hasNextPage"]:
                return nodes
            after = page["pageInfo"]["endCursor"]

    async def snapshot(self) -> Snapshot:
        companies = await self._paginate(queries.COMPANIES, "companies")
        candidates = await self._paginate(queries.CANDIDATES, "candidates")
        jobs = await self._paginate(queries.JOBS, "jobs")
        applications = [app for job in jobs for app in job.pop("applications")]
        for job in jobs:
            job["companyName"] = job.pop("company")["name"]
        return Snapshot(
            companies=[Company.model_validate(c) for c in companies],
            candidates=[Candidate.model_validate(c) for c in candidates],
            jobs=[Job.model_validate(j) for j in jobs],
            applications=[Application.model_validate(a) for a in applications],
        )

    @staticmethod
    def _raise_payload_errors(payload: dict[str, Any]) -> None:
        if errors := payload.get("errors"):
            first = errors[0]
            raise HireFlowError(first.get("__typename", "Error"), first.get("message", "Unknown error"))

    async def move_application(self, application_id: str, stage: Stage, note: str | None) -> Stage:
        data = await self._execute(
            queries.MOVE_APPLICATION,
            {"input": {"applicationId": application_id, "stage": stage.value, "note": note}},
        )
        payload = data["moveApplication"]
        self._raise_payload_errors(payload)
        return Stage(payload["jobApplication"]["stage"])

    async def schedule_interview(
        self,
        application_id: str,
        kind: InterviewKind,
        scheduled_at: datetime,
        duration_minutes: int,
        interviewer: str,
    ) -> str:
        data = await self._execute(
            queries.SCHEDULE_INTERVIEW,
            {
                "input": {
                    "applicationId": application_id,
                    "kind": kind.value,
                    "scheduledAt": scheduled_at.isoformat(),
                    "durationMinutes": duration_minutes,
                    "interviewer": interviewer,
                }
            },
        )
        payload = data["scheduleInterview"]
        self._raise_payload_errors(payload)
        interview_id: str = payload["interview"]["id"]
        return interview_id
