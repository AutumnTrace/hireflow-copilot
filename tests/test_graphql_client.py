import json

import httpx
import pytest
import respx

from hireflow_copilot.hireflow.client import GraphQLHireFlowClient
from hireflow_copilot.hireflow.models import HireFlowError, Stage

URL = "http://hireflow.test/graphql"


def _page(nodes: list[dict], has_next: bool = False, cursor: str | None = None) -> dict:
    return {"pageInfo": {"hasNextPage": has_next, "endCursor": cursor}, "nodes": nodes}


JOB = {
    "id": "j1",
    "companyId": "c1",
    "title": "AI Engineer",
    "description": "RAG",
    "location": None,
    "workMode": "REMOTE",
    "employmentType": "FULL_TIME",
    "status": "OPEN",
    "requiredSkills": ["Python"],
    "salary": None,
    "company": {"name": "Bluefin"},
    "applications": [
        {
            "id": "a1",
            "jobPostingId": "j1",
            "candidateId": "p1",
            "stage": "APPLIED",
            "coverLetter": None,
            "appliedAt": "2026-10-01T10:00:00Z",
            "allowedNextStages": ["SCREENING", "REJECTED", "WITHDRAWN"],
            "history": [],
            "interviews": [],
            "skillMatch": {"score": 1.0, "matchedSkills": ["Python"], "missingSkills": []},
        }
    ],
}


def _responder(request: httpx.Request) -> httpx.Response:
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer dev-token"
    query, after = body["query"], body["variables"].get("after")
    if "companies(" in query:
        return httpx.Response(200, json={"data": {"companies": _page([{"id": "c1", "name": "Bluefin"}])}})
    if "candidates(" in query:
        # Two pages, to exercise cursor pagination.
        person = {"fullName": "V", "email": "v@x", "headline": None, "yearsOfExperience": 5, "skills": ["Python"]}
        if after is None:
            return httpx.Response(200, json={"data": {"candidates": _page([{"id": "p1", **person}], True, "c1")}})
        return httpx.Response(200, json={"data": {"candidates": _page([{"id": "p2", **person}])}})
    if "jobs(" in query:
        return httpx.Response(200, json={"data": {"jobs": _page([JOB])}})
    if "moveApplication" in query:
        error = {"__typename": "InvalidStageTransitionError", "message": "Cannot move APPLIED to HIRED."}
        return httpx.Response(200, json={"data": {"moveApplication": {"jobApplication": None, "errors": [error]}}})
    raise AssertionError(query)


@respx.mock
async def test_snapshot_paginates_and_flattens_applications() -> None:
    respx.post("http://hireflow.test/auth/token").respond(json={"accessToken": "dev-token"})
    respx.post(URL).mock(side_effect=_responder)

    snapshot = await GraphQLHireFlowClient(URL).snapshot()

    assert [c.id for c in snapshot.candidates] == ["p1", "p2"]
    assert snapshot.jobs[0].company_name == "Bluefin"
    assert snapshot.applications[0].allowed_next_stages == [Stage.SCREENING, Stage.REJECTED, Stage.WITHDRAWN]


@respx.mock
async def test_mutation_errors_become_hireflow_errors() -> None:
    respx.post("http://hireflow.test/auth/token").respond(json={"accessToken": "dev-token"})
    respx.post(URL).mock(side_effect=_responder)

    with pytest.raises(HireFlowError) as exc:
        await GraphQLHireFlowClient(URL).move_application("a1", Stage.HIRED, None)
    assert exc.value.kind == "InvalidStageTransitionError"


@respx.mock
async def test_graphql_errors_on_400_keep_their_message() -> None:
    respx.post("http://hireflow.test/auth/token").respond(json={"accessToken": "dev-token"})
    error = {"message": "The maximum allowed field cost was exceeded.", "extensions": {"code": "HC0047"}}
    respx.post(URL).respond(400, json={"errors": [error]})

    with pytest.raises(HireFlowError) as exc:
        await GraphQLHireFlowClient(URL).snapshot()
    assert exc.value.kind == "HC0047"
    assert "field cost" in exc.value.message
