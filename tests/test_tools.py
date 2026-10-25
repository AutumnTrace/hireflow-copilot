import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from hireflow_copilot.agent.tools import CandidateInput, ToolContext, ToolError, get_candidate
from hireflow_copilot.hireflow.client import FixtureHireFlowClient
from hireflow_copilot.hireflow.models import Candidate


async def context(*extra: Candidate) -> ToolContext:
    """get_candidate only reads the snapshot, so a stand-in with one attribute is enough."""
    snapshot = await FixtureHireFlowClient().snapshot()
    snapshot.candidates.extend(extra)
    return cast(ToolContext, SimpleNamespace(snapshot=snapshot))


async def lookup(ctx: ToolContext, query: str) -> list[str]:
    result = await get_candidate(ctx, CandidateInput(name_or_email=query))
    rows = cast(list[dict[str, Any]], json.loads(cast(str, result)))
    return [row["full_name"] for row in rows]


STEFAN = Candidate(
    id="c0000000-0000-4000-8000-0000000000ff",
    full_name="Ștefan Țurcanu",
    email="stefan.turcanu@example.com",
    years_of_experience=3,
)


async def test_finds_by_part_of_a_name_in_any_case() -> None:
    assert await lookup(await context(), "POPESCU") == ["Ana Popescu"]


async def test_words_can_come_in_any_order() -> None:
    assert await lookup(await context(), "popescu ana") == ["Ana Popescu"]


async def test_accents_are_optional() -> None:
    ctx = await context(STEFAN)
    assert await lookup(ctx, "stefan turcanu") == ["Ștefan Țurcanu"]
    assert await lookup(ctx, "Ștefan") == ["Ștefan Țurcanu"]


async def test_email_must_match_whole() -> None:
    ctx = await context(STEFAN)
    assert await lookup(ctx, " Stefan.Turcanu@Example.com ") == ["Ștefan Țurcanu"]
    with pytest.raises(ToolError):
        await lookup(ctx, "turcanu@example")


async def test_blank_query_is_a_tool_error_not_everyone() -> None:
    with pytest.raises(ToolError, match="name or email"):
        await lookup(await context(), "   ")
