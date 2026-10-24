"""The agent loop: Claude picks tools, we run them, repeat until it answers or hits a cap.

A hand-written loop rather than the SDK's tool runner, because each step needs things the runner
doesn't expose: a spend check before every call, tool results that are ``search_result`` blocks
whose order must be tracked for citations, and a hard step limit.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime
from typing import Any, Literal

from opentelemetry import trace
from pydantic import BaseModel

from hireflow_copilot.agent.tools import TOOLS, TOOLS_BY_NAME, ToolContext, ToolError
from hireflow_copilot.guard import LimitExceeded
from hireflow_copilot.llm.citations import CitedAnswer, render
from hireflow_copilot.llm.claude import Claude
from hireflow_copilot.llm.pricing import Usage

log = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)

SYSTEM = """You are HireFlow Copilot, an assistant for recruiters working in HireFlow, an \
applicant tracking system with companies, job postings, candidates, applications and interviews.

Use the tools to look things up; never invent ids, names, stages or ratings. Prefer \
search_hireflow for open questions and the structured tools for exact lists. When you state \
something you found through search_hireflow, cite it.

You cannot change HireFlow directly. To move an application or book an interview, call \
propose_stage_change or propose_interview; that creates a pending action a recruiter must \
confirm. Say clearly that it is pending. Only propose a change when the user asks for one.

Keep answers short and concrete: a few sentences or a compact list."""


class ToolCall(BaseModel):
    name: str
    input: dict[str, Any]
    is_error: bool


class ProposedAction(BaseModel):
    id: str
    kind: Literal["move_application", "schedule_interview"]
    summary: str
    # The mutation's input, so a client holding its own HireFlow session can apply it directly.
    params: dict[str, Any]
    confirm_url: str


class AgentResult(CitedAnswer):
    stop: str
    steps: int
    tool_calls: list[ToolCall]
    proposed_actions: list[ProposedAction]
    usage: dict[str, float | int]


async def _run_tool(ctx: ToolContext, block: Any, calls: list[ToolCall]) -> dict[str, Any]:
    tool = TOOLS_BY_NAME.get(block.name)
    with tracer.start_as_current_span(f"tool.{block.name}") as span:
        try:
            if tool is None:
                raise ToolError(f"Unknown tool {block.name}.")
            content = await tool(ctx, block.input)
            is_error = False
        except ToolError as exc:
            content, is_error = str(exc), True
        span.set_attribute("tool.is_error", is_error)
    calls.append(ToolCall(name=block.name, input=block.input, is_error=is_error))
    result: dict[str, Any] = {"type": "tool_result", "tool_use_id": block.id, "content": content}
    if is_error:
        result["is_error"] = True
    return result


async def run_agent(message: str, ctx: ToolContext, claude: Claude) -> AgentResult:
    settings = claude.settings
    usage = Usage()
    calls: list[ToolCall] = []
    # The date goes in the user turn, not the system prompt, so the cached prefix stays byte-identical.
    messages: list[Any] = [{"role": "user", "content": f"Today is {datetime.now(UTC):%A %Y-%m-%d} (UTC).\n\n{message}"}]
    tool_defs = [t.definition() for t in TOOLS]
    response = None
    stop = "max_steps"

    for step in range(1, settings.max_agent_steps + 1):
        with tracer.start_as_current_span("agent.step") as span:
            span.set_attribute("agent.step", step)
            try:
                claude.ledger.check()
            except LimitExceeded:
                stop = "budget"
                break

            response = await claude.create(
                system=SYSTEM, messages=messages, tools=tool_defs, effort=settings.agent_effort, usage=usage
            )
            # Send back the full content (thinking, text, tool_use) unchanged on the next call.
            messages.append({"role": "assistant", "content": response.content})

            if response.stop_reason != "tool_use":
                stop = response.stop_reason or "end_turn"
                break

            tool_uses = [b for b in response.content if b.type == "tool_use"]
            # Run parallel tool calls concurrently; all results go back in ONE user message.
            results = await asyncio.gather(*(_run_tool(ctx, b, calls) for b in tool_uses))
            messages.append({"role": "user", "content": list(results)})

    if response is None:
        cited = CitedAnswer(answer="The daily demo budget is used up. Try again after 00:00 UTC.", sources=[])
    elif stop == "refusal":
        cited = CitedAnswer(answer="The model declined this request.", sources=[])
    else:
        cited = render(response.content, ctx.sent_documents)
        if stop == "max_steps":
            cited.answer = (cited.answer + "\n\n" if cited.answer else "") + (
                f"(Stopped after {settings.max_agent_steps} steps.)"
            )
        elif stop == "max_tokens":
            cited.answer += "\n\n(Answer cut off at the token limit.)"
        elif stop == "budget":
            cited.answer += "\n\n(Stopped early: the daily demo budget is used up.)"

    log.info("agent finished", extra={"stop": stop, "steps": usage.calls, "tools": [c.name for c in calls]})
    return AgentResult(
        **cited.model_dump(),
        stop=stop,
        steps=usage.calls,
        tool_calls=calls,
        proposed_actions=[
            ProposedAction(
                id=a.id, kind=a.kind, summary=a.summary, params=a.params, confirm_url=f"/v1/actions/{a.id}/confirm"
            )
            for a in ctx.proposed
        ],
        usage={
            "input_tokens": usage.input_tokens,
            "output_tokens": usage.output_tokens,
            "cache_read_tokens": usage.cache_read_tokens,
            "cost_usd": round(usage.cost_usd, 6),
        },
    )
