"""FastAPI app: retrieval, cited answers, and an agent over HireFlow, behind cost guardrails."""

from __future__ import annotations

import hmac
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Annotated, Literal

import anthropic
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import BaseModel, Field

from hireflow_copilot import observability
from hireflow_copilot.agent.loop import AgentResult, run_agent
from hireflow_copilot.agent.tools import ActionStore, ToolContext
from hireflow_copilot.config import Settings, get_settings
from hireflow_copilot.guard import LimitExceeded, RateLimiter, SpendLedger
from hireflow_copilot.hireflow.client import FixtureHireFlowClient, GraphQLHireFlowClient, HireFlowClient
from hireflow_copilot.hireflow.models import HireFlowError, InterviewKind, Stage
from hireflow_copilot.knowledge import KnowledgeBase
from hireflow_copilot.llm.claude import Claude
from hireflow_copilot.rag.answer import AskResult, RetrievedDoc, answer, extractive_answer
from hireflow_copilot.rag.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder

log = logging.getLogger(__name__)


@dataclass
class Services:
    settings: Settings
    kb: KnowledgeBase
    claude: Claude | None
    limiter: RateLimiter
    ledger: SpendLedger
    actions: ActionStore


def build_hireflow(settings: Settings) -> HireFlowClient:
    if settings.source == "graphql":
        token = settings.hireflow_token.get_secret_value() if settings.hireflow_token else None
        return GraphQLHireFlowClient(settings.hireflow_url, token=token, dev_token=settings.hireflow_dev_token)
    return FixtureHireFlowClient(fixture=settings.fixture)


def build_embedder(settings: Settings) -> Embedder:
    if settings.embedder == "hashing":
        return HashingEmbedder()
    return FastEmbedEmbedder(settings.embedding_model)


def create_app(
    settings: Settings | None = None,
    hireflow: HireFlowClient | None = None,
    embedder: Embedder | None = None,
    anthropic_client: anthropic.AsyncAnthropic | None = None,
) -> FastAPI:
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        provider = observability.setup(settings)
        ledger = SpendLedger(settings.daily_budget_usd)
        kb = KnowledgeBase(hireflow or build_hireflow(settings), embedder or build_embedder(settings))
        await kb.refresh()
        claude = None if settings.demo_mode and anthropic_client is None else Claude(settings, ledger, anthropic_client)
        if claude is not None and provider is not None:
            # Nest the SDK's own spans (one per HTTP call) under ours.
            claude.client = claude.client.with_options(open_telemetry={"tracer_provider": provider})
        app.state.services = Services(
            settings=settings,
            kb=kb,
            claude=claude,
            limiter=RateLimiter(settings.rate_limit_per_minute, settings.rate_limit_per_day),
            ledger=ledger,
            actions=ActionStore(),
        )
        log.info(
            "copilot ready",
            extra={"mode": "demo" if claude is None else "llm", "source": settings.source, "model": settings.model},
        )
        yield

    app = FastAPI(
        title="HireFlow Copilot",
        version="0.1.0",
        description="RAG and a tool-using Claude agent over the HireFlow GraphQL API.",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["content-type", "x-copilot-key"],
        expose_headers=["retry-after"],
    )

    if settings.otel_exporter != "none":
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

        FastAPIInstrumentor.instrument_app(app, excluded_urls="health")

    @app.exception_handler(LimitExceeded)
    async def _limit(_: Request, exc: LimitExceeded) -> JSONResponse:
        return JSONResponse({"detail": exc.reason}, status_code=429, headers={"Retry-After": str(exc.retry_after)})

    @app.exception_handler(anthropic.APIError)
    async def _upstream(_: Request, exc: anthropic.APIError) -> JSONResponse:
        log.warning("claude api error", extra={"error": type(exc).__name__, "detail": str(exc)})
        if isinstance(exc, anthropic.RateLimitError | anthropic.OverloadedError | anthropic.APIConnectionError):
            return JSONResponse({"detail": "Claude is busy right now. Try again shortly."}, status_code=503)
        return JSONResponse({"detail": "The model call failed."}, status_code=502)

    return _routes(app)


def services(request: Request) -> Services:
    svc: Services = request.app.state.services
    return svc


Svc = Annotated[Services, Depends(services)]


def require_key(svc: Svc, x_copilot_key: Annotated[str | None, Header()] = None) -> None:
    expected = svc.settings.access_key
    if expected is not None and not hmac.compare_digest(x_copilot_key or "", expected.get_secret_value()):
        raise HTTPException(401, "Missing or wrong X-Copilot-Key header.")


def client_id(request: Request, svc: Svc) -> str:
    if svc.settings.trust_proxy and (forwarded := request.headers.get("x-forwarded-for")):
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


def llm_guard(request: Request, svc: Svc, _: Annotated[None, Depends(require_key)]) -> None:
    """Run before every endpoint that can spend money."""
    svc.limiter.check(client_id(request, svc))
    svc.ledger.check()


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=5, ge=1, le=20)
    kind: Literal["job", "candidate", "application"] | None = None


class AskRequest(BaseModel):
    question: str = Field(
        min_length=1, examples=["Who has production RAG experience, and what did the interviewer think?"]
    )


class AgentRequest(BaseModel):
    message: str = Field(
        min_length=1, examples=["Who should move forward for the AI Engineer role? Propose next steps."]
    )


class ConfirmResult(BaseModel):
    id: str
    status: str
    result: str | None


def _check_length(svc: Services, text: str) -> None:
    if len(text) > svc.settings.max_question_chars:
        raise HTTPException(422, f"Keep it under {svc.settings.max_question_chars} characters.")


def _routes(app: FastAPI) -> FastAPI:
    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    async def health(svc: Svc) -> dict[str, object]:
        return {
            "status": "ok",
            "mode": "demo" if svc.claude is None else "llm",
            "source": svc.settings.source,
            "fixture": svc.settings.fixture if svc.settings.source == "fixtures" else None,
            "model": svc.settings.model if svc.claude else None,
            "embedder": svc.kb.embedder.name,
            "documents": len(svc.kb.index.documents),
        }

    @app.get("/v1/usage")
    async def usage(svc: Svc) -> dict[str, object]:
        """Today's spend against the global budget, plus the per-client limits."""
        return {
            **svc.ledger.snapshot(),
            "limits": {
                "per_minute": svc.settings.rate_limit_per_minute,
                "per_day": svc.settings.rate_limit_per_day,
                "max_tokens_per_call": svc.settings.max_tokens,
                "max_agent_steps": svc.settings.max_agent_steps,
            },
        }

    @app.post("/v1/search")
    async def search(body: SearchRequest, svc: Svc) -> list[RetrievedDoc]:
        """Retrieval only, no model call. Useful for seeing what the RAG step feeds Claude."""
        await svc.kb.ensure_fresh()
        hits = svc.kb.index.search(body.query, k=body.k, kinds={body.kind} if body.kind else None)
        return [
            RetrievedDoc(
                title=h.document.title,
                source=h.document.source,
                score=round(h.score, 5),
                dense_rank=h.dense_rank,
                sparse_rank=h.sparse_rank,
            )
            for h in hits
        ]

    @app.post("/v1/ask", dependencies=[Depends(llm_guard)])
    async def ask(body: AskRequest, svc: Svc) -> AskResult:
        """One retrieval, one Claude call, an answer with citations back to HireFlow records."""
        _check_length(svc, body.question)
        await svc.kb.ensure_fresh()
        if svc.claude is None:
            return extractive_answer(body.question, svc.kb.index, svc.settings.top_k)
        return await answer(body.question, svc.kb.index, svc.claude, svc.settings.top_k)

    @app.post("/v1/agent", dependencies=[Depends(llm_guard)])
    async def agent(body: AgentRequest, svc: Svc) -> AgentResult:
        """Tool-using agent: searches, reads pipelines, and proposes (never performs) changes."""
        _check_length(svc, body.message)
        if svc.claude is None:
            raise HTTPException(503, "The agent needs ANTHROPIC_API_KEY. /v1/ask and /v1/search work in demo mode.")
        await svc.kb.ensure_fresh()
        ctx = ToolContext(kb=svc.kb, actions=svc.actions, top_k=svc.settings.top_k)
        return await run_agent(body.message, ctx, svc.claude)

    @app.post("/v1/actions/{action_id}/confirm", dependencies=[Depends(require_key)])
    async def confirm(action_id: str, svc: Svc) -> ConfirmResult:
        """Human-in-the-loop: carry out a change the agent proposed."""
        action = svc.actions.get(action_id)
        if action is None:
            raise HTTPException(404, "No such pending action (they expire after an hour).")
        if action.status != "pending":
            return ConfirmResult(id=action.id, status=action.status, result=action.result)
        p = action.params
        try:
            if action.kind == "move_application":
                stage = await svc.kb.client.move_application(p["application_id"], Stage(p["stage"]), p.get("note"))
                action.result = f"Application is now {stage}."
            else:
                interview_id = await svc.kb.client.schedule_interview(
                    p["application_id"],
                    InterviewKind(p["kind"]),
                    datetime.fromisoformat(p["scheduled_at"]),
                    p["duration_minutes"],
                    p["interviewer"],
                )
                action.result = f"Interview {interview_id} scheduled."
            action.status = "done"
        except HireFlowError as exc:
            action.status, action.result = "failed", f"{exc.kind}: {exc.message}"
        await svc.kb.refresh()
        return ConfirmResult(id=action.id, status=action.status, result=action.result)

    @app.post("/v1/reindex", dependencies=[Depends(require_key)])
    async def reindex(svc: Svc) -> dict[str, int]:
        await svc.kb.refresh()
        return {"documents": len(svc.kb.index.documents)}

    return app


app = create_app()
