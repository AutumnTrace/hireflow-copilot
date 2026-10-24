"""Single-shot RAG: retrieve, hand the hits to Claude as search results, return a cited answer."""

from __future__ import annotations

from opentelemetry import trace
from pydantic import BaseModel

from hireflow_copilot.llm.citations import CitedAnswer, Source, render, search_result_block
from hireflow_copilot.llm.claude import Claude
from hireflow_copilot.llm.pricing import Usage
from hireflow_copilot.rag.index import Hit, HybridIndex

tracer = trace.get_tracer(__name__)

SYSTEM = """You answer recruiters' questions about the hiring pipeline in HireFlow, an applicant \
tracking system. The user's message contains search results retrieved from HireFlow for this \
question. Answer only from those results. If they do not contain the answer, say so plainly \
instead of guessing. Be brief: a few sentences or a short list. Use names, stages and numbers \
exactly as they appear in the results."""


class RetrievedDoc(BaseModel):
    title: str
    source: str
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class AskResult(CitedAnswer):
    mode: str
    retrieved: list[RetrievedDoc]
    usage: dict[str, float | int]


def _retrieved(hits: list[Hit]) -> list[RetrievedDoc]:
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


def _usage(usage: Usage) -> dict[str, float | int]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cost_usd": round(usage.cost_usd, 6),
    }


def extractive_answer(question: str, index: HybridIndex, k: int) -> AskResult:
    """Demo mode: no model call. Returns the best-matching passages so the retrieval path is visible."""
    with tracer.start_as_current_span("retrieval.search"):
        hits = index.search(question, k=k)
    sources = [Source.of(n, h.document, list(h.document.blocks[:2])) for n, h in enumerate(hits[:3], start=1)]
    answer = (
        "Demo mode (no ANTHROPIC_API_KEY): showing the top passages instead of a generated answer."
        if hits
        else "Nothing in HireFlow matched that question."
    )
    return AskResult(answer=answer, sources=sources, mode="demo", retrieved=_retrieved(hits), usage=_usage(Usage()))


async def answer(question: str, index: HybridIndex, claude: Claude, k: int) -> AskResult:
    with tracer.start_as_current_span("retrieval.search") as span:
        hits = index.search(question, k=k)
        span.set_attribute("retrieval.hits", len(hits))

    if not hits:
        return AskResult(
            answer="Nothing in HireFlow matched that question.",
            sources=[],
            mode="llm",
            retrieved=[],
            usage=_usage(Usage()),
        )

    docs = [h.document for h in hits]
    usage = Usage()
    response = await claude.create(
        system=SYSTEM,
        messages=[
            {
                "role": "user",
                "content": [*(search_result_block(d) for d in docs), {"type": "text", "text": question}],
            }
        ],
        effort=claude.settings.ask_effort,
        usage=usage,
    )
    if response.stop_reason == "refusal":
        cited = CitedAnswer(answer="The model declined to answer this question.", sources=[])
    else:
        cited = render(response.content, docs)
        if response.stop_reason == "max_tokens":
            cited.answer += "\n\n(Answer cut off at the token limit.)"
    return AskResult(**cited.model_dump(), mode="llm", retrieved=_retrieved(hits), usage=_usage(usage))
