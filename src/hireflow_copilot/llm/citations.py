"""Turn retrieved documents into ``search_result`` blocks, and Claude's cited answer back into JSON."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from pydantic import BaseModel

from hireflow_copilot.rag.documents import Document


class Source(BaseModel):
    number: int
    title: str
    source: str
    quotes: list[str]


class CitedAnswer(BaseModel):
    answer: str
    sources: list[Source]


def search_result_block(doc: Document) -> dict[str, Any]:
    return {
        "type": "search_result",
        "source": doc.source,
        "title": doc.title,
        "content": [{"type": "text", "text": block} for block in doc.blocks],
        # Citations are all-or-nothing per request, so every search_result we send enables them.
        "citations": {"enabled": True},
    }


def render(content: Sequence[Any], sent: Sequence[Document]) -> CitedAnswer:
    """Join the text blocks of a response and append ``[n]`` markers where Claude cited a result.

    ``search_result_index`` counts every search_result block in the request, in order, so ``sent``
    must list the documents in exactly the order they were sent.
    """
    numbers: dict[int, int] = {}
    quotes: dict[int, list[str]] = {}
    parts: list[str] = []
    for block in content:
        if block.type != "text":
            continue
        parts.append(block.text)
        markers: list[int] = []
        for citation in block.citations or []:
            if citation.type != "search_result_location":
                continue
            index = citation.search_result_index
            number = numbers.setdefault(index, len(numbers) + 1)
            quote = citation.cited_text.strip()
            if quote not in quotes.setdefault(index, []):
                quotes[index].append(quote)
            if number not in markers:
                markers.append(number)
        if markers:
            parts.append("".join(f"[{n}]" for n in markers))

    sources = [
        Source(number=number, title=sent[index].title, source=sent[index].source, quotes=quotes[index])
        for index, number in sorted(numbers.items(), key=lambda item: item[1])
        if index < len(sent)
    ]
    return CitedAnswer(answer="".join(parts).strip(), sources=sources)
