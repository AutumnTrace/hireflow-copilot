"""Keeps the latest HireFlow snapshot and the search index built from it."""

from __future__ import annotations

import asyncio
import logging
import time

from hireflow_copilot.hireflow.client import HireFlowClient
from hireflow_copilot.hireflow.models import Snapshot
from hireflow_copilot.rag.documents import build_documents
from hireflow_copilot.rag.embeddings import Embedder
from hireflow_copilot.rag.index import HybridIndex

log = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self, client: HireFlowClient, embedder: Embedder, max_age_seconds: float = 300) -> None:
        self.client = client
        self.embedder = embedder
        self.max_age_seconds = max_age_seconds
        self.snapshot = Snapshot()
        self.index = HybridIndex([], embedder)
        self._loaded_at = 0.0
        self._lock = asyncio.Lock()

    async def refresh(self) -> None:
        """Re-read HireFlow and rebuild the index. Cheap at demo scale; see README for the large-scale path."""
        async with self._lock:
            started = time.perf_counter()
            snapshot = await self.client.snapshot()
            documents = build_documents(snapshot)
            # Embedding is CPU-bound; keep it off the event loop.
            index = await asyncio.to_thread(HybridIndex, documents, self.embedder)
            self.snapshot, self.index, self._loaded_at = snapshot, index, time.monotonic()
            log.info(
                "index rebuilt",
                extra={"documents": len(documents), "ms": round((time.perf_counter() - started) * 1000)},
            )

    async def ensure_fresh(self) -> None:
        if time.monotonic() - self._loaded_at > self.max_age_seconds:
            await self.refresh()
