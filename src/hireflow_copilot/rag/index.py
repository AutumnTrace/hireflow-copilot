"""Hybrid retrieval: BM25 for exact terms ("C#", "Kafka"), embeddings for meaning, fused with RRF."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Literal

import numpy as np

from hireflow_copilot.rag.documents import Document, Kind
from hireflow_copilot.rag.embeddings import Embedder, tokenize

# Reciprocal Rank Fusion constant from Cormack et al. (2009). Larger values flatten the curve.
RRF_K = 60


class BM25:
    """Okapi BM25 over a fixed corpus. Small enough to read in one sitting."""

    def __init__(self, corpus: list[list[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [Counter(tokens) for tokens in corpus]
        self.lengths = [len(tokens) for tokens in corpus]
        self.avg_length = sum(self.lengths) / max(len(corpus), 1)
        df = Counter(term for doc in self.docs for term in doc)
        n = len(corpus)
        self.idf = {term: math.log(1 + (n - freq + 0.5) / (freq + 0.5)) for term, freq in df.items()}

    def scores(self, query: list[str]) -> list[float]:
        result = []
        for doc, length in zip(self.docs, self.lengths, strict=True):
            score = 0.0
            for term in query:
                tf = doc.get(term, 0)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg_length)
                    score += self.idf[term] * tf * (self.k1 + 1) / norm
            result.append(score)
        return result


@dataclass(frozen=True)
class Hit:
    document: Document
    score: float
    dense_rank: int | None
    sparse_rank: int | None


class HybridIndex:
    def __init__(self, documents: list[Document], embedder: Embedder) -> None:
        self.documents = documents
        self.embedder = embedder
        self._bm25 = BM25([tokenize(d.text) for d in documents])
        self._vectors = embedder.embed_documents([d.text for d in documents]) if documents else np.zeros((0, 1))

    def search(
        self,
        query: str,
        k: int = 5,
        kinds: set[Kind] | None = None,
        mode: Literal["hybrid", "dense", "bm25"] = "hybrid",
    ) -> list[Hit]:
        """Top ``k`` documents. ``mode`` lets the eval compare each retriever on its own."""
        if not self.documents:
            return []
        allowed = [i for i, d in enumerate(self.documents) if kinds is None or d.kind in kinds]

        dense = self._vectors @ self.embedder.embed_query(query)
        sparse = self._bm25.scores(tokenize(query))
        dense_order = sorted(allowed, key=lambda i: -dense[i])
        # Documents with no shared term get no BM25 rank instead of an arbitrary one.
        sparse_order = sorted((i for i in allowed if sparse[i] > 0), key=lambda i: -sparse[i])

        dense_rank = {doc: rank for rank, doc in enumerate(dense_order, start=1)}
        sparse_rank = {doc: rank for rank, doc in enumerate(sparse_order, start=1)}
        if mode == "dense":
            scores = {i: float(dense[i]) for i in dense_order}
        elif mode == "bm25":
            scores = {i: sparse[i] for i in sparse_order}
        else:
            scores = {
                i: 1 / (RRF_K + dense_rank[i]) + (1 / (RRF_K + sparse_rank[i]) if i in sparse_rank else 0.0)
                for i in allowed
            }
        top = sorted(scores, key=lambda i: -scores[i])[:k]
        return [Hit(self.documents[i], scores[i], dense_rank.get(i), sparse_rank.get(i)) for i in top]
