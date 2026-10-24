"""Embedding backends. Claude has no embeddings endpoint, so vectors are computed locally."""

from __future__ import annotations

import hashlib
import re
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

Vectors = NDArray[np.float32]

TOKEN = re.compile(r"[a-z0-9#+]+(?:\.[a-z0-9]+)*|\.net", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Lowercased word tokens that keep tech names whole: "C#", ".NET", "CI/CD" -> "ci", "cd"."""
    return [t.lower() for t in TOKEN.findall(text)]


class Embedder(Protocol):
    name: str

    def embed_documents(self, texts: list[str]) -> Vectors: ...

    def embed_query(self, text: str) -> Vectors: ...


def _normalize(matrix: Vectors) -> Vectors:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return (matrix / np.where(norms == 0, 1, norms)).astype(np.float32)


class HashingEmbedder:
    """Deterministic bag-of-words vectors (feature hashing). No model download; used in tests and CI."""

    def __init__(self, dim: int = 512) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def _vector(self, text: str) -> Vectors:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.md5(token.encode(), usedforsecurity=False).digest()
            index = int.from_bytes(digest[:4], "little") % self.dim
            vec[index] += 1.0 if digest[4] & 1 else -1.0
        return vec

    def embed_documents(self, texts: list[str]) -> Vectors:
        return _normalize(np.stack([self._vector(t) for t in texts]))

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = _normalize(self._vector(text)[None, :])[0]
        return vector


class FastEmbedEmbedder:
    """Small ONNX sentence-embedding model (bge-small, 384 dims) running on CPU via fastembed."""

    def __init__(self, model_name: str) -> None:
        from fastembed import TextEmbedding  # imported lazily: loading onnxruntime takes a moment

        self.name = model_name
        self._model = TextEmbedding(model_name=model_name)

    def embed_documents(self, texts: list[str]) -> Vectors:
        return _normalize(np.array(list(self._model.passage_embed(texts)), dtype=np.float32))

    def embed_query(self, text: str) -> Vectors:
        vector: Vectors = _normalize(np.array(list(self._model.query_embed(text)), dtype=np.float32))[0]
        return vector
