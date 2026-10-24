"""Offline retrieval eval: hit rate, MRR and recall at k for BM25, dense, and hybrid retrieval.

uv run python evals/run_retrieval.py                     # fastembed model (downloads ~70 MB once)
uv run python evals/run_retrieval.py --embedder hashing
uv run python evals/run_retrieval.py --min-hit-rate 0.9  # exit 1 below the bar (used in CI)
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from hireflow_copilot.hireflow.client import FixtureHireFlowClient
from hireflow_copilot.rag.documents import build_documents
from hireflow_copilot.rag.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder
from hireflow_copilot.rag.index import HybridIndex

HERE = Path(__file__).parent
MODES = ("bm25", "dense", "hybrid")


def evaluate(index: HybridIndex, cases: list[dict[str, Any]], k: int, mode: str) -> dict[str, Any]:
    hits = 0
    reciprocal = recall = 0.0
    misses = []
    for case in cases:
        ranked = [h.document.id for h in index.search(case["question"], k=len(index.documents), mode=mode)]  # type: ignore[arg-type]
        relevant = set(case["relevant"])
        first = next((rank for rank, doc in enumerate(ranked, start=1) if doc in relevant), None)
        if first is not None and first <= k:
            hits += 1
        else:
            misses.append(case["question"])
        reciprocal += 1 / first if first else 0.0
        recall += len(relevant & set(ranked[:k])) / len(relevant)
    n = len(cases)
    return {"hit_rate": hits / n, "mrr": reciprocal / n, "recall": recall / n, "misses": misses}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("-k", type=int, default=5)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--json", type=Path, help="Also write results to this file.")
    args = parser.parse_args()

    embedder: Embedder = HashingEmbedder() if args.embedder == "hashing" else FastEmbedEmbedder(args.model)
    snapshot = asyncio.run(FixtureHireFlowClient().snapshot())
    index = HybridIndex(build_documents(snapshot), embedder)
    cases = yaml.safe_load((HERE / "retrieval.yaml").read_text("utf-8"))

    unknown = {doc for c in cases for doc in c["relevant"]} - {d.id for d in index.documents}
    if unknown:
        print(f"eval references unknown documents: {sorted(unknown)}", file=sys.stderr)
        return 2

    results = {mode: evaluate(index, cases, args.k, mode) for mode in MODES}

    print(f"Retrieval eval: {len(cases)} questions, {len(index.documents)} documents, {embedder.name}, k={args.k}\n")
    print(f"| retriever | hit@{args.k} | MRR  | recall@{args.k} |")
    print("|-----------|-------|------|----------|")
    for mode, r in results.items():
        print(f"| {mode:<9} | {r['hit_rate']:.2f}  | {r['mrr']:.2f} | {r['recall']:.2f}     |")
    if results["hybrid"]["misses"]:
        print("\nHybrid misses:")
        for question in results["hybrid"]["misses"]:
            print(f"  - {question}")

    if args.json:
        args.json.write_text(json.dumps(results, indent=2), "utf-8")

    if results["hybrid"]["hit_rate"] < args.min_hit_rate:
        print(f"\nFAIL: hybrid hit@{args.k} {results['hybrid']['hit_rate']:.2f} < {args.min_hit_rate}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
