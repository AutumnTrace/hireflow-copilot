"""Live end-to-end eval against the real Claude API, with a hard spend cap.

    ANTHROPIC_API_KEY=... uv run python evals/run_live.py --budget 0.50
    uv run python evals/run_live.py --only kafka,propose-move

Each case gets a fresh fixture HireFlow, so proposals in one case can't leak into the next.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml
from fastapi.testclient import TestClient

from hireflow_copilot.config import Settings
from hireflow_copilot.hireflow.client import FixtureHireFlowClient
from hireflow_copilot.main import create_app
from hireflow_copilot.rag.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder

HERE = Path(__file__).parent


def check(case: dict[str, Any], body: dict[str, Any]) -> list[str]:
    problems = []
    text = body["answer"].lower()
    for needle in case.get("must_include", []):
        if needle.lower() not in text:
            problems.append(f"missing '{needle}'")
    for needle in case.get("must_not_include", []):
        if needle.lower() in text:
            problems.append(f"should not say '{needle}'")
    if (source := case.get("cites")) and source not in {s["source"] for s in body["sources"]}:
        problems.append(f"no citation of {source}")
    called = {c["name"] for c in body.get("tool_calls", [])}
    problems += [f"did not call {t}" for t in case.get("tools", []) if t not in called]
    if "proposes" in case and len(body.get("proposed_actions", [])) != case["proposes"]:
        problems.append(f"expected {case['proposes']} proposal(s), got {len(body.get('proposed_actions', []))}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget", type=float, default=0.50, help="Stop once this many USD are spent.")
    parser.add_argument("--only", help="Comma-separated case ids.")
    parser.add_argument("--embedder", choices=["fastembed", "hashing"], default="fastembed")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Set ANTHROPIC_API_KEY to run the live eval.", file=sys.stderr)
        return 2

    cases = yaml.safe_load((HERE / "answers.yaml").read_text("utf-8"))
    if args.only:
        wanted = set(args.only.split(","))
        cases = [c for c in cases if c["id"] in wanted]

    settings = Settings(
        embedder=args.embedder,
        daily_budget_usd=args.budget,
        rate_limit_per_minute=1000,
        rate_limit_per_day=1000,
        otel_exporter="none",
    )
    embedder: Embedder = (
        HashingEmbedder() if args.embedder == "hashing" else FastEmbedEmbedder(settings.embedding_model)
    )

    rows, spent = [], 0.0
    for case in cases:
        if spent >= args.budget:
            rows.append({"id": case["id"], "passed": False, "problems": ["skipped: budget reached"]})
            continue
        app = create_app(settings, hireflow=FixtureHireFlowClient(), embedder=embedder)
        with TestClient(app) as client:
            payload = {"question": case["question"]} if case["endpoint"] == "ask" else {"message": case["question"]}
            started = time.perf_counter()
            response = client.post(f"/v1/{case['endpoint']}", json=payload)
            seconds = time.perf_counter() - started
        if response.status_code != 200:
            rows.append({"id": case["id"], "passed": False, "problems": [f"HTTP {response.status_code}"]})
            continue
        body = response.json()
        cost = body["usage"]["cost_usd"]
        spent += cost
        problems = check(case, body)
        rows.append(
            {
                "id": case["id"],
                "passed": not problems,
                "problems": problems,
                "cost_usd": cost,
                "seconds": round(seconds, 1),
                "tools": [c["name"] for c in body.get("tool_calls", [])],
                "answer": body["answer"],
            }
        )

    passed = sum(r["passed"] for r in rows)
    print(f"Live eval: {passed}/{len(rows)} passed, ${spent:.4f} spent, model {settings.model}\n")
    print("| case | result | cost | time | tools |")
    print("|------|--------|------|------|-------|")
    for r in rows:
        result = "pass" if r["passed"] else "FAIL: " + "; ".join(r["problems"])
        print(
            f"| {r['id']} | {result} | ${r.get('cost_usd', 0):.4f} | {r.get('seconds', 0)}s "
            f"| {', '.join(r.get('tools', []))} |"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), "utf-8")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
