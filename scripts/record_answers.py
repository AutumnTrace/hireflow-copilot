"""Record real copilot answers for HireFlow Web's offline demo.

The web demo runs entirely in the browser and has no copilot server, so it replays these
recordings, labelled as recorded. They are made against the ``web-demo`` fixture, the same data
the demo board shows, so names, stages and application ids line up.

    uv run python scripts/record_answers.py ../hireflow-web/src/copilot/recorded.json --budget 0.50
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from fastapi.testclient import TestClient

from hireflow_copilot.config import Settings
from hireflow_copilot.hireflow.client import FixtureHireFlowClient
from hireflow_copilot.main import create_app
from hireflow_copilot.rag.embeddings import FastEmbedEmbedder

QUESTIONS = [
    ("shortlist", "agent", "Who are the strongest candidates for the Senior .NET Engineer role, and why?"),
    ("week", "agent", "What interviews are coming up this week?"),
    ("rejection", "ask", "Why was Nicoleta Turcan rejected for the Senior .NET Engineer job?"),
    ("frontend-fit", "agent", "Which candidates who haven't applied would fit the Frontend Engineer (React) job?"),
    (
        "advance-ana",
        "agent",
        "Ana Popescu's phone screen went well. Move her Senior .NET application to the offer stage.",
    ),
    ("hire-ana", "agent", "Mark Ana Popescu as hired."),
]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("out", type=Path)
    parser.add_argument("--budget", type=float, default=0.50)
    args = parser.parse_args()

    settings = Settings(
        fixture="web-demo",
        daily_budget_usd=args.budget,
        rate_limit_per_minute=1000,
        rate_limit_per_day=1000,
        otel_exporter="none",
    )
    if settings.demo_mode:
        print("Set ANTHROPIC_API_KEY (environment or .env) to record answers.", file=sys.stderr)
        return 2
    embedder = FastEmbedEmbedder(settings.embedding_model)

    items, spent = [], 0.0
    for item_id, endpoint, question in QUESTIONS:
        # A fresh dataset per question, so one recording's proposals can't affect the next.
        app = create_app(settings, hireflow=FixtureHireFlowClient(fixture="web-demo"), embedder=embedder)
        with TestClient(app) as client:
            payload = {"question": question} if endpoint == "ask" else {"message": question}
            response = client.post(f"/v1/{endpoint}", json=payload)
        if response.status_code != 200:
            print(f"{item_id}: HTTP {response.status_code} {response.text}", file=sys.stderr)
            return 1
        body = response.json()
        spent += body["usage"]["cost_usd"]
        items.append({"id": item_id, "endpoint": endpoint, "question": question, "response": body})
        print(f"{item_id}: ${body['usage']['cost_usd']:.4f}")

    recording = {
        "recordedAt": datetime.now(UTC).isoformat(timespec="seconds"),
        "model": settings.model,
        "items": items,
    }
    args.out.write_text(json.dumps(recording, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(f"Wrote {len(items)} recordings to {args.out} (${spent:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
