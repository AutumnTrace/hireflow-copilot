from fastapi.testclient import TestClient

from tests.conftest import APP_ANA, APP_VICTORIA, llm_client, message


def test_health_reports_demo_mode(demo_client: TestClient) -> None:
    body = demo_client.get("/health").json()
    assert body["mode"] == "demo"
    assert body["documents"] > 20


def test_search_returns_ranked_documents(demo_client: TestClient) -> None:
    hits = demo_client.post("/v1/search", json={"query": "Playwright", "k": 3}).json()
    assert len(hits) == 3
    assert hits[0]["source"].startswith("hireflow://")


def test_ask_in_demo_mode_is_extractive(demo_client: TestClient) -> None:
    body = demo_client.post("/v1/ask", json={"question": "Who knows Kafka?"}).json()
    assert body["mode"] == "demo"
    assert body["usage"]["cost_usd"] == 0
    assert "Elena" in body["sources"][0]["title"]


def test_agent_needs_a_key(demo_client: TestClient) -> None:
    assert demo_client.post("/v1/agent", json={"message": "hi"}).status_code == 503


def test_question_length_is_capped(demo_client: TestClient) -> None:
    assert demo_client.post("/v1/ask", json={"question": "x" * 1001}).status_code == 422


def test_ask_sends_search_results_and_renders_citations() -> None:
    cited = {
        "type": "text",
        "text": "Elena Ciobanu has Kafka experience.",
        "citations": [
            {
                "type": "search_result_location",
                "cited_text": "Skills: Python, Kafka, SQL.",
                "source": "hireflow://candidates/x",
                "title": "Candidate: Elena Ciobanu",
                "search_result_index": 0,
                "start_block_index": 1,
                "end_block_index": 2,
            }
        ],
    }
    client, fake = llm_client([message([cited])])
    with client:
        body = client.post("/v1/ask", json={"question": "Who knows Kafka?"}).json()

    assert body["answer"] == "Elena Ciobanu has Kafka experience.[1]"
    assert body["sources"][0]["quotes"] == ["Skills: Python, Kafka, SQL."]
    assert body["sources"][0]["candidate_id"]  # lets a UI link the citation to the candidate page
    request = fake.messages.requests[0]
    blocks = request["messages"][0]["content"]
    assert blocks[0]["type"] == "search_result" and blocks[0]["citations"] == {"enabled": True}
    assert blocks[-1] == {"type": "text", "text": "Who knows Kafka?"}
    assert request["fallbacks"] == "default"
    assert request["max_tokens"] == 4096
    assert "temperature" not in request


def test_rate_limit_returns_429_with_retry_after() -> None:
    client, _ = llm_client([message([{"type": "text", "text": "ok"}])] * 3, rate_limit_per_minute=2)
    with client:
        assert client.post("/v1/ask", json={"question": "a"}).status_code == 200
        assert client.post("/v1/ask", json={"question": "b"}).status_code == 200
        blocked = client.post("/v1/ask", json={"question": "c"})
    assert blocked.status_code == 429
    assert int(blocked.headers["retry-after"]) > 0


def test_daily_budget_stops_llm_calls() -> None:
    # One call of 1000 in / 200 out on Opus 5.5 costs $0.008, above a $0.005 budget.
    client, _ = llm_client([message([{"type": "text", "text": "ok"}])] * 2, daily_budget_usd=0.005)
    with client:
        assert client.post("/v1/ask", json={"question": "a"}).status_code == 200
        assert client.post("/v1/ask", json={"question": "b"}).status_code == 429
        usage = client.get("/v1/usage").json()
    assert usage["spent_usd"] == 0.008


def test_access_key_is_enforced() -> None:
    client, _ = llm_client([message([{"type": "text", "text": "ok"}])], access_key="s3cret")
    with client:
        assert client.post("/v1/ask", json={"question": "a"}).status_code == 401
        ok = client.post("/v1/ask", json={"question": "a"}, headers={"X-Copilot-Key": "s3cret"})
    assert ok.status_code == 200


def test_agent_proposes_then_human_confirms() -> None:
    responses = [
        message(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "propose_stage_change",
                    "input": {"application_id": APP_VICTORIA, "stage": "INTERVIEW", "note": "Great screen"},
                }
            ],
            stop_reason="tool_use",
        ),
        message(
            [{"type": "text", "text": "I proposed moving Victoria to INTERVIEW. It is pending your confirmation."}]
        ),
    ]
    client, fake = llm_client(responses)
    with client:
        body = client.post("/v1/agent", json={"message": "Move Victoria forward"}).json()
        assert body["stop"] == "end_turn"
        assert body["steps"] == 2
        action = body["proposed_actions"][0]
        assert "Victoria Sandu from SCREENING to INTERVIEW" in action["summary"]
        assert action["kind"] == "move_application"
        assert action["params"] == {"application_id": APP_VICTORIA, "stage": "INTERVIEW", "note": "Great screen"}

        kb = client.app.state.services.kb  # type: ignore[attr-defined]
        assert kb.snapshot.application(APP_VICTORIA).stage == "SCREENING"  # proposing changed nothing

        done = client.post(action["confirm_url"]).json()
        assert done == {"id": action["id"], "status": "done", "result": "Application is now INTERVIEW."}
        assert kb.snapshot.application(APP_VICTORIA).stage == "INTERVIEW"  # index rebuilt after the write
        again = client.post(action["confirm_url"]).json()
        assert again["status"] == "done"

    # Second request carried the tool result back in a single user message.
    second = fake.messages.requests[1]["messages"]
    assert second[-1]["role"] == "user"
    assert second[-1]["content"][0]["tool_use_id"] == "toolu_1"


def test_agent_gets_tool_errors_as_is_error() -> None:
    responses = [
        message(
            [
                {
                    "type": "tool_use",
                    "id": "toolu_1",
                    "name": "propose_stage_change",
                    "input": {"application_id": APP_ANA, "stage": "HIRED"},
                }
            ],
            stop_reason="tool_use",
        ),
        message([{"type": "text", "text": "Ana is in INTERVIEW, so she can only move to OFFER next."}]),
    ]
    client, fake = llm_client(responses)
    with client:
        body = client.post("/v1/agent", json={"message": "Hire Ana"}).json()
    assert body["tool_calls"] == [
        {"name": "propose_stage_change", "input": {"application_id": APP_ANA, "stage": "HIRED"}, "is_error": True}
    ]
    assert body["proposed_actions"] == []
    result = fake.messages.requests[1]["messages"][-1]["content"][0]
    assert result["is_error"] is True
    assert "Allowed next stages: OFFER, REJECTED, WITHDRAWN" in result["content"]


def test_agent_search_citations_index_into_sent_documents() -> None:
    responses = [
        message(
            [{"type": "tool_use", "id": "toolu_1", "name": "search_hireflow", "input": {"query": "Kafka"}}],
            stop_reason="tool_use",
        ),
        message(
            [
                {
                    "type": "text",
                    "text": "Elena knows Kafka.",
                    "citations": [
                        {
                            "type": "search_result_location",
                            "cited_text": "Skills: Python, Kafka, SQL.",
                            "source": "hireflow://candidates/30000000-0000-4000-8000-000000000003",
                            "title": "Candidate: Elena Ciobanu",
                            "search_result_index": 0,
                            "start_block_index": 1,
                            "end_block_index": 2,
                        }
                    ],
                }
            ]
        ),
    ]
    client, fake = llm_client(responses)
    with client:
        body = client.post("/v1/agent", json={"message": "Who knows Kafka?"}).json()
    sent = fake.messages.requests[1]["messages"][-1]["content"][0]["content"]
    assert all(block["type"] == "search_result" for block in sent)
    assert body["sources"][0]["source"] == sent[0]["source"]
    assert body["answer"].endswith("[1]")


def test_agent_stops_at_step_limit() -> None:
    loop = message(
        [{"type": "tool_use", "id": "toolu_x", "name": "list_jobs", "input": {}}],
        stop_reason="tool_use",
    )
    client, fake = llm_client([loop, loop], max_agent_steps=2)
    with client:
        body = client.post("/v1/agent", json={"message": "loop forever"}).json()
    assert body["stop"] == "max_steps"
    assert len(fake.messages.requests) == 2
    assert "Stopped after 2 steps" in body["answer"]


def test_cors_allows_the_web_app(demo_client: TestClient) -> None:
    response = demo_client.options(
        "/v1/ask",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
    blocked = demo_client.options(
        "/v1/ask", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}
    )
    assert "access-control-allow-origin" not in blocked.headers
