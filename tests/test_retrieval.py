import asyncio

from hireflow_copilot.hireflow.client import FixtureHireFlowClient
from hireflow_copilot.hireflow.models import SkillMatch
from hireflow_copilot.rag.documents import build_documents
from hireflow_copilot.rag.embeddings import HashingEmbedder, tokenize
from hireflow_copilot.rag.index import BM25, HybridIndex


def _index() -> HybridIndex:
    snapshot = asyncio.run(FixtureHireFlowClient().snapshot())
    return HybridIndex(build_documents(snapshot), HashingEmbedder())


def test_tokenizer_keeps_tech_names_whole() -> None:
    assert tokenize("C#, .NET and Node.js on CI/CD") == ["c#", ".net", "and", "node.js", "on", "ci", "cd"]


def test_bm25_prefers_rarer_terms() -> None:
    bm25 = BM25([["python", "sql"], ["python", "kafka"], ["python"]])
    scores = bm25.scores(["kafka"])
    assert scores[1] > 0 and scores[0] == scores[2] == 0


def test_documents_cover_every_record() -> None:
    snapshot = asyncio.run(FixtureHireFlowClient().snapshot())
    docs = build_documents(snapshot)
    kinds = [d.kind for d in docs]
    assert kinds.count("job") == len(snapshot.jobs)
    assert kinds.count("candidate") == len(snapshot.candidates)
    assert kinds.count("application") == len(snapshot.applications)
    assert len({d.id for d in docs}) == len(docs)


def test_interview_feedback_is_its_own_block() -> None:
    snapshot = asyncio.run(FixtureHireFlowClient().snapshot())
    elena = next(
        d for d in build_documents(snapshot) if d.title == "Application: Elena Ciobanu for Data Platform Engineer"
    )
    assert any("Best data-modelling answers" in b and "rated 5/5" in b for b in elena.blocks)


def test_hybrid_search_finds_exact_skill() -> None:
    top = _index().search("Kafka", k=3)
    assert "Elena Ciobanu" in top[0].document.title


def test_kind_filter() -> None:
    hits = _index().search("Python", k=10, kinds={"job"})
    assert hits and all(h.document.kind == "job" for h in hits)


def test_skill_match_matches_hireflow_rules() -> None:
    match = SkillMatch.evaluate(["C#", ".NET", "GraphQL"], ["c#", "SQL"])
    assert match.score == 0.333
    assert match.matched_skills == ["C#"]
    assert match.missing_skills == [".NET", "GraphQL"]
    assert SkillMatch.evaluate([], ["anything"]).score == 1.0


def test_web_demo_fixture_matches_the_board() -> None:
    snapshot = asyncio.run(FixtureHireFlowClient(fixture="web-demo").snapshot())
    assert len(snapshot.candidates) == 14
    ana = next(c for c in snapshot.candidates if c.full_name == "Ana Popescu")
    app = next(a for a in snapshot.applications if a.candidate_id == ana.id and a.stage == "INTERVIEW")
    # "+1d@14" in the export: tomorrow at 14:00 UTC, exactly as the web demo builds it.
    assert any(i.scheduled_at.hour == 14 for i in app.interviews)
