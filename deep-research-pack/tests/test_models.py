"""Tests for Pydantic data models."""

from research_pack.models import (
    CrawlResult,
    KeyQuote,
    PlanResult,
    RunResult,
    RunStats,
    SourceRecord,
)


def test_source_record_defaults():
    rec = SourceRecord(id="S1", url="https://example.com")
    assert rec.id == "S1"
    assert rec.duplicate_of is None
    assert rec.summary_bullets == []
    assert rec.tags == []
    assert rec.content_type == "text/html"


def test_source_record_with_quotes():
    rec = SourceRecord(
        id="S2",
        url="https://example.com/page",
        key_quotes=[
            KeyQuote(quote="hello world", start_offset=0, end_offset=11),
        ],
    )
    assert len(rec.key_quotes) == 1
    assert rec.key_quotes[0].quote == "hello world"


def test_plan_result():
    plan = PlanResult(
        queries=["q1", "q2"],
        seed_urls=["https://a.com", "https://b.com"],
    )
    assert len(plan.queries) == 2
    assert plan.parameters == {}


def test_run_result_serialisation():
    rr = RunResult(
        run_id="test_001",
        topic="test topic",
        plan=PlanResult(queries=["q1"], seed_urls=["https://a.com"]),
        stats=RunStats(fetched_count=5, kept_count=4, duplicates_count=1),
    )
    d = rr.model_dump()
    assert d["run_id"] == "test_001"
    assert d["stats"]["kept_count"] == 4

    # round-trip
    rr2 = RunResult(**d)
    assert rr2.topic == "test topic"


def test_crawl_result_error():
    cr = CrawlResult(url="https://fail.com", error="timeout")
    assert cr.error == "timeout"
    assert cr.text == ""
