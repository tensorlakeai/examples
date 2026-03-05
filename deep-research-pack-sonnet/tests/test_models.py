"""Tests for Pydantic data models."""

import pytest
from datetime import datetime
from research_pack.models import KeyQuote, RunResult, RunStats, SourceRecord, ResearchPlan


def test_source_record_defaults():
    rec = SourceRecord(id="S1", url="https://example.com", content_type="text/html")
    assert rec.id == "S1"
    assert rec.duplicate_of is None
    assert rec.summary_bullets == []
    assert rec.key_quotes == []
    assert rec.tags == []
    assert isinstance(rec.retrieved_at, datetime)


def test_key_quote():
    kq = KeyQuote(quote="hello world", start_offset=0, end_offset=11)
    assert kq.quote == "hello world"
    assert kq.start_offset == 0
    assert kq.end_offset == 11


def test_run_result_defaults():
    rr = RunResult(run_id="run_abc", topic="test topic")
    assert rr.status == "pending"
    assert rr.sources == []
    assert rr.stats.fetched_count == 0


def test_run_result_serialization():
    rr = RunResult(run_id="run_abc", topic="test topic")
    data = rr.model_dump(mode="json")
    assert data["run_id"] == "run_abc"
    assert data["topic"] == "test topic"
    assert isinstance(data["started_at"], str)


def test_research_plan():
    plan = ResearchPlan(
        queries=["q1", "q2"],
        seed_urls=["https://example.com"],
    )
    assert len(plan.queries) == 2
    assert len(plan.seed_urls) == 1
