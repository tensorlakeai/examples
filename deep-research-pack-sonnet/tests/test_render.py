"""Tests for report rendering."""

import json
import tempfile
from pathlib import Path
from datetime import datetime

import pytest

from research_pack.models import KeyQuote, ResearchPlan, RunResult, SourceRecord, RunStats
from research_pack.render.report import render_report


def _make_result() -> RunResult:
    src1 = SourceRecord(
        id="S1",
        url="https://example.com/article-one",
        title="Example Article One",
        content_type="text/html",
        summary_bullets=["Key point 1", "Key point 2", "Key point 3"],
        reliability_notes="This is a reputable source.",
        key_quotes=[KeyQuote(quote="This is a quote", start_offset=0, end_offset=15)],
        retrieved_at=datetime(2024, 1, 15),
    )
    src2 = SourceRecord(
        id="S2",
        url="https://other.org/page",
        title="Other Page",
        content_type="text/html",
        duplicate_of="S1",
        retrieved_at=datetime(2024, 1, 15),
    )
    return RunResult(
        run_id="run_test123",
        topic="test topic",
        plan=ResearchPlan(queries=["test query"], seed_urls=["https://example.com"]),
        stats=RunStats(fetched_count=2, kept_count=1, duplicates_count=1, failures_count=0),
        sources=[src1, src2],
    )


def test_render_creates_file():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.md"
        render_report(result, out_path)
        assert out_path.exists()
        content = out_path.read_text()
        assert len(content) > 100


def test_render_has_required_sections():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.md"
        render_report(result, out_path)
        content = out_path.read_text()
        assert "## Executive Summary" in content
        assert "## Key Findings" in content
        assert "## Deep Dive" in content
        assert "## What to Read Next" in content
        assert "## Source Library" in content


def test_render_contains_citation():
    result = _make_result()
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.md"
        render_report(result, out_path)
        content = out_path.read_text()
        assert "[S1]" in content


def test_render_empty_sources():
    result = RunResult(run_id="run_empty", topic="empty topic")
    with tempfile.TemporaryDirectory() as tmpdir:
        out_path = Path(tmpdir) / "report.md"
        render_report(result, out_path)
        content = out_path.read_text()
        assert "empty topic" in content
        # Should not crash
        assert len(content) > 0
