"""Tests for the orchestration workflow in offline mode (no network, no LLM)."""

import tempfile
from pathlib import Path

import pytest

from research_pack.workflow import RunConfig, run_research


def test_offline_run_produces_artifacts():
    """Full offline run — verifies artifact structure is created."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RunConfig(
            topic="test topic",
            out_dir=Path(tmpdir),
            offline=True,
            depth=0,
            max_sources=5,
        )
        result = run_research(config)

        assert result.status == "done"
        assert result.run_id.startswith("run_")
        assert result.topic == "test topic"

        out_dir = Path(tmpdir) / result.run_id
        assert (out_dir / "run.json").exists()
        assert (out_dir / "report.md").exists()
        assert (out_dir / "library.json").exists()
        assert (out_dir / "artifacts" / "sources").exists()
        assert (out_dir / "artifacts" / "logs").exists()


def test_offline_run_report_content():
    """Verify report.md has required sections in offline mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RunConfig(
            topic="AI safety research",
            out_dir=Path(tmpdir),
            offline=True,
        )
        result = run_research(config)

        report = (Path(tmpdir) / result.run_id / "report.md").read_text()
        assert "AI safety research" in report
        assert "## Executive Summary" in report
        assert "## Key Findings" in report


def test_offline_run_library_json():
    """Verify library.json is valid JSON with sources key."""
    import json

    with tempfile.TemporaryDirectory() as tmpdir:
        config = RunConfig(
            topic="test",
            out_dir=Path(tmpdir),
            offline=True,
        )
        result = run_research(config)

        lib_path = Path(tmpdir) / result.run_id / "library.json"
        data = json.loads(lib_path.read_text())
        assert "sources" in data
        assert isinstance(data["sources"], list)


def test_offline_run_stats():
    """Stats should be populated even in offline mode."""
    with tempfile.TemporaryDirectory() as tmpdir:
        config = RunConfig(
            topic="test",
            out_dir=Path(tmpdir),
            offline=True,
        )
        result = run_research(config)
        # In offline mode, fetched_count is 0 since we skip crawl
        assert result.stats.fetched_count == 0
        assert result.stats.failures_count == 0
