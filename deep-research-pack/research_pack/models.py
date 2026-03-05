"""Pydantic data models for the Research Pack pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field


class KeyQuote(BaseModel):
    """An exact text span extracted from a source."""

    quote: str
    start_offset: int
    end_offset: int


class SourceRecord(BaseModel):
    """Normalised metadata for a single acquired source."""

    id: str  # S1, S2, …
    url: str
    canonical_url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    content_type: str = "text/html"
    text_path: str = ""
    raw_path: Optional[str] = None
    summary_bullets: list[str] = Field(default_factory=list)
    reliability_notes: str = ""
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    duplicate_of: Optional[str] = None
    tags: list[str] = Field(default_factory=list)


class PlanResult(BaseModel):
    """Output of the planning stage."""

    queries: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)


class RunStats(BaseModel):
    """Aggregate statistics for a pipeline run."""

    fetched_count: int = 0
    kept_count: int = 0
    duplicates_count: int = 0
    failures_count: int = 0


class RunResult(BaseModel):
    """Full structured state for a completed run."""

    run_id: str
    topic: str
    plan: PlanResult
    stats: RunStats = Field(default_factory=RunStats)
    sources: list[SourceRecord] = Field(default_factory=list)
    report_path: str = ""
    library_path: str = ""


class CrawlResult(BaseModel):
    """Result from crawling a single URL."""

    url: str
    canonical_url: Optional[str] = None
    content_type: str = "unknown"
    status: Optional[int] = None
    error: Optional[str] = None
    title: str = ""
    author: str = ""
    published: str = ""
    text: str = ""
    html: str = ""
    links: list[str] = Field(default_factory=list)
    pdf_text_unavailable: bool = False
