"""Pydantic data models for Research Pack."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class KeyQuote(BaseModel):
    quote: str
    start_offset: int
    end_offset: int


class RunStats(BaseModel):
    fetched_count: int = 0
    kept_count: int = 0
    duplicates_count: int = 0
    failures_count: int = 0


class ResearchPlan(BaseModel):
    queries: list[str] = Field(default_factory=list)
    seed_urls: list[str] = Field(default_factory=list)
    parameters: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SourceRecord
# ---------------------------------------------------------------------------


class SourceRecord(BaseModel):
    id: str  # S1, S2, ...
    url: str
    canonical_url: Optional[str] = None
    title: Optional[str] = None
    author: Optional[str] = None
    published_at: Optional[str] = None
    retrieved_at: datetime = Field(default_factory=datetime.utcnow)
    content_type: str = "text/html"
    text_path: Optional[str] = None   # relative path inside output dir
    raw_path: Optional[str] = None    # optional raw html
    summary_bullets: list[str] = Field(default_factory=list)
    reliability_notes: str = ""
    key_quotes: list[KeyQuote] = Field(default_factory=list)
    duplicate_of: Optional[str] = None  # source id
    tags: list[str] = Field(default_factory=list)
    # Internal crawl metadata — not part of user-visible library
    fetch_error: Optional[str] = None
    pdf_text_unavailable: bool = False


# ---------------------------------------------------------------------------
# RunResult
# ---------------------------------------------------------------------------


class RunResult(BaseModel):
    run_id: str
    topic: str
    plan: ResearchPlan = Field(default_factory=ResearchPlan)
    stats: RunStats = Field(default_factory=RunStats)
    sources: list[SourceRecord] = Field(default_factory=list)
    report_path: Optional[str] = None
    library_path: Optional[str] = None
    started_at: datetime = Field(default_factory=datetime.utcnow)
    finished_at: Optional[datetime] = None
    status: str = "pending"  # pending | running | done | failed
