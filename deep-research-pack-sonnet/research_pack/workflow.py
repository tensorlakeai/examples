"""
Research Pack — core orchestration workflow.

This module is imported by both:
  - research_pack/app.py  (Tensorlake cloud execution)
  - research_pack/cli.py  (local execution via run_local_application)
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

from research_pack import llm
from research_pack.crawl.crawler import CrawledPage, crawl
from research_pack.dedupe.similarity import deduplicate
from research_pack.models import ResearchPlan, RunResult, SourceRecord
from research_pack.render.report import render_report
from research_pack.utils.fs import (
    make_output_dir,
    write_html_artifact,
    write_json,
    write_log,
    write_text_artifact,
)
from research_pack.utils.ids import new_run_id, source_id
from research_pack.utils.logging import get_logger, setup_logging

logger = get_logger("workflow")


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


class RunConfig:
    def __init__(
        self,
        topic: str,
        out_dir: Path,
        depth: int = 1,
        max_sources: int = 20,
        max_pages_per_domain: int = 3,
        offline: bool = False,
        keep_html: bool = True,
        rate_delay: float = 1.0,
        fetch_timeout: float = 20.0,
        run_id: Optional[str] = None,
    ):
        self.topic = topic
        self.out_dir = out_dir
        self.depth = depth
        self.max_sources = max_sources
        self.max_pages_per_domain = max_pages_per_domain
        self.offline = offline
        self.keep_html = keep_html
        self.rate_delay = rate_delay
        self.fetch_timeout = fetch_timeout
        self.run_id = run_id or new_run_id()


# ---------------------------------------------------------------------------
# Progress helper
# ---------------------------------------------------------------------------

ProgressFn = Callable[[str, int, int], None]


def _noop_progress(msg: str, current: int, total: int) -> None:
    pass


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_research(
    config: RunConfig,
    progress: ProgressFn = _noop_progress,
) -> RunResult:
    """
    Synchronous wrapper so Tensorlake @function can call this directly.
    Internally runs the async pipeline.
    """
    setup_logging()
    return asyncio.run(_run_async(config, progress))


async def _run_async(config: RunConfig, progress: ProgressFn) -> RunResult:
    """Full research pipeline — async implementation."""
    run_id = config.run_id
    out_dir = make_output_dir(config.out_dir, run_id)
    log_lines: list[str] = []

    def log(msg: str) -> None:
        ts = datetime.utcnow().isoformat()
        log_lines.append(f"[{ts}] {msg}")
        logger.info(msg)

    result = RunResult(
        run_id=run_id,
        topic=config.topic,
        status="running",
    )

    # ── Stage 1: Plan ────────────────────────────────────────────────────
    progress("Planning research strategy…", 0, 5)
    log(f"Stage 1: Planning for topic='{config.topic}'")

    if config.offline:
        plan = ResearchPlan(
            queries=[config.topic],
            seed_urls=[],
            parameters={"mode": "offline"},
        )
    else:
        plan = llm.generate_plan(config.topic)

    plan.parameters = {
        "depth": config.depth,
        "max_sources": config.max_sources,
        "max_pages_per_domain": config.max_pages_per_domain,
        "offline": config.offline,
    }
    result.plan = plan
    log(f"Plan: {len(plan.queries)} queries, {len(plan.seed_urls)} seed URLs")

    # Persist plan immediately
    write_json(out_dir / "run.json", result)

    # ── Stage 2: Acquire sources ─────────────────────────────────────────
    progress("Crawling sources…", 1, 5)
    log(f"Stage 2: Crawling {len(plan.seed_urls)} seed URLs (depth={config.depth})")

    crawled_pages: list[CrawledPage] = []

    if config.offline or not plan.seed_urls:
        log("Offline mode or no seed URLs — skipping crawl")
    else:
        crawled_pages = await crawl(
            seed_urls=plan.seed_urls,
            depth=config.depth,
            max_pages=config.max_sources * 2,  # fetch more to allow for dedup losses
            max_per_domain=config.max_pages_per_domain,
            rate_delay=config.rate_delay,
            timeout=config.fetch_timeout,
            keep_html=config.keep_html,
            progress=lambda msg, cur, tot: progress(f"Crawling: {msg}", 1, 5),
        )

    log(f"Crawled {len(crawled_pages)} pages")
    result.stats.fetched_count = len(crawled_pages)

    # Count failures
    failures = [p for p in crawled_pages if p.error]
    result.stats.failures_count = len(failures)

    # Useful pages (have text)
    useful = [p for p in crawled_pages if p.text and not p.pdf_text_unavailable]
    log(f"Useful pages (with text): {len(useful)}")

    # ── Stage 3: Normalize and dedupe ────────────────────────────────────
    progress("Deduplicating sources…", 2, 5)
    log("Stage 3: Deduplication")

    texts = [p.text for p in useful]
    dedup = deduplicate(texts)

    result.stats.duplicates_count = len(dedup.duplicate_map)
    log(
        f"Dedup complete: {len(dedup.kept_indices)} kept, "
        f"{len(dedup.duplicate_map)} duplicates removed"
    )

    # Build SourceRecord list — assign IDs
    all_records: dict[int, SourceRecord] = {}

    for idx, page in enumerate(useful):
        sid = source_id(idx + 1)
        rec = SourceRecord(
            id=sid,
            url=page.url,
            canonical_url=page.final_url if page.final_url != page.url else None,
            title=page.title or None,
            content_type=page.content_type,
            pdf_text_unavailable=page.pdf_text_unavailable if hasattr(page, "pdf_text_unavailable") else False,
        )
        all_records[idx] = rec

    # Mark duplicates
    for dup_idx, kept_idx in dedup.duplicate_map.items():
        if dup_idx in all_records and kept_idx in all_records:
            all_records[dup_idx].duplicate_of = all_records[kept_idx].id

    # ── Write text artifacts (for all pages, incl. duplicates) ───────────
    for idx, page in enumerate(useful):
        rec = all_records[idx]
        if page.text:
            rel = write_text_artifact(out_dir, rec.id, page.text)
            rec.text_path = rel
        if page.html and config.keep_html:
            rel = write_html_artifact(out_dir, rec.id, page.html)
            rec.raw_path = rel

    # Also record failed pages as sources with error
    for page in failures:
        if page.url not in {r.url for r in all_records.values()}:
            sid = source_id(len(all_records) + 1)
            rec = SourceRecord(
                id=sid,
                url=page.url,
                content_type=page.content_type or "unknown",
                fetch_error=page.error,
            )
            all_records[len(all_records)] = rec

    kept_records = [all_records[i] for i in dedup.kept_indices if i in all_records]
    kept_records = kept_records[: config.max_sources]

    result.stats.kept_count = len(kept_records)
    log(f"Sources to enrich: {len(kept_records)}")

    # ── Stage 4: Enrich (parallel) ────────────────────────────────────────
    progress("Enriching sources with AI…", 3, 5)
    log("Stage 4: Parallel source enrichment")

    async def _enrich_one(rec: SourceRecord, text: str) -> SourceRecord:
        loop = asyncio.get_event_loop()
        # Run blocking LLM call in thread pool
        return await loop.run_in_executor(None, llm.enrich_source, rec, text)

    enrichment_tasks = []
    for rec in kept_records:
        # Find the original text
        idx = next(
            (i for i, r in all_records.items() if r.id == rec.id), None
        )
        text = useful[idx].text if idx is not None and idx < len(useful) else ""
        enrichment_tasks.append(_enrich_one(rec, text))

    if not config.offline:
        enriched = await asyncio.gather(*enrichment_tasks, return_exceptions=True)
        for i, res in enumerate(enriched):
            if isinstance(res, Exception):
                logger.warning("Enrichment task %d failed: %s", i, res)
                # Keep original record
            else:
                kept_records[i] = res
    else:
        log("Offline mode — skipping enrichment")

    log("Stage 4 complete")

    # ── Assemble full source list for result ─────────────────────────────
    # kept + duplicates + failures
    dup_records = [
        all_records[i]
        for i in dedup.duplicate_map
        if i in all_records
    ]
    fail_records = [r for r in all_records.values() if r.fetch_error]
    result.sources = kept_records + dup_records + fail_records

    # ── Stage 5: Synthesize report ────────────────────────────────────────
    progress("Writing research report…", 4, 5)
    log("Stage 5: Synthesizing report")

    # Optional: LLM-written executive summary
    if not config.offline and kept_records:
        try:
            loop = asyncio.get_event_loop()
            summary = await loop.run_in_executor(
                None, llm.synthesize_report_intro, config.topic, kept_records
            )
            # Inject into render via a synthetic "meta" source tag
            result.plan.parameters["llm_summary"] = summary
        except Exception as exc:
            logger.warning("Synthesis failed: %s", exc)

    report_path = out_dir / "report.md"
    render_report(result, report_path)
    result.report_path = str(report_path)

    library_path = out_dir / "library.json"
    write_json(library_path, {"sources": [s.model_dump(mode="json") for s in result.sources]})
    result.library_path = str(library_path)

    # ── Finalize ──────────────────────────────────────────────────────────
    result.status = "done"
    result.finished_at = datetime.utcnow()
    write_json(out_dir / "run.json", result)
    write_log(out_dir, "run.log", log_lines)

    progress("Done!", 5, 5)
    log(
        f"Run complete: {result.stats.kept_count} sources kept, "
        f"{result.stats.duplicates_count} duplicates removed, "
        f"{result.stats.failures_count} failures"
    )

    return result
