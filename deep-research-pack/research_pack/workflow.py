"""Core pipeline orchestrator.

Called directly by the CLI (local mode) and wrapped by ``app.py``
for Tensorlake deployment.  Uses :pymod:`concurrent.futures` for
local parallelism.
"""

from __future__ import annotations

import json
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import openai

from research_pack.crawl.crawler import crawl_seed
from research_pack.dedupe.similarity import find_duplicates
from research_pack.enrich.enricher import enrich_source
from research_pack.models import (
    CrawlResult,
    KeyQuote,
    PlanResult,
    RunResult,
    RunStats,
    SourceRecord,
)
from research_pack.render.report import synthesize_report
from research_pack.utils.logging import get_logger
from research_pack.utils.text import normalize_text

ProgressFn = Optional[Callable[[str, str], None]]

# ---------------------------------------------------------------------------
# Stage 1 — Planning
# ---------------------------------------------------------------------------


def generate_plan(topic: str, model: str = "gpt-4o") -> PlanResult:
    """Use OpenAI to produce search queries and seed URLs for *topic*."""
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    prompt = (
        "You are a research planner. Given a topic, generate:\n"
        "1. 6-10 targeted search queries to research this topic thoroughly\n"
        "2. 10-20 seed URLs that are likely to contain high-quality information\n"
        "   Use a mix: official documentation, reputable blogs, academic sources, "
        "vendor pages, Wikipedia, news outlets.\n\n"
        f"Topic: {topic}\n\n"
        "Respond with a JSON object:\n"
        '{"queries": ["…"], "seed_urls": ["https://…"]}\n\n'
        "Requirements:\n"
        "- Seed URLs MUST be real, well-known URLs likely to exist and return content.\n"
        "- Prefer HTTPS.\n"
        "- Return ONLY valid JSON. No markdown fences."
    )
    response = client.chat.completions.create(
        model=model,
        max_tokens=2500,
        messages=[
            {"role": "system", "content": "You are a helpful research planner. Always respond with valid JSON only."},
            {"role": "user", "content": prompt},
        ],
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        raw = "\n".join(
            lines[1 : -1 if lines[-1].strip().startswith("```") else len(lines)]
        )
    data = json.loads(raw)
    return PlanResult(
        queries=data.get("queries", []),
        seed_urls=data.get("seed_urls", []),
    )


# ---------------------------------------------------------------------------
# Stage 2 — Acquisition (parallel)
# ---------------------------------------------------------------------------


def _crawl_one(
    url: str, depth: int, max_pages_per_domain: int
) -> list[CrawlResult]:
    """Thin wrapper so we can submit to ThreadPoolExecutor."""
    try:
        return crawl_seed(url, max_depth=depth, max_pages_per_domain=max_pages_per_domain)
    except Exception as exc:
        return [CrawlResult(url=url, error=str(exc))]


def acquire_sources(
    seed_urls: list[str],
    depth: int = 1,
    max_pages_per_domain: int = 5,
    max_sources: int = 20,
    progress: ProgressFn = None,
    max_workers: int = 6,
) -> list[CrawlResult]:
    """Crawl *seed_urls* in parallel with a thread pool."""
    urls = seed_urls[:max_sources]
    all_results: list[CrawlResult] = []
    total = len(urls)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_crawl_one, url, depth, max_pages_per_domain): url
            for url in urls
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            url = futures[future]
            if progress:
                progress("crawling", f"[{done_count}/{total}] {url}")
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as exc:
                all_results.append(CrawlResult(url=url, error=str(exc)))

    return all_results


# ---------------------------------------------------------------------------
# Stage 3 — Normalise & dedupe
# ---------------------------------------------------------------------------


def normalise_and_dedupe(
    records: list[SourceRecord],
    base: Path,
) -> tuple[list[SourceRecord], int]:
    """Normalise text and mark near-duplicates.

    Returns ``(all_records_with_dupe_flags, duplicate_count)``.
    """
    texts: list[str] = []
    for rec in records:
        text_path = base / rec.text_path
        raw = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
        texts.append(normalize_text(raw))

    dupes = find_duplicates(texts)
    for dup_idx, orig_idx in dupes:
        records[dup_idx].duplicate_of = records[orig_idx].id

    return records, len(dupes)


# ---------------------------------------------------------------------------
# Stage 4 — Enrichment (parallel)
# ---------------------------------------------------------------------------


def _enrich_one(source: SourceRecord, base: Path) -> SourceRecord:
    text_path = base / source.text_path
    text = text_path.read_text(encoding="utf-8") if text_path.exists() else ""
    enrichment = enrich_source(text, source.url, source.title or "")
    source.title = enrichment.get("title", source.title)
    source.author = enrichment.get("author") or source.author
    source.published_at = enrichment.get("published_at") or source.published_at
    source.summary_bullets = enrichment.get("summary_bullets", [])
    source.reliability_notes = enrichment.get("reliability_notes", "")
    source.key_quotes = [KeyQuote(**q) for q in enrichment.get("key_quotes", [])]
    source.tags = enrichment.get("tags", [])
    return source


def enrich_sources(
    sources: list[SourceRecord],
    base: Path,
    progress: ProgressFn = None,
    max_workers: int = 4,
) -> list[SourceRecord]:
    """Enrich non-duplicate sources in parallel."""
    kept = [s for s in sources if not s.duplicate_of]
    total = len(kept)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_enrich_one, s, base): s for s in kept
        }
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            s = futures[future]
            if progress:
                progress("enriching", f"[{done_count}/{total}] {s.url}")
            try:
                future.result()  # updates source in-place
            except Exception as exc:
                s.reliability_notes = f"Enrichment failed: {exc}"

    return sources  # return full list (including dupes)


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------


def run_pipeline(
    topic: str,
    out_dir: str = "./output",
    depth: int = 1,
    max_sources: int = 20,
    max_pages_per_domain: int = 5,
    offline: bool = False,
    progress_callback: ProgressFn = None,
) -> RunResult:
    """Execute the complete five-stage research pipeline.

    Parameters
    ----------
    topic:
        The research question / topic.
    out_dir:
        Root output directory.  A timestamped sub-folder is created.
    depth:
        Maximum crawl depth (1 or 2).
    max_sources:
        Cap on the number of seed URLs to crawl.
    max_pages_per_domain:
        Per-domain page limit while crawling.
    offline:
        If *True*, skip all LLM calls (useful for testing the pipeline
        mechanics without burning API credits).
    progress_callback:
        ``(stage, detail) -> None`` — called whenever the pipeline
        transitions or makes progress.
    """
    run_id = (
        datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        + "_"
        + uuid.uuid4().hex[:8]
    )
    base = Path(out_dir) / run_id
    sources_dir = base / "artifacts" / "sources"
    html_dir = base / "artifacts" / "html"
    logs_dir = base / "artifacts" / "logs"

    for d in (sources_dir, html_dir, logs_dir):
        d.mkdir(parents=True, exist_ok=True)

    log = get_logger(log_dir=logs_dir)

    def _p(stage: str, detail: str = "") -> None:
        log.info("%s: %s", stage, detail)
        if progress_callback:
            progress_callback(stage, detail)

    # ── Stage 1: Plan ─────────────────────────────────────────────────────
    _p("planning", "Generating research plan\u2026")
    if offline:
        plan = PlanResult(
            queries=[f"{topic} overview", f"{topic} analysis", f"{topic} research"],
            seed_urls=[],
            parameters={"offline": True},
        )
    else:
        plan = generate_plan(topic)

    plan.parameters = {
        "depth": depth,
        "max_sources": max_sources,
        "max_pages_per_domain": max_pages_per_domain,
    }
    (base / "plan.json").write_text(json.dumps(plan.model_dump(), indent=2))
    _p(
        "planning",
        f"Plan ready — {len(plan.queries)} queries, {len(plan.seed_urls)} seed URLs",
    )

    # ── Stage 2: Acquire ──────────────────────────────────────────────────
    _p("crawling", f"Crawling {len(plan.seed_urls)} seed URLs\u2026")
    all_crawl = acquire_sources(
        seed_urls=plan.seed_urls,
        depth=depth,
        max_pages_per_domain=max_pages_per_domain,
        max_sources=max_sources,
        progress=_p,
    )

    # persist raw artefacts and build SourceRecords
    source_records: list[SourceRecord] = []
    source_idx = 0
    failures = 0

    for cr in all_crawl:
        if cr.error:
            failures += 1
            continue
        if not cr.text or len(cr.text.strip()) < 50:
            continue

        source_idx += 1
        sid = f"S{source_idx}"

        text_rel = f"artifacts/sources/{sid}.txt"
        (base / text_rel).write_text(cr.text, encoding="utf-8")

        raw_rel: str | None = None
        if cr.html:
            raw_rel = f"artifacts/html/{sid}.html"
            (base / raw_rel).write_text(cr.html, encoding="utf-8")

        source_records.append(
            SourceRecord(
                id=sid,
                url=cr.url,
                canonical_url=cr.canonical_url,
                title=cr.title or None,
                author=cr.author or None,
                published_at=cr.published or None,
                content_type=cr.content_type,
                text_path=text_rel,
                raw_path=raw_rel,
            )
        )

    fetched_count = len(source_records)
    _p("crawling", f"Fetched {fetched_count} sources with usable content")

    # ── Stage 3: Normalise & dedupe ───────────────────────────────────────
    _p("deduplicating", "Finding near-duplicate sources\u2026")
    source_records, dup_count = normalise_and_dedupe(source_records, base)
    kept_count = fetched_count - dup_count
    _p("deduplicating", f"Kept {kept_count}, removed {dup_count} duplicates")

    # ── Stage 4: Enrich ───────────────────────────────────────────────────
    if not offline and source_records:
        _p("enriching", f"Enriching {kept_count} sources\u2026")
        source_records = enrich_sources(source_records, base, progress=_p)

    # ── Stage 5: Synthesise report ────────────────────────────────────────
    _p("writing", "Synthesising report\u2026")
    if offline:
        report_md = f"# Research Report: {topic}\n\n"
        report_md += "*Generated in offline mode — no AI synthesis.*\n\n## Sources\n\n"
        for s in source_records:
            if not s.duplicate_of:
                report_md += f"- **[{s.id}]** {s.title or s.url}\n"
    else:
        sources_data = [s.model_dump() for s in source_records]
        report_md = synthesize_report(topic, sources_data)

    # ── Write final artefacts ─────────────────────────────────────────────
    report_path = base / "report.md"
    report_path.write_text(report_md, encoding="utf-8")

    library_data = [s.model_dump() for s in source_records]
    library_path = base / "library.json"
    library_path.write_text(
        json.dumps(library_data, indent=2, default=str), encoding="utf-8"
    )

    stats = RunStats(
        fetched_count=fetched_count,
        kept_count=kept_count,
        duplicates_count=dup_count,
        failures_count=failures,
    )

    run_result = RunResult(
        run_id=run_id,
        topic=topic,
        plan=plan,
        stats=stats,
        sources=source_records,
        report_path=str(report_path),
        library_path=str(library_path),
    )

    run_json_path = base / "run.json"
    run_json_path.write_text(
        json.dumps(run_result.model_dump(), indent=2, default=str), encoding="utf-8"
    )

    _p("done", f"Complete! Output: {base}")
    return run_result
