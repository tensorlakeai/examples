"""Tensorlake application entry point.

Wraps the research pipeline stages as Tensorlake ``@function`` s with
parallel execution via Futures.  Deploy with::

    tensorlake deploy research_pack/app.py

Or run locally::

    python -m research_pack.app
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from tensorlake.applications import (
    Future,
    Image,
    RETURN_WHEN,
    RequestContext,
    application,
    function,
    run_local_application,
)

from research_pack.models import (
    CrawlResult,
    KeyQuote,
    PlanResult,
    RunResult,
    RunStats,
    SourceRecord,
)
from research_pack.utils.text import normalize_text

logger = logging.getLogger("research_pack.app")

# ---------------------------------------------------------------------------
# Container image shared by every function
# ---------------------------------------------------------------------------
research_image = Image(name="research-pack").run(
    "pip install openai httpx beautifulsoup4 readability-lxml lxml pydantic"
)


# ---------------------------------------------------------------------------
# Stage functions — each runs in its own sandboxed container
# ---------------------------------------------------------------------------


@function(image=research_image, secrets=["OPENAI_API_KEY"])
def plan_research(topic: str) -> str:
    """Generate search queries and seed URLs for *topic*."""
    from research_pack.workflow import generate_plan

    plan = generate_plan(topic)
    return plan.model_dump_json()


@function(image=research_image)
def crawl_url_task(input_json: str) -> str:
    """Crawl a single seed URL to the configured depth."""
    from research_pack.crawl.crawler import crawl_seed

    data = json.loads(input_json)
    results = crawl_seed(
        data["url"],
        max_depth=data.get("depth", 1),
        max_pages_per_domain=data.get("max_pages_per_domain", 5),
    )
    return json.dumps([r.model_dump() for r in results], default=str)


@function(image=research_image, secrets=["OPENAI_API_KEY"])
def enrich_source_task(input_json: str) -> str:
    """Enrich a single source with summaries, quotes and reliability notes."""
    from research_pack.enrich.enricher import enrich_source

    data = json.loads(input_json)
    result = enrich_source(data["text"], data["url"], data.get("title", ""))
    return json.dumps(result, default=str)


@function(image=research_image, secrets=["OPENAI_API_KEY"])
def write_report_task(input_json: str) -> str:
    """Synthesise the final Markdown report."""
    from research_pack.render.report import synthesize_report

    data = json.loads(input_json)
    return synthesize_report(data["topic"], data["sources"])


# ---------------------------------------------------------------------------
# Orchestrator — the main Tensorlake application
# ---------------------------------------------------------------------------


@application()
@function(image=research_image, secrets=["OPENAI_API_KEY"])
def research_pipeline(input_json: str) -> str:
    """Run the full five-stage deep-research pipeline.

    Accepts a JSON string with keys: ``topic``, ``depth`` (default 1),
    ``max_sources`` (default 20), ``max_pages_per_domain`` (default 5).
    """
    from research_pack.dedupe.similarity import find_duplicates

    ctx = RequestContext.get()
    config = json.loads(input_json)
    topic = config["topic"]
    depth = config.get("depth", 1)
    max_sources = config.get("max_sources", 20)
    max_ppd = config.get("max_pages_per_domain", 5)

    # ── Stage 1: Plan ─────────────────────────────────────────────────────
    ctx.progress.update(1, 5, "Planning research\u2026", {})
    plan_json = plan_research(topic)
    plan = PlanResult.model_validate_json(plan_json)

    # ── Stage 2: Crawl in parallel ────────────────────────────────────────
    seed_urls = plan.seed_urls[:max_sources]
    ctx.progress.update(
        2, 5, f"Crawling {len(seed_urls)} seed URLs\u2026", {}
    )

    crawl_futures: list[Future] = []
    for url in seed_urls:
        payload = json.dumps(
            {"url": url, "depth": depth, "max_pages_per_domain": max_ppd}
        )
        crawl_futures.append(crawl_url_task.future(payload).run())

    Future.wait(crawl_futures, return_when=RETURN_WHEN.ALL_COMPLETED)

    all_crawl: list[CrawlResult] = []
    failures = 0
    for fut in crawl_futures:
        try:
            items = json.loads(fut.result())
            for item in items:
                all_crawl.append(CrawlResult(**item))
        except Exception:
            failures += 1

    # Build SourceRecords
    records: list[SourceRecord] = []
    texts_for_dedupe: list[str] = []
    idx = 0
    for cr in all_crawl:
        if cr.error or not cr.text or len(cr.text.strip()) < 50:
            if cr.error:
                failures += 1
            continue
        idx += 1
        sid = f"S{idx}"
        records.append(
            SourceRecord(
                id=sid,
                url=cr.url,
                canonical_url=cr.canonical_url,
                title=cr.title or None,
                author=cr.author or None,
                published_at=cr.published or None,
                content_type=cr.content_type,
                text_path=f"(in-memory:{sid})",
            )
        )
        texts_for_dedupe.append(normalize_text(cr.text))

    fetched = len(records)

    # ── Stage 3: Dedupe ───────────────────────────────────────────────────
    ctx.progress.update(3, 5, "Deduplicating sources\u2026", {})
    dupes = find_duplicates(texts_for_dedupe)
    for dup_idx, orig_idx in dupes:
        records[dup_idx].duplicate_of = records[orig_idx].id
    dup_count = len(dupes)

    # ── Stage 4: Enrich in parallel ───────────────────────────────────────
    kept = [r for r in records if not r.duplicate_of]
    ctx.progress.update(
        4, 5, f"Enriching {len(kept)} sources\u2026", {}
    )

    text_lookup = {}
    ti = 0
    for cr in all_crawl:
        if cr.error or not cr.text or len(cr.text.strip()) < 50:
            continue
        ti += 1
        text_lookup[f"S{ti}"] = cr.text

    enrich_futures: list[tuple[str, Future]] = []
    for rec in kept:
        payload = json.dumps(
            {
                "text": (text_lookup.get(rec.id, ""))[:30000],
                "url": rec.url,
                "title": rec.title or "",
            }
        )
        enrich_futures.append(
            (rec.id, enrich_source_task.future(payload).run())
        )

    Future.wait(
        [f for _, f in enrich_futures],
        return_when=RETURN_WHEN.ALL_COMPLETED,
    )

    rec_map = {r.id: r for r in records}
    for sid, fut in enrich_futures:
        try:
            enrichment = json.loads(fut.result())
            r = rec_map[sid]
            r.title = enrichment.get("title", r.title)
            r.author = enrichment.get("author") or r.author
            r.published_at = enrichment.get("published_at") or r.published_at
            r.summary_bullets = enrichment.get("summary_bullets", [])
            r.reliability_notes = enrichment.get("reliability_notes", "")
            r.key_quotes = [
                KeyQuote(**q) for q in enrichment.get("key_quotes", [])
            ]
            r.tags = enrichment.get("tags", [])
        except Exception:
            pass

    # ── Stage 5: Synthesise report ────────────────────────────────────────
    ctx.progress.update(5, 5, "Writing report\u2026", {})
    report_payload = json.dumps(
        {"topic": topic, "sources": [r.model_dump() for r in records]}
    )
    report_md = write_report_task(report_payload)

    stats = RunStats(
        fetched_count=fetched,
        kept_count=fetched - dup_count,
        duplicates_count=dup_count,
        failures_count=failures,
    )

    result = RunResult(
        run_id=datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S"),
        topic=topic,
        plan=plan,
        stats=stats,
        sources=records,
        report_path="(returned inline)",
        library_path="(returned inline)",
    )

    output = {
        "run": result.model_dump(),
        "report_md": report_md,
        "library": [r.model_dump() for r in records],
    }
    return json.dumps(output, indent=2, default=str)


# ---------------------------------------------------------------------------
# Local testing
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    config = json.dumps(
        {"topic": "Impact of AI on healthcare", "depth": 1, "max_sources": 5}
    )
    print("Starting local run\u2026")
    request = run_local_application(research_pipeline, config)
    print(request.output())
