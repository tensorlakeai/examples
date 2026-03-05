"""
Tensorlake Application entry point for Research Pack.

Deploy:
    tensorlake deploy research_pack/app.py

HTTP endpoints:
    POST /applications/research_pack_run      → trigger a run
    GET  /applications/research_pack_run/requests/{id} → fetch status/results

Local test:
    python -m research_pack.app
"""

import logging
import os
from pathlib import Path

from pydantic import BaseModel
from tensorlake.applications import (
    Future,
    Image,
    RequestContext,
    Retries,
    application,
    function,
    run_local_application,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Container Images
# ---------------------------------------------------------------------------

# Crawler image: needs httpx, bs4, pdfminer, readability, protego, tldextract
crawler_image = (
    Image(name="research-pack-crawler")
    .copy("research_pack", "/app/research_pack")
    .env("PYTHONPATH", "/app")
    .run("pip install httpx beautifulsoup4 lxml readability-lxml pdfminer.six protego tldextract anyio")
)

# Enrichment image: needs openai + scikit-learn
enrichment_image = (
    Image(name="research-pack-enrichment")
    .copy("research_pack", "/app/research_pack")
    .env("PYTHONPATH", "/app")
    .run("pip install openai scikit-learn numpy pydantic python-dateutil")
)

# Orchestrator image: full dependencies
orchestrator_image = (
    Image(name="research-pack-orchestrator")
    .copy("research_pack", "/app/research_pack")
    .env("PYTHONPATH", "/app")
    .run(
        "pip install openai httpx beautifulsoup4 lxml readability-lxml "
        "pdfminer.six protego tldextract anyio scikit-learn numpy pydantic "
        "python-dateutil tensorlake typer rich"
    )
)

# ---------------------------------------------------------------------------
# Input / Output schemas
# ---------------------------------------------------------------------------


class ResearchRequest(BaseModel):
    topic: str
    depth: int = 1
    max_sources: int = 20
    max_pages_per_domain: int = 3
    offline: bool = False
    out_dir: str = "/tmp/research_pack_runs"


class ResearchResponse(BaseModel):
    run_id: str
    status: str
    topic: str
    sources_kept: int = 0
    sources_fetched: int = 0
    duplicates_removed: int = 0
    failures: int = 0
    report_path: str | None = None
    library_path: str | None = None


# ---------------------------------------------------------------------------
# Tensorlake Functions
# ---------------------------------------------------------------------------


@function(
    image=crawler_image,
    cpu=2,
    memory=4,
    timeout=600,
    retries=Retries(max_retries=1),
    ephemeral_disk=10,
)
def crawl_sources(
    seed_urls: list[str],
    depth: int,
    max_pages: int,
    max_per_domain: int,
    rate_delay: float,
    timeout: float,
    keep_html: bool,
) -> list[dict]:
    """
    Sandboxed crawl function.
    Returns a list of serialised CrawledPage dicts.
    """
    import asyncio

    from research_pack.crawl.crawler import crawl
    from research_pack.utils.logging import setup_logging

    setup_logging()
    ctx = RequestContext.get()

    async def _run():
        pages = await crawl(
            seed_urls=seed_urls,
            depth=depth,
            max_pages=max_pages,
            max_per_domain=max_per_domain,
            rate_delay=rate_delay,
            timeout=timeout,
            keep_html=keep_html,
            progress=lambda msg, cur, tot: ctx.progress.update(cur, tot, msg),
        )
        return [
            {
                "url": p.url,
                "final_url": p.final_url,
                "title": p.title,
                "text": p.text,
                "html": p.html,
                "content_type": p.content_type,
                "pdf_text_unavailable": p.pdf_text_unavailable,
                "error": p.error,
                "depth": p.depth,
            }
            for p in pages
        ]

    return asyncio.run(_run())


@function(
    image=enrichment_image,
    cpu=1,
    memory=2,
    timeout=120,
    retries=Retries(max_retries=2),
    secrets=["OPENAI_API_KEY"],
)
def enrich_one_source(source_dict: dict, text: str) -> dict:
    """
    Sandboxed enrichment function for a single SourceRecord.
    Returns updated source dict.
    """
    from research_pack.models import SourceRecord
    from research_pack import llm

    rec = SourceRecord(**source_dict)
    enriched = llm.enrich_source(rec, text)
    return enriched.model_dump(mode="json")


@function(
    image=enrichment_image,
    cpu=1,
    memory=2,
    timeout=60,
    secrets=["OPENAI_API_KEY"],
)
def generate_plan_remote(topic: str) -> dict:
    """Generate a research plan remotely."""
    from research_pack import llm

    plan = llm.generate_plan(topic)
    return plan.model_dump(mode="json")


@function(
    image=enrichment_image,
    cpu=1,
    memory=2,
    timeout=60,
    secrets=["OPENAI_API_KEY"],
)
def synthesize_intro_remote(topic: str, sources_dicts: list[dict]) -> str:
    """Generate LLM executive summary remotely."""
    from research_pack import llm
    from research_pack.models import SourceRecord

    sources = [SourceRecord(**d) for d in sources_dicts]
    return llm.synthesize_report_intro(topic, sources)


# ---------------------------------------------------------------------------
# Application entry point — orchestrator
# ---------------------------------------------------------------------------


@application()
@function(
    image=orchestrator_image,
    cpu=2,
    memory=8,
    timeout=1800,
    ephemeral_disk=20,
    secrets=["OPENAI_API_KEY"],
)
def research_pack_run(request: ResearchRequest) -> ResearchResponse:
    """
    Main Tensorlake application function.

    Orchestrates:
      1) Plan generation (remote sandboxed)
      2) Parallel crawl (remote sandboxed)
      3) Dedup (in-process)
      4) Parallel enrichment (remote sandboxed via .map())
      5) Report synthesis and render
    """
    import json
    from pathlib import Path
    from datetime import datetime

    from research_pack.dedupe.similarity import deduplicate
    from research_pack.models import ResearchPlan, RunResult, SourceRecord
    from research_pack.render.report import render_report
    from research_pack.utils.fs import (
        make_output_dir,
        write_html_artifact,
        write_json,
        write_text_artifact,
    )
    from research_pack.utils.ids import new_run_id, source_id
    from research_pack.utils.logging import setup_logging

    setup_logging()
    ctx = RequestContext.get()
    run_id = new_run_id()
    out_dir = make_output_dir(Path(request.out_dir), run_id)

    result = RunResult(run_id=run_id, topic=request.topic, status="running")
    write_json(out_dir / "run.json", result)

    # ── Stage 1: Plan ────────────────────────────────────────────────────
    ctx.progress.update(0, 5, "Generating research plan…")
    if request.offline:
        plan_dict = {"queries": [request.topic], "seed_urls": [], "parameters": {}}
    else:
        plan_dict = generate_plan_remote(request.topic)

    plan = ResearchPlan(**plan_dict)
    plan.parameters = {
        "depth": request.depth,
        "max_sources": request.max_sources,
        "offline": request.offline,
    }
    result.plan = plan
    write_json(out_dir / "run.json", result)

    # ── Stage 2: Crawl (parallel, sandboxed) ─────────────────────────────
    ctx.progress.update(1, 5, f"Crawling {len(plan.seed_urls)} sources…")

    pages_data: list[dict] = []
    if not request.offline and plan.seed_urls:
        # Launch crawl in sandbox; in a full Tensorlake deployment this
        # runs in an isolated container
        pages_data = crawl_sources(
            seed_urls=plan.seed_urls,
            depth=request.depth,
            max_pages=request.max_sources * 2,
            max_per_domain=request.max_pages_per_domain,
            rate_delay=1.0,
            timeout=20.0,
            keep_html=True,
        )

    result.stats.fetched_count = len(pages_data)
    result.stats.failures_count = sum(1 for p in pages_data if p.get("error"))

    # ── Stage 3: Dedup ───────────────────────────────────────────────────
    ctx.progress.update(2, 5, "Deduplicating sources…")

    useful = [p for p in pages_data if p.get("text") and not p.get("pdf_text_unavailable")]
    texts = [p["text"] for p in useful]
    dedup = deduplicate(texts)

    result.stats.duplicates_count = len(dedup.duplicate_map)

    # Build SourceRecord objects
    all_records: dict[int, SourceRecord] = {}
    for idx, page in enumerate(useful):
        sid = source_id(idx + 1)
        rec = SourceRecord(
            id=sid,
            url=page["url"],
            canonical_url=page["final_url"] if page["final_url"] != page["url"] else None,
            title=page.get("title") or None,
            content_type=page.get("content_type", "text/html"),
        )
        # Write text artifact
        if page.get("text"):
            rel = write_text_artifact(out_dir, sid, page["text"])
            rec.text_path = rel
        if page.get("html"):
            rel = write_html_artifact(out_dir, sid, page["html"])
            rec.raw_path = rel
        all_records[idx] = rec

    # Mark duplicates
    for dup_idx, kept_idx in dedup.duplicate_map.items():
        if dup_idx in all_records and kept_idx in all_records:
            all_records[dup_idx].duplicate_of = all_records[kept_idx].id

    kept_records = [
        all_records[i] for i in dedup.kept_indices if i in all_records
    ][: request.max_sources]

    result.stats.kept_count = len(kept_records)

    # ── Stage 4: Parallel enrichment via .map() ───────────────────────────
    ctx.progress.update(3, 5, f"Enriching {len(kept_records)} sources…")

    if not request.offline and kept_records:
        source_dicts = [r.model_dump(mode="json") for r in kept_records]
        source_texts = [
            useful[next(i for i, r in all_records.items() if r.id == rec.id)]["text"]
            for rec in kept_records
        ]

        # Parallel enrichment using Tensorlake Futures
        from tensorlake.applications import Future, RETURN_WHEN

        futures: list[Future] = [
            enrich_one_source.future(sd, st).run()
            for sd, st in zip(source_dicts, source_texts)
        ]
        Future.wait(futures, return_when=RETURN_WHEN.ALL_COMPLETED)
        enriched_dicts: list[dict] = [f.result() for f in futures]
        kept_records = [SourceRecord(**d) for d in enriched_dicts]

    # ── Stage 5: Report synthesis ─────────────────────────────────────────
    ctx.progress.update(4, 5, "Writing report…")

    if not request.offline and kept_records:
        try:
            summary = synthesize_intro_remote(
                request.topic,
                [r.model_dump(mode="json") for r in kept_records],
            )
            result.plan.parameters["llm_summary"] = summary
        except Exception as exc:
            logger.warning("Synthesis failed: %s", exc)

    # Assemble full source list
    dup_records = [all_records[i] for i in dedup.duplicate_map if i in all_records]
    fail_records = [
        SourceRecord(
            id=source_id(len(all_records) + j + 1),
            url=p["url"],
            content_type=p.get("content_type", "unknown"),
            fetch_error=p["error"],
        )
        for j, p in enumerate(pages_data)
        if p.get("error") and p["url"] not in {r.url for r in all_records.values()}
    ]
    result.sources = kept_records + dup_records + fail_records

    # Render
    report_path = out_dir / "report.md"
    render_report(result, report_path)
    result.report_path = str(report_path)

    library_path = out_dir / "library.json"
    write_json(library_path, {"sources": [s.model_dump(mode="json") for s in result.sources]})
    result.library_path = str(library_path)

    result.status = "done"
    result.finished_at = datetime.utcnow()
    write_json(out_dir / "run.json", result)

    ctx.progress.update(5, 5, "Research complete!")

    return ResearchResponse(
        run_id=run_id,
        status="done",
        topic=request.topic,
        sources_kept=result.stats.kept_count,
        sources_fetched=result.stats.fetched_count,
        duplicates_removed=result.stats.duplicates_count,
        failures=result.stats.failures_count,
        report_path=str(report_path),
        library_path=str(library_path),
    )


# ---------------------------------------------------------------------------
# Local test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tensorlake.applications import run_local_application

    req = ResearchRequest(
        topic="quantum computing recent advances",
        depth=1,
        max_sources=5,
        offline=False,
    )
    request = run_local_application(research_pack_run, req)
    output = request.output()
    print(output)
