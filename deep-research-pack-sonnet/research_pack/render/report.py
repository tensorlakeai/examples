"""Render a Markdown research report from enriched SourceRecords."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

from research_pack.models import RunResult, SourceRecord
from research_pack.utils.logging import get_logger

logger = get_logger("render")


def render_report(result: RunResult, out_path: Path) -> None:
    """Write report.md to out_path."""
    lines = _build_report(result)
    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Report written: %s", out_path)


def _build_report(result: RunResult) -> list[str]:
    topic = result.topic
    sources = [s for s in result.sources if not s.duplicate_of and not s.fetch_error]
    stats = result.stats

    lines: list[str] = []

    # ── Title ────────────────────────────────────────────────────────────────
    lines += [
        f"# Deep Research Report: {topic}",
        "",
        f"*Generated on {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} "
        f"| Run ID: `{result.run_id}`*",
        "",
        "---",
        "",
    ]

    # ── Executive Summary ─────────────────────────────────────────────────
    llm_summary = result.plan.parameters.get("llm_summary") if result.plan else None
    exec_summary = llm_summary if llm_summary else _executive_summary(topic, sources)
    lines += [
        "## Executive Summary",
        "",
        exec_summary,
        "",
    ]

    # ── Key Findings ──────────────────────────────────────────────────────
    lines += [
        "## Key Findings",
        "",
    ]
    for src in sources[:8]:
        if src.summary_bullets:
            lines.append(f"### From {_cite(src)} — {src.title or src.url}")
            for bullet in src.summary_bullets:
                lines.append(f"- {bullet}")
            lines.append("")

    # ── Deep Dive ─────────────────────────────────────────────────────────
    lines += [
        "## Deep Dive",
        "",
        _deep_dive(sources),
        "",
    ]

    # ── Key Quotes ───────────────────────────────────────────────────────
    has_quotes = any(s.key_quotes for s in sources)
    if has_quotes:
        lines += [
            "## Notable Quotes",
            "",
        ]
        for src in sources:
            for kq in src.key_quotes:
                lines.append(f"> \"{kq.quote}\"")
                lines.append(f">")
                lines.append(f"> — {_cite(src)}: {src.title or src.url}")
                lines.append("")

    # ── Contradictions and Uncertainty ───────────────────────────────────
    lines += [
        "## Contradictions and Uncertainty",
        "",
        (
            "The sources collected span multiple perspectives and were crawled automatically. "
            "Not all claims have been independently verified. Where sources conflict, the "
            "reader should consult the original documents directly. Sources with low "
            "reliability scores (see Source Library) should be treated with additional skepticism."
        ),
        "",
        "Information gaps identified during this research run:",
        "",
    ]
    if stats.fetched_count > 0 and stats.failures_count > 0:
        lines.append(
            f"- **{stats.failures_count} source(s)** could not be retrieved or were "
            f"inaccessible at the time of this run."
        )
    if stats.duplicates_count > 0:
        lines.append(
            f"- **{stats.duplicates_count} near-duplicate source(s)** were removed; "
            "they may contain additional nuance not captured here."
        )
    lines.append("")

    # ── What to Read Next ─────────────────────────────────────────────────
    top5 = sources[:5]
    lines += [
        "## What to Read Next",
        "",
        "These are the top-ranked sources from this research run:",
        "",
    ]
    for i, src in enumerate(top5, 1):
        title = src.title or src.url
        lines.append(f"{i}. **[{title}]({src.url})** {_cite(src)}")
        if src.reliability_notes:
            lines.append(f"   *{src.reliability_notes[:200]}*")
        lines.append("")

    # ── Source Library Reference ──────────────────────────────────────────
    lines += [
        "## Source Library",
        "",
        "Full metadata for all sources is available in `library.json`.",
        "",
        "| ID | Title | URL | Retrieved |",
        "|----|-------|-----|-----------|",
    ]
    for src in sources:
        title = (src.title or "—")[:50]
        url_short = src.url[:60] + ("…" if len(src.url) > 60 else "")
        date = src.retrieved_at.strftime("%Y-%m-%d") if src.retrieved_at else "—"
        lines.append(f"| {src.id} | {title} | {url_short} | {date} |")

    lines += [
        "",
        "---",
        "",
        "*This report was generated automatically by **Research Pack**. "
        "Always verify claims against primary sources.*",
    ]

    return lines


def _cite(src: SourceRecord) -> str:
    return f"[{src.id}]"


def _executive_summary(topic: str, sources: list[SourceRecord]) -> str:
    if not sources:
        return (
            f"This report covers the topic **{topic}**. "
            "No sources were successfully retrieved during this run. "
            "Please check network connectivity or try with different seed URLs."
        )
    n = len(sources)
    domains = list({s.url.split("/")[2] for s in sources if "/" in s.url})[:5]
    domain_str = ", ".join(domains)
    return (
        f"This report synthesizes findings on **{topic}** from {n} sources "
        f"spanning domains including {domain_str}. "
        "The sections below present key findings, deep-dive analysis, notable quotes, "
        "and a curated reading list."
    )


def _deep_dive(sources: list[SourceRecord]) -> str:
    if not sources:
        return "*No sources available for deep-dive analysis.*"

    paras: list[str] = []
    for src in sources:
        bullets = src.summary_bullets
        if not bullets:
            continue
        cite = _cite(src)
        title = src.title or src.url
        paras.append(f"### {title} {cite}")
        paras.append("")
        paras.append(
            f"Source reliability: *{src.reliability_notes or 'No reliability assessment available.'}*"
        )
        paras.append("")
        for b in bullets:
            paras.append(f"- {b}")
        paras.append("")

    return "\n".join(paras) if paras else "*No detailed summaries available.*"
