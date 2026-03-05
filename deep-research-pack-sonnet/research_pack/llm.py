"""LLM interactions via OpenAI for planning, enrichment, and synthesis."""

from __future__ import annotations

import json
import os
import re
from typing import Optional

from openai import OpenAI

from research_pack.models import KeyQuote, ResearchPlan, SourceRecord
from research_pack.utils.logging import get_logger

logger = get_logger("llm")

_MODEL = "gpt-4o"
_MAX_TEXT_CHARS = 12_000  # truncate long source texts before sending to LLM


def _client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable not set. "
            "Export it before running research-pack."
        )
    return OpenAI(api_key=api_key)


def _parse_json(raw: str) -> dict:
    """Strip markdown fences and parse JSON."""
    raw = raw.strip()
    raw = re.sub(r"^```[a-z]*\n?", "", raw)
    raw = re.sub(r"\n?```$", "", raw)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Stage 1: Planning
# ---------------------------------------------------------------------------


def generate_plan(topic: str) -> ResearchPlan:
    """Ask the LLM to produce search queries and seed URLs for the topic."""
    prompt = f"""You are a research planning assistant.

Given the research topic below, produce a JSON object with:
- "queries": an array of 6 to 10 targeted search queries (strings) that would help gather comprehensive information on this topic
- "seed_urls": an array of 10 to 20 specific, real URLs to reputable pages, documentation, blogs, academic sources, or authoritative references about this topic

Only return valid JSON. No markdown fences. No commentary.

Topic: {topic}
"""
    logger.info("Generating research plan for topic: %s", topic)
    try:
        resp = _client().chat.completions.create(
            model=_MODEL,
            max_tokens=2048,
            messages=[
                {"role": "system", "content": "You are a research planning assistant. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = _parse_json(raw)
        plan = ResearchPlan(
            queries=data.get("queries", []),
            seed_urls=data.get("seed_urls", []),
        )
        logger.info(
            "Plan generated: %d queries, %d seed URLs",
            len(plan.queries), len(plan.seed_urls),
        )
        return plan
    except Exception as exc:
        logger.error("Plan generation failed: %s", exc)
        return ResearchPlan(queries=[], seed_urls=[])


# ---------------------------------------------------------------------------
# Stage 4: Per-source enrichment
# ---------------------------------------------------------------------------


def enrich_source(src: SourceRecord, text: str) -> SourceRecord:
    """
    Fill in summary_bullets, reliability_notes, and key_quotes for a source.
    Returns the same object mutated.
    """
    truncated = text[:_MAX_TEXT_CHARS]
    prompt = f"""You are a research analyst. Analyze the following web page content and return a JSON object with:

- "title": page title (string, or null if not determinable)
- "author": author name if mentioned (string, or null)
- "published_at": publication date if mentioned (ISO-8601 string, or null)
- "summary_bullets": list of 5 to 8 bullet point strings summarizing key insights
- "reliability_notes": 1 to 5 sentences assessing the reliability, credibility, and potential bias of this source
- "key_quotes": list of up to 3 objects, each with:
  - "quote": exact short quote from the text (under 200 chars)
  - "start_offset": character offset in the original text where the quote starts
  - "end_offset": character offset where it ends

Only return valid JSON. No markdown fences. No commentary.

Source URL: {src.url}
Source content (truncated to {_MAX_TEXT_CHARS} chars):
---
{truncated}
---
"""
    try:
        resp = _client().chat.completions.create(
            model=_MODEL,
            max_tokens=1500,
            messages=[
                {"role": "system", "content": "You are a research analyst. Return only valid JSON."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = resp.choices[0].message.content or ""
        data = _parse_json(raw)

        if not src.title and data.get("title"):
            src.title = data["title"]
        if not src.author and data.get("author"):
            src.author = data["author"]
        if not src.published_at and data.get("published_at"):
            src.published_at = data["published_at"]
        src.summary_bullets = data.get("summary_bullets", [])
        src.reliability_notes = data.get("reliability_notes", "")
        quotes_raw = data.get("key_quotes", [])
        src.key_quotes = [
            KeyQuote(
                quote=q.get("quote", ""),
                start_offset=int(q.get("start_offset", 0)),
                end_offset=int(q.get("end_offset", 0)),
            )
            for q in quotes_raw
            if q.get("quote")
        ]
        logger.debug(
            "Enriched %s: %d bullets, %d quotes",
            src.id, len(src.summary_bullets), len(src.key_quotes),
        )
    except Exception as exc:
        logger.warning("Enrichment failed for %s: %s", src.id, exc)
        src.summary_bullets = ["[Enrichment failed — see raw text artifact]"]
        src.reliability_notes = "Enrichment could not be completed for this source."

    return src


# ---------------------------------------------------------------------------
# Stage 5: Synthesis
# ---------------------------------------------------------------------------


def synthesize_report_intro(topic: str, sources: list[SourceRecord]) -> str:
    """
    Generate an LLM-written executive summary paragraph for the final report.
    Injected into the Markdown report renderer.
    """
    if not sources:
        return f"No sources were retrieved for topic: {topic}."

    brief_parts = []
    for src in sources[:10]:
        bullets = "\n".join(f"  - {b}" for b in src.summary_bullets[:3])
        brief_parts.append(f"[{src.id}] {src.url}\n{bullets}")
    brief = "\n\n".join(brief_parts)

    prompt = f"""You are a senior researcher. Write a concise executive summary (3 to 5 paragraphs, no headers)
for a deep research report on the topic below. Use the source summaries provided.
Cite sources inline like [S1], [S2], etc. Be precise. Do not invent facts not supported by the sources.
If information is uncertain or missing, say so explicitly.

Topic: {topic}

Source summaries:
{brief}
"""
    try:
        resp = _client().chat.completions.create(
            model=_MODEL,
            max_tokens=800,
            messages=[
                {"role": "system", "content": "You are a senior research analyst."},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.warning("Synthesis failed: %s", exc)
        return f"Research summary for **{topic}** based on {len(sources)} sources."
