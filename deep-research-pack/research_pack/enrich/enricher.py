"""Source enrichment via OpenAI — summaries, reliability notes, key quotes."""

from __future__ import annotations

import json
import logging

import openai

logger = logging.getLogger("research_pack.enrich")

_MAX_CONTENT_CHARS = 30_000


def enrich_source(
    text: str,
    url: str,
    title: str = "",
    model: str = "gpt-4o",
) -> dict:
    """Call OpenAI to produce structured enrichment for one source.

    Returns a dict with keys: ``title``, ``author``, ``published_at``,
    ``summary_bullets``, ``reliability_notes``, ``key_quotes``, ``tags``.
    """
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env
    content = text[:_MAX_CONTENT_CHARS]

    prompt = (
        "Analyze this web source and provide structured metadata.\n\n"
        f"URL: {url}\n"
        f"Title: {title}\n\n"
        f"Content:\n{content}\n\n"
        "Respond with a JSON object containing EXACTLY these fields:\n"
        '- "title": string (refined page title)\n'
        '- "author": string or null\n'
        '- "published_at": string (ISO-8601) or null\n'
        '- "summary_bullets": list of 5-8 concise bullet-point strings\n'
        '- "reliability_notes": string (1-5 sentences on source strength/weakness)\n'
        '- "key_quotes": list of up to 3 objects each with '
        '"quote" (exact span from content), "start_offset" (int), "end_offset" (int)\n'
        '- "tags": list of 3-5 topic tags\n\n'
        "Return ONLY valid JSON. No markdown fences."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=2000,
            messages=[
                {"role": "system", "content": "You are a source analysis assistant. Always respond with valid JSON only."},
                {"role": "user", "content": prompt},
            ],
        )
        raw = response.choices[0].message.content.strip()

        # strip markdown fences if the model added them anyway
        if raw.startswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(
                lines[1 : -1 if lines[-1].strip().startswith("```") else len(lines)]
            )

        return json.loads(raw)

    except Exception:
        logger.exception("enrichment failed for %s", url)
        return {
            "title": title,
            "author": None,
            "published_at": None,
            "summary_bullets": ["Enrichment unavailable — source was fetched but could not be analysed."],
            "reliability_notes": "Automatic enrichment failed.",
            "key_quotes": [],
            "tags": [],
        }
