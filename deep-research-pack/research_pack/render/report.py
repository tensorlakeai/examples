"""Report synthesis — uses OpenAI to write the final Markdown report."""

from __future__ import annotations

import logging

import openai

logger = logging.getLogger("research_pack.render")


def synthesize_report(
    topic: str,
    sources: list[dict],
    model: str = "gpt-4o",
) -> str:
    """Return a Markdown research report citing the given *sources*.

    Each source dict is expected to have at least ``id``, ``url``,
    ``title``, ``summary_bullets``, ``key_quotes``, and
    ``reliability_notes``.
    """
    client = openai.OpenAI()  # reads OPENAI_API_KEY from env

    # build per-source context block
    blocks: list[str] = []
    for s in sources:
        if s.get("duplicate_of"):
            continue
        lines = [f"[{s['id']}] {s.get('title') or 'Untitled'} ({s['url']})"]
        for b in s.get("summary_bullets", []):
            lines.append(f"  - {b}")
        for q in s.get("key_quotes", []):
            lines.append(f'  > "{q["quote"]}"')
        rel = s.get("reliability_notes", "")
        if rel:
            lines.append(f"  Reliability: {rel}")
        blocks.append("\n".join(lines))

    sources_text = "\n\n---\n\n".join(blocks)

    prompt = (
        "Write a comprehensive, well-structured research report in Markdown.\n\n"
        f"TOPIC: {topic}\n\n"
        f"SOURCES:\n{sources_text}\n\n"
        "REQUIRED SECTIONS (use ## headers):\n"
        "1. Executive Summary (2-3 paragraphs)\n"
        "2. Key Findings (numbered list of 5-10 findings)\n"
        "3. Deep Dive (3-5 sub-sections with detailed analysis)\n"
        "4. Contradictions and Uncertainty\n"
        "5. What to Read Next (top 5 sources, each with a 1-sentence reason)\n\n"
        "CITATION RULES:\n"
        "- Every major claim MUST have an inline citation like [S1] or [S3].\n"
        "- Use only the source IDs provided above.\n"
        "- If information is missing or uncertain, say so explicitly.\n"
        "- Do NOT hallucinate facts beyond what the sources state.\n\n"
        "Output clean Markdown. Do NOT wrap the output in code fences."
    )

    try:
        response = client.chat.completions.create(
            model=model,
            max_tokens=8000,
            messages=[
                {"role": "system", "content": "You are a senior research analyst. Write thorough, well-cited reports."},
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content.strip()

    except Exception:
        logger.exception("report synthesis failed")
        md = f"# Research Report: {topic}\n\n"
        md += "*Report synthesis failed. Below are the sources that were collected.*\n\n"
        for s in sources:
            if s.get("duplicate_of"):
                continue
            md += f"- **[{s['id']}]** {s.get('title') or s['url']}\n"
            for b in s.get("summary_bullets", []):
                md += f"  - {b}\n"
        return md
