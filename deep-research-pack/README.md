# Research Pack

**Deep Research Pack** — an AI-powered research synthesis pipeline built on
[Tensorlake](https://tensorlake.ai).

Given a topic, Research Pack crawls the web, extracts and deduplicates content,
enriches each source with structured metadata via OpenAI, and synthesises a
Markdown report with inline citations backed by a machine-readable source
library.

---

## Quick start

### 1. Install

```bash
# Clone the repo and install in editable mode
git clone <repo-url> && cd research_pack
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

### 2. Set your API key

```bash
export OPENAI_API_KEY="sk-…"
```

### 3. Run

```bash
deeprp run "Impact of large language models on software engineering"
```

The CLI shows live progress and writes all artefacts to `./output/<run_id>/`.

---

## CLI reference

### `deeprp run`

```
deeprp run "<topic>" [OPTIONS]

Options:
  -o, --out DIR                Output root directory  [default: ./output]
  -d, --depth 1|2              Crawl depth            [default: 1]
  -n, --max-sources N          Max seed URLs to crawl [default: 20]
  --max-pages-per-domain N     Per-domain page cap    [default: 5]
  --offline                    Skip all LLM calls (pipeline test mode)
```

### `deeprp status <run_id>`

Print the status and stats of a previous run.

### `deeprp open <run_id>`

Print the paths to `report.md`, `library.json`, and the run directory.

---

## Output contract

Every run produces a folder at `<out>/<run_id>/` containing:

| File | Description |
|------|-------------|
| `report.md` | Markdown report with inline `[S1]` citations |
| `library.json` | Machine-readable source metadata |
| `run.json` | Full structured run state |
| `plan.json` | The generated research plan |
| `artifacts/sources/` | Extracted text per source (`S1.txt`, …) |
| `artifacts/html/` | Raw HTML snapshots |
| `artifacts/logs/` | Structured JSON log |

---

## Pipeline stages

| # | Stage | Description | Parallel? |
|---|-------|-------------|-----------|
| 1 | **Plan** | GPT-4o generates 6–10 search queries and 10–20 seed URLs | — |
| 2 | **Acquire** | Crawl seed URLs (depth 1–2), extract content | Yes |
| 3 | **Dedupe** | TF-IDF cosine similarity, threshold 0.85 | — |
| 4 | **Enrich** | GPT-4o produces summaries, reliability notes, key quotes | Yes |
| 5 | **Synthesise** | GPT-4o writes the final report with citations | — |

---

## Running on Tensorlake

Research Pack is also a first-class Tensorlake application. Each pipeline
stage is wrapped with `@function` for sandboxed container execution, and
stages 2 and 4 use Tensorlake Futures for parallel processing.

### Deploy

```bash
pip install tensorlake
tensorlake login
tensorlake deploy research_pack/app.py
```

### Invoke via HTTP

```bash
curl -X POST https://api.tensorlake.ai/applications/research_pipeline \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"topic": "Impact of AI on healthcare", "depth": 1, "max_sources": 10}'
```

### Run locally with Tensorlake runtime

```bash
python -m research_pack.app
```

This uses `run_local_application()` to execute the full Tensorlake graph
locally without deploying.

---

## Configuration knobs

| Knob | Default | Description |
|------|---------|-------------|
| `OPENAI_API_KEY` | (required) | OpenAI API key |
| `RESEARCH_PACK_OUT` | `./output` | Default output directory |
| `--depth` | `1` | Crawl depth (1 = seed page only, 2 = follow links one hop) |
| `--max-sources` | `20` | Maximum number of seed URLs to crawl |
| `--max-pages-per-domain` | `5` | Per-domain page cap during crawling |
| `--offline` | `False` | Skip all LLM calls |
| Dedup threshold | `0.85` | Cosine similarity threshold (edit in `dedupe/similarity.py`) |
| Rate limit | `1.0 s` | Minimum delay between requests to same domain |
| Fetch timeout | `15 s` | Per-request HTTP timeout |
| Fetch retries | `2` | Retry count with exponential back-off |

---

## Project layout

```
research_pack/
├── __init__.py
├── app.py              # Tensorlake @application / @function entry
├── cli.py              # Typer + Rich CLI
├── models.py           # Pydantic data models
├── workflow.py          # Core pipeline orchestrator
├── crawl/
│   ├── crawler.py      # Rate-limited, retrying web crawler
│   ├── extractor.py    # HTML → text via readability-lxml
│   └── robots.py       # Best-effort robots.txt
├── dedupe/
│   └── similarity.py   # TF-IDF cosine deduplication
├── enrich/
│   └── enricher.py     # Per-source OpenAI enrichment
├── render/
│   └── report.py       # Report synthesis via OpenAI
└── utils/
    ├── logging.py       # Structured JSON logging
    └── text.py          # Normalisation helpers
tests/
├── test_models.py
├── test_text.py
├── test_dedupe.py
└── test_extractor.py
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest -v
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `OPENAI_API_KEY is not set` | Export the key: `export OPENAI_API_KEY=sk-…` |
| Many fetch failures | The seed URLs generated by GPT-4o may not all be reachable. This is expected — the pipeline continues with whatever succeeded. |
| Empty report | Check `artifacts/logs/run.log` for errors. Try increasing `--max-sources`. |
| Rate-limited by a site | The crawler respects a 1 s per-domain delay. Reduce `--max-pages-per-domain` if needed. |
| `readability-lxml` import error | Ensure `lxml` C libraries are installed: `apt-get install libxml2-dev libxslt-dev` |
| Slow enrichment | Each source makes an OpenAI API call. Reduce `--max-sources` for faster runs. |
| `--offline` mode | Skips all LLM calls. Useful for testing crawl + dedupe mechanics without API costs. |

---

## Limitations

- **No real search API.** The planner generates seed URLs heuristically via
  GPT-4o. For production use, plug in a search provider (Tavily, Brave, etc.)
  by extending `workflow.generate_plan()`.
- **PDF text extraction** is not included. PDFs are downloaded and flagged as
  `pdf_text_unavailable`. Add a PDF parser (e.g. `pymupdf`) if needed.
- **robots.txt** parsing is simplified — it checks `Disallow` directives but
  does not handle `Allow`, wildcards, or `Crawl-delay`.
- **Single-machine parallelism** in CLI mode uses Python threads. For true
  distributed execution, deploy to Tensorlake.

---

## License

MIT
