# Research Pack

**Deep Research Pack** — a production-grade AI research pipeline that generates high-quality Markdown reports with inline citations and a machine-readable source library JSON. Built on [Tensorlake](https://tensorlake.ai), powered by OpenAI (`gpt-4o`).

---

## What it does

Given a topic prompt, Research Pack:

1. **Plans** — asks the LLM to generate 6–10 search queries and 10–20 seed URLs
2. **Crawls** — fetches all seed pages in parallel (BFS to configurable depth), extracts clean text, handles PDFs
3. **Deduplicates** — removes near-duplicate sources via TF-IDF cosine similarity
4. **Enriches** — asks the LLM for summary bullets, reliability notes, and key quotes per source
5. **Synthesizes** — writes a structured `report.md` with inline `[S3]`-style citations and a `library.json` source index

### Output structure

```
~/.research_pack/runs/<run_id>/
├── report.md          # Full research report with citations
├── library.json       # Machine-readable source library
├── run.json           # Full structured run state (RunResult)
└── artifacts/
    ├── sources/       # Extracted plain text per source (S1.txt, S2.txt, ...)
    ├── html/          # Optional raw HTML per source
    └── logs/          # run.log
```

---

## Setup

### Prerequisites

- Python 3.11+
- An [OpenAI API key](https://platform.openai.com/api-keys) (`gpt-4o`)
- (Optional) A [Tensorlake account](https://cloud.tensorlake.ai) for cloud deployment

### Install locally

```bash
# Clone / enter the repo
git clone <repo_url>
cd research_pack

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# Install in editable mode
pip install -e .

# Set your OpenAI key
export OPENAI_API_KEY=sk-...
```

---

## Running locally

### Basic run

```bash
research-pack run "large language model alignment techniques"
```

### With options

```bash
research-pack run "quantum computing hardware 2024" \
  --depth 1 \
  --max-sources 15 \
  --max-pages-per-domain 3 \
  --out ./my_research
```

### Offline mode (structure test, no network/LLM)

```bash
research-pack run "some topic" --offline
```

### Check a previous run

```bash
research-pack status run_a1b2c3d4e5f6
```

### Open artifacts for a run

```bash
research-pack open run_a1b2c3d4e5f6
```

---

## Running on Tensorlake

### Prerequisites

```bash
pip install tensorlake
tensorlake login
tensorlake secrets set OPENAI_API_KEY sk-...
```

### Deploy

```bash
tensorlake deploy research_pack/app.py
```

### Trigger a run via HTTP

```bash
curl https://api.tensorlake.ai/applications/research_pack_run \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  --json '{
    "topic": "quantum computing hardware advances",
    "depth": 1,
    "max_sources": 20
  }'
```

Response:

```json
{ "request_id": "req_abc123" }
```

### Check status

```bash
curl https://api.tensorlake.ai/applications/research_pack_run/requests/req_abc123 \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY"
```

### Fetch result via Python SDK

```python
from tensorlake.applications import get_remote_request

request = get_remote_request("research_pack_run", "req_abc123")
print(request.output())
```

### Replay a failed run (durable execution)

```python
request = get_remote_request("research_pack_run", "req_abc123")
request.replay()
```

---

## CLI reference

```
research-pack run "<topic>" [OPTIONS]

  Options:
    --out, -o PATH              Output directory (default: ~/.research_pack/runs)
    --depth, -d INT             Crawl depth: 0=seeds only, 1=+1 hop, 2=+2 hops [default: 1]
    --max-sources, -n INT       Max sources to keep after dedup [default: 20]
    --max-pages-per-domain INT  Max pages per domain [default: 3]
    --offline                   Skip network/LLM (structure test)
    --verbose, -v               Verbose logging

research-pack status <run_id> [--out PATH]
research-pack open   <run_id> [--out PATH]
```

---

## Configuration knobs

| CLI option               | Default | Description                                 |
| ------------------------ | ------- | ------------------------------------------- |
| `--depth`                | 1       | BFS crawl depth (0=seeds only, 2=2 hops)    |
| `--max-sources`          | 20      | Max sources to enrich and include in report |
| `--max-pages-per-domain` | 3       | Per-domain page cap (politeness)            |
| `--offline`              | false   | Skip all network and LLM calls              |

### Environment variables

| Variable             | Required          | Description                  |
| -------------------- | ----------------- | ---------------------------- |
| `OPENAI_API_KEY`  | Yes (non-offline) | OpenAI API key (`gpt-4o`)    |
| `TENSORLAKE_API_KEY` | For cloud deploy  | Tensorlake platform key      |

---

## Architecture

### How Tensorlake concepts map to this project

| Tensorlake concept              | Research Pack usage                                                           |
| ------------------------------- | ----------------------------------------------------------------------------- |
| `@application()`                | `research_pack_run` — the single HTTP-triggerable entry point                 |
| `@function(image=...)`          | `crawl_sources`, `enrich_one_source` — sandboxed in dedicated containers      |
| `.map()` parallel execution     | Parallel per-source enrichment in Stage 4                                     |
| `RequestContext.progress`       | Emits stage-level progress: planning → crawling → dedup → enrichment → report |
| Durable execution               | Long runs checkpoint via Tensorlake; `request.replay()` resumes failed runs   |
| `Retries`                       | `enrich_one_source` has `max_retries=2`; `crawl_sources` has `max_retries=1`  |
| `secrets=["OPENAI_API_KEY"]` | API key injected securely into enrichment containers                          |
| Custom `Image`                  | Separate crawler/enrichment images with minimal dependencies                  |

### Module layout

```
research_pack/
├── __init__.py
├── app.py          Tensorlake application entry point
├── cli.py          CLI (Typer + Rich)
├── workflow.py     Local orchestrator (mirrors app.py logic without Tensorlake decorators)
├── models.py       Pydantic data models: SourceRecord, RunResult, etc.
├── llm.py          OpenAI calls: planning, enrichment, synthesis
├── crawl/
│   ├── crawler.py  BFS async crawler
│   ├── extract.py  HTML/PDF content extraction
│   ├── fetcher.py  Rate-limited async HTTP fetcher
│   └── robots.py   Best-effort robots.txt checker
├── dedupe/
│   └── similarity.py  TF-IDF cosine deduplication
├── render/
│   └── report.py   Markdown report renderer
└── utils/
    ├── fs.py       Filesystem helpers
    ├── ids.py      Run/source ID generation
    └── logging.py  Structured logging setup
```

---

## Data models

### SourceRecord

```python
SourceRecord(
    id="S3",
    url="https://arxiv.org/abs/2401.12345",
    canonical_url=None,
    title="Scaling Laws for Neural Language Models",
    author="Kaplan et al.",
    published_at="2020-01-23",
    retrieved_at=datetime(...),
    content_type="text/html",
    text_path="artifacts/sources/S3.txt",
    raw_path="artifacts/html/S3.html",
    summary_bullets=["Key finding 1", "Key finding 2", ...],
    reliability_notes="Peer-reviewed, widely cited.",
    key_quotes=[KeyQuote(quote="...", start_offset=42, end_offset=120)],
    duplicate_of=None,
    tags=[],
)
```

### RunResult

```python
RunResult(
    run_id="run_a1b2c3d4e5f6",
    topic="...",
    plan=ResearchPlan(queries=[...], seed_urls=[...]),
    stats=RunStats(fetched_count=18, kept_count=12, duplicates_count=3, failures_count=3),
    sources=[...],
    report_path="/path/to/report.md",
    library_path="/path/to/library.json",
)
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

Expected output with all tests passing:

```
tests/test_dedupe.py      ......    6 passed
tests/test_extract.py     .......   7 passed
tests/test_ids.py         ...       3 passed
tests/test_models.py      .....     5 passed
tests/test_render.py      ....      4 passed
tests/test_workflow_offline.py .... 4 passed
```

---

## Troubleshooting

### `OPENAI_API_KEY is not set`

Export the key: `export OPENAI_API_KEY=sk-...`

### `ModuleNotFoundError: No module named 'research_pack'`

Run `pip install -e .` from the repo root.

### Crawl returns 0 sources

- The seed URLs generated by the LLM may be inaccessible. Try `--offline` to verify the pipeline structure.
- Some sites block bots. Check `artifacts/logs/run.log` for per-URL errors.
- Try `--depth 0` to fetch only seeds (no link following).

### Report is sparse / missing sections

- Enrichment requires a valid `OPENAI_API_KEY`.
- In offline mode, no LLM enrichment occurs — bullets and summaries will be empty.

### Rate limit errors from OpenAI

- The enrichment runs in parallel. If you hit rate limits, reduce `--max-sources`.
- The `Retries(max_retries=2)` on enrichment functions handles transient errors.

### `tensorlake deploy` fails

- Ensure `tensorlake login` has been run and your key is set.
- The `Image(...).run(...)` builder installs all deps in the container; first deploy may take a few minutes.

---

## Limitations

- **No search API integration**: Uses LLM-generated seed URLs, not live web search results. Seeds may occasionally be stale or incorrect. The `--offline` flag documents this clearly.
- **PDF extraction**: Best-effort via `pdfminer.six`. Complex/scanned PDFs may yield poor text; they are stored as `pdf_text_unavailable=true` in the library.
- **robots.txt**: Best-effort; requires `protego` package. If unavailable, all pages are crawled.
- **Content freshness**: Crawled at run time; no caching between runs.
- **LLM hallucinations**: The report renderer is designed to cite only from retrieved sources. However, the LLM executive summary step (`synthesize_report_intro`) may occasionally generalize beyond the provided sources. Always verify claims.

---

## License

MIT
