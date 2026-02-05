# 🎙️ Podcast Agent (HackerNews Top 3)

A Tensorlake-powered workflow that fetches the top 3 Hacker News articles, summarizes each with Gemini, and produces a ZIP containing three podcast-ready MP3s using ElevenLabs.

## Features

- **Auto fetch**: Top 3 Hacker News articles
- **Crawl + clean**: Depth-first crawl with a Chromium-based scraper (PyDoll)
- **Summarize**: Gemini generates concise 5-line podcast scripts
- **Narrate**: ElevenLabs converts the scripts to audio
- **UI**: Streamlit interface for easy testing

## Architecture

- **Entry point**: `podcast_agent` Tensorlake application
- **Steps**: fetch_hackernews_top_article → crawl → extract_clean_text → summarize_with_gemini → generate_audio
- **Secrets**: `GEMINI_API_KEY`, `ELEVENLABS_API_KEY`

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt --upgrade
```

### 2. Configure API keys

Create a `.env` file (or export environment variables) with:

```bash
GEMINI_API_KEY=your_gemini_key
ELEVENLABS_API_KEY=your_elevenlabs_key
```

### 3. Run locally

```bash
python podcast_agent.py
```

This will generate `podcasts.zip` with 3 MP3 files.

### 4. Run the Streamlit UI

```bash
streamlit run app.py
```

## Deploy to Tensorlake Cloud

```bash
tensorlake login
tensorlake secrets set GEMINI_API_KEY=your_gemini_key
tensorlake secrets set ELEVENLABS_API_KEY=your_elevenlabs_key
tensorlake deploy podcast_agent.py
```

## Invoke the deployed app

```bash
python - <<'PY'
from tensorlake.applications import run_remote_application
from podcast_agent import CrawlInput

request = run_remote_application(
    "podcast_agent",
    CrawlInput(url="https://news.ycombinator.com", max_depth=1, max_links=1),
)
print(request.output())
PY
```

## Project Structure

```
.
├── podcast_agent.py    # Tensorlake application + functions
├── app.py              # Streamlit UI
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
