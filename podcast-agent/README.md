# 🎙️ Podcast Agent (HackerNews Podcast Generator)

A Tensorlake-powered workflow that crawls an article URL, summarizes the content with Gemini, and produces a podcast-ready MP3 using ElevenLabs.

## Project Metadata

- **Name**: Podcast Agent (HackerNews Podcast Generator)
- **Author**: Arindam Majumder
- **Platform**: Tensorlake Applications

## Features

- **Crawl + clean**: Depth-first crawl with a Chromium-based scraper (PyDoll)
- **Summarize**: Gemini generates a concise podcast script
- **Narrate**: ElevenLabs converts the script to audio
- **UI**: Streamlit interface for easy testing

## Architecture

- **Entry point**: `podcast_agent` Tensorlake application
- **Steps**: crawl → extract_clean_text → summarize_with_gemini → generate_audio
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
