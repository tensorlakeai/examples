"""
Web Crawler with Depth-First Search - Tensorlake + PyDoll

Auto-fetches the top 3 Hacker News articles, crawls each,
summarizes the content (5 lines each), and generates
separate podcast audio files.
"""

import asyncio
import base64
import json
import mimetypes
import platform
from urllib.parse import urlparse

from dotenv import load_dotenv

load_dotenv()

from pydantic import BaseModel, Field
from tensorlake.applications import Image, Retries, application, function, File

# ---------------------------------------------------------------------
# Image with Chromium and pydoll for web scraping function
# ---------------------------------------------------------------------
scraper_image = (
    Image(name="web-scraper-image", base_image="python:3.11.0")
    .env("DEBIAN_FRONTEND", "noninteractive")
    .run(
        """apt-get update && \
apt-get install -y \
gnupg \
wget \
iproute2 \
wkhtmltopdf \
libx11-xcb1 \
libdbus-glib-1-2 \
git \
tini \
chromium"""
    )
    .run("pip install --upgrade pip wheel")
    .run("pip install pydoll-python tensorlake requests beautifulsoup4")
    .run("apt-get clean && rm -rf /var/lib/apt/lists/*")
    .run("pip cache purge")
)

# ---------------------------------------------------------------------
# File extensions that should be base64 encoded
# ---------------------------------------------------------------------
BINARY_EXTENSIONS = {
    ".pdf",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".svg",
    ".zip",
    ".tar",
    ".gz",
    ".rar",
    ".7z",
    ".doc",
    ".docx",
    ".xls",
    ".xlsx",
    ".ppt",
    ".pptx",
    ".mp3",
    ".mp4",
    ".wav",
    ".avi",
    ".mov",
    ".webm",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".otf",
}


# ---------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------
class CrawlInput(BaseModel):
    url: str = Field(description="Starting URL (ignored, auto-fetched)")
    max_depth: int = Field(default=1, ge=0)
    max_links: int = Field(default=1, ge=1)


# ---------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------
def is_binary_url(url: str) -> bool:
    return any(urlparse(url).path.lower().endswith(ext) for ext in BINARY_EXTENSIONS)


def get_file_type(url: str) -> str:
    mime_type, _ = mimetypes.guess_type(urlparse(url).path)
    return mime_type or "application/octet-stream"


def is_same_domain(base_url: str, target_url: str) -> bool:
    return urlparse(base_url).netloc == urlparse(target_url).netloc


def normalize_url(url: str) -> str:
    parsed = urlparse(url)._replace(fragment="")
    result = parsed.geturl()
    return result[:-1] if result.endswith("/") and parsed.path != "/" else result


# =====================================================================
# Tensorlake-native Hacker News fetcher (TOP 3)
# =====================================================================
@function(
    cpu=1,
    memory=1,
    timeout=30,
    retries=Retries(max_retries=2),
)
def fetch_hackernews_top_article() -> list[str]:
    import requests
    from bs4 import BeautifulSoup

    response = requests.get(
        "https://news.ycombinator.com/",
        headers={"User-Agent": "TensorlakeCrawler/1.0"},
        timeout=10,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")

    links = soup.select("span.titleline > a")[:3]

    if len(links) < 3:
        raise RuntimeError("Failed to fetch top 3 Hacker News articles")

    return [link["href"] for link in links]


# =====================================================================
# Podcast Agent (GENERATES 3 PODCASTS)
# =====================================================================
@application()
@function(
    secrets=["GEMINI_API_KEY", "ELEVENLABS_API_KEY"],
    cpu=2,
    memory=4,
    timeout=600,
)
def podcast_agent(_: CrawlInput) -> File:
    import io
    import zipfile

    article_urls = fetch_hackernews_top_article()

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, "w", zipfile.ZIP_DEFLATED) as zipf:
        for idx, article_url in enumerate(article_urls, start=1):
            print(f"\n========== ARTICLE {idx} ==========")
            print(article_url)
            print("==================================\n")

            crawl_input = CrawlInput(
                url=article_url,
                max_depth=1,
                max_links=1,
            )

            crawl_result = crawl(crawl_input)
            clean_text = extract_clean_text(crawl_result)
            summary_text = summarize_with_gemini(clean_text)
            audio_file = generate_audio(summary_text)

            # Write each podcast into the ZIP
            zipf.writestr(
                f"podcast_{idx}.mp3",
                audio_file.content,
            )

    zip_buffer.seek(0)

    return File(
        content=zip_buffer.read(),
        content_type="application/zip",
    )


# ---------------------------------------------------------------------
# Crawler
# ---------------------------------------------------------------------
@function()
def crawl(input: CrawlInput) -> dict:
    visited = set()
    results = {}
    base_url = input.url
    stack = [(normalize_url(base_url), 0)]

    while stack and len(results) < input.max_links:
        current_url, depth = stack.pop()
        if current_url in visited or depth > input.max_depth:
            continue

        visited.add(current_url)
        content_result = fetch_content(current_url)
        results[current_url] = content_result

        if content_result.get("success") and depth < input.max_depth:
            for link in content_result.get("links", []):
                link = normalize_url(link)
                if is_same_domain(base_url, link):
                    stack.append((link, depth + 1))

    return {
        "base_url": base_url,
        "pages": results,
    }


# ---------------------------------------------------------------------
# Content Fetching
# ---------------------------------------------------------------------
@function(image=scraper_image, timeout=60, retries=Retries(max_retries=2), memory=4)
def fetch_content(url: str) -> dict:
    return asyncio.run(_fetch_content_async(url))


async def _fetch_content_async(url: str) -> dict:
    if is_binary_url(url):
        return await _fetch_binary_content(url)
    return await _fetch_page_content(url)


async def _fetch_binary_content(url: str) -> dict:
    import urllib.request

    with urllib.request.urlopen(url, timeout=30) as response:
        content = response.read()
        return {
            "url": url,
            "success": True,
            "content_type": get_file_type(url),
            "is_binary": True,
            "content_base64": base64.b64encode(content).decode(),
            "links": [],
        }


def extract_value(result):
    try:
        return result["result"]["result"]["value"]
    except Exception:
        return result


async def _fetch_page_content(url: str) -> dict:
    from pydoll.browser.chromium import Chrome
    from pydoll.browser.options import ChromiumOptions

    options = ChromiumOptions()
    if platform.system() == "Linux":
        options.binary_location = "/usr/bin/chromium"

    options.add_argument("--headless=new")
    options.add_argument("--no-sandbox")

    async with Chrome(options=options) as browser:
        tab = await browser.start()
        await tab.go_to(url)
        await asyncio.sleep(2)

        text = extract_value(
            await tab.execute_script("return document.body.innerText || ''")
        )

        links_json = extract_value(
            await tab.execute_script(
                """
            return JSON.stringify(
                Array.from(document.querySelectorAll('a[href]'))
                .map(a => a.href)
                .filter(h => h.startsWith('http'))
            )
            """
            )
        )

        return {
            "url": url,
            "success": True,
            "content_type": "text/html",
            "text_content": text,
            "links": list(set(json.loads(links_json or "[]"))),
        }


# ---------------------------------------------------------------------
# Text Extraction
# ---------------------------------------------------------------------
@function()
def extract_clean_text(crawl_result: dict) -> str:
    pages = crawl_result.get("pages", {})
    return "\n\n".join(
        page["text_content"]
        for page in pages.values()
        if isinstance(page.get("text_content"), str)
    )


# ---------------------------------------------------------------------
# Gemini Summarization
# ---------------------------------------------------------------------
@function(secrets=["GEMINI_API_KEY"])
def summarize_with_gemini(clean_text: str) -> str:
    from google import genai
    import os

    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    prompt = f"""
    Summarize the following article in exactly 5 short lines.
    Keep the tone clear, neutral, and suitable for a podcast.

    Article:
    {clean_text[:6000]}
    """

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )

    return response.text


# ---------------------------------------------------------------------
# ElevenLabs Audio
# ---------------------------------------------------------------------
@function(secrets=["ELEVENLABS_API_KEY"])
def generate_audio(script_text: str) -> File:
    import os
    import requests

    VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

    response = requests.post(
        f"https://api.elevenlabs.io/v1/text-to-speech/{VOICE_ID}",
        headers={
            "xi-api-key": os.environ["ELEVENLABS_API_KEY"],
            "Content-Type": "application/json",
            "Accept": "audio/mpeg",
        },
        json={
            "text": script_text,
            "model_id": "eleven_v3",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.5,
            },
        },
    )

    if response.status_code != 200:
        raise RuntimeError(
            f"ElevenLabs TTS failed: {response.status_code} {response.text}"
        )

    return File(
        content=response.content,
        content_type="audio/mpeg",
    )


if __name__ == "__main__":
    from tensorlake.applications import run_local_application

    print("Running Hacker News Podcast Agent (Top 3)...")
    print("-" * 50)

    dummy_input = CrawlInput(
        url="https://news.ycombinator.com/",
        max_depth=1,
        max_links=1,
    )

    request = run_local_application(podcast_agent, dummy_input)
    zip_file = request.output()

    with open("podcasts.zip", "wb") as f:
        f.write(zip_file.content)

    print("Saved podcasts.zip (contains 3 podcast MP3 files)")
