import os
import streamlit as st
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from tensorlake.applications import run_local_application

from podcast_agent import podcast_agent, CrawlInput

load_dotenv()

# -------------------------
# PAGE CONFIG
# -------------------------
st.set_page_config(
    page_title="Article to Podcast Generator",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------
# CUSTOM CSS
# -------------------------
st.markdown(
    """
<style>
.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}
section[data-testid="stSidebar"] {
    background-color: #161B22;
    border-right: 1px solid #30363D;
}
.feature-card {
    background-color: #1A1C23;
    padding: 24px;
    border-radius: 12px;
    border: 1px solid #333;
    min-height: 250px;
    margin-bottom: 20px;
}
.banner {
    background: linear-gradient(90deg, #1E1E2F 0%, #2D2D44 100%);
    padding: 30px;
    border-radius: 12px;
    border-left: 5px solid #6C5CE7;
    margin-bottom: 30px;
}
</style>
""",
    unsafe_allow_html=True,
)

# -------------------------
# SIDEBAR
# -------------------------
with st.sidebar:
    st.image(
        "https://mintcdn.com/tensorlake-35e9e726/fVE8-oNRlpqs-U2A/logo/TL-Dark.svg?fit=max&auto=format&n=fVE8-oNRlpqs-U2A&q=85&s=33578bd4a1a4952a009f923081d6056e",
        width=250,
    )

    st.subheader("🔑 API Configuration")

    if "gemini_api_key" not in st.session_state:
        st.session_state.gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
    if "elevenlabs_api_key" not in st.session_state:
        st.session_state.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY") or ""

    st.session_state.gemini_api_key = st.text_input(
        "Gemini API Key",
        st.session_state.gemini_api_key,
        type="password",
    )
    st.session_state.elevenlabs_api_key = st.text_input(
        "ElevenLabs API Key",
        st.session_state.elevenlabs_api_key,
        type="password",
    )

    if st.button("💾 Save API Keys"):
        st.success("API Keys Saved!")

    max_depth = st.slider("Crawl Depth", 0, 3, 1)

    st.markdown("---")
    st.subheader("🎯 Key Capabilities")
    st.markdown(
        """
- **Google AI Research**: Gemini 3 Pro
- **Extraction Analysis**: Tensorlake crawler
- **Voice Synthesis**: ElevenLabs
- **Smart Optimization**: AI-powered refinement
"""
    )

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
<div style="display:flex; align-items:center;">
    <h1 style="font-size:2.5rem;">
        🎙️ Article to Podcast with
        <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/gemini-color.png"
             style="height:45px; vertical-align:middle;"/>
        Gemini &
        <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.75.0/files/dark/elevenlabs-text.png"
             style="height:38px; vertical-align:middle;"/>
    </h1>
</div>
""",
    unsafe_allow_html=True,
)

# =====================================================================
# Hacker News – Fetch TOP 3 Articles
# =====================================================================
@st.cache_data(ttl=300)
def fetch_hackernews_top_articles():
    response = requests.get(
        "https://news.ycombinator.com/",
        headers={"User-Agent": "HNFetcher/1.0"},
        timeout=10,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    links = soup.select("span.titleline > a")[:3]

    return [(link.get_text(strip=True), link["href"]) for link in links]


articles = fetch_hackernews_top_articles()

# -------------------------
# DISPLAY ARTICLES
# -------------------------
st.markdown(
    """
<div class="banner">
    <h3>📰 Top Hacker News Articles (Top 3)</h3>
</div>
""",
    unsafe_allow_html=True,
)

for idx, (title, url) in enumerate(articles, start=1):
    st.markdown(f"**{idx}. {title}**  \n{url}")

# -------------------------
# GENERATE BUTTON
# -------------------------
if st.button("🚀 Generate Podcast Audio"):
    if not st.session_state.gemini_api_key or not st.session_state.elevenlabs_api_key:
        st.error("❌ Please configure your API keys.")
        st.stop()

    os.environ["GEMINI_API_KEY"] = st.session_state.gemini_api_key
    os.environ["ELEVENLABS_API_KEY"] = st.session_state.elevenlabs_api_key

    st.subheader("🎧 Generated Podcasts")

    # ✅ SIMPLE LOOP — ONE PODCAST PER ARTICLE
    for idx, (title, url) in enumerate(articles, start=1):
        with st.status(f"🎬 Generating podcast {idx}/3...", expanded=False):
            try:
                input_data = CrawlInput(
                    url=url,
                    max_depth=max_depth,
                    max_links=1,
                )

                request = run_local_application(podcast_agent, input_data)
                audio_file = request.output()
                audio_bytes = bytes(audio_file.content)

                st.markdown(f"### 🎙️ Podcast {idx}: {title}")
                st.audio(audio_bytes, format="audio/mp3")

                st.download_button(
                    label=f"📥 Download Podcast {idx}",
                    data=audio_bytes,
                    file_name=f"podcast_{idx}.mp3",
                    mime="audio/mpeg",
                )

            except Exception as e:
                st.error(f"❌ Failed to generate podcast {idx}")
                st.code(str(e))
