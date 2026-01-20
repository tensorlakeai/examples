import os
import streamlit as st
from dotenv import load_dotenv
from tensorlake.applications import run_local_application
# IMPORTANT: import the Tensorlake application + input model
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
# CUSTOM CSS (Exact from desired UI)
# -------------------------
st.markdown(
    """
<style>
    /* Dark theme background */
    .stApp {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #161B22;
        border-right: 1px solid #30363D;
    }
    
    /* Mode Cards styling */
    .feature-card {
        background-color: #1A1C23;
        padding: 24px;
        border-radius: 12px;
        border: 1px solid #333;
        min-height: 250px;
        margin-bottom: 20px;
    }
    
    /* Banner/Header card */
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
    gemini_key = st.text_input(
        "Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Paste your Google Gemini API key",
    )
    eleven_key = st.text_input(
        "ElevenLabs API Key",
        value=st.session_state.elevenlabs_api_key,
        type="password",
        help="Paste your ElevenLabs API key",
    )
    if st.button("💾 Save API Keys"):
        st.session_state.gemini_api_key = gemini_key
        st.session_state.elevenlabs_api_key = eleven_key
        st.success("API Keys Saved!")
    max_depth = st.slider("Crawl Depth", min_value=0, max_value=3, value=1, step=1)
    st.markdown("---")
    st.subheader("🎯 Key Capabilities")
    st.markdown(
        """
    - **Google AI Research**: Gemini 3 Pro for logic.
    - **Extraction Analysis**: Tensorlake crawler logic.
    - **Voice Synthesis**: ElevenLabs natural voice engine.
    - **Smart Optimization**: AI-powered content refinement.
    """
    )
    st.markdown("---")
    st.markdown("Developed with ❤️ by [Arindam](https://github.com/arindam)")

# -------------------------
# MAIN HEADER
# -------------------------
title_html = """
<div style="display: flex; width: 100%; align-items: center;">
    <h1 style="margin: 0; padding: 0; font-size: 2.5rem; font-weight: bold;">
        <span style="font-size:2.5rem;">🎙️</span> Article to Podcast with
        <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/latest/files/dark/gemini-color.png" style="height: 50px; vertical-align: middle; margin: 0 5px;"/>
        <span style="font-size:3rem">Gemini</span> &
        <img src="https://registry.npmmirror.com/@lobehub/icons-static-png/1.75.0/files/dark/elevenlabs-text.png" style="height: 40px; vertical-align: middle; margin: 0 5px;"/>
    </h1>
</div>
"""
st.markdown(title_html, unsafe_allow_html=True)

st.markdown(
    """
        <div style="background: linear-gradient(135deg, rgba(102, 126, 234, 0.3) 0%, rgba(118, 75, 162, 0.3) 100%); padding: 25px; border-radius: 15px; color: white; margin: 20px 0; border: 1px solid rgba(255, 255, 255, 0.1);">
            <h3 style="color: #ffffff; margin-top: 0;">✨ Transform Your Articles into Podcasts</h3>
            <p style="font-size: 16px; margin-bottom: 0; color: #e0e0e0;"> Generate professional podcast audio from your articles.
                Get comprehensive article summarization, podcast script generation, and natural voice synthesis.
            </p>
        </div>
        """,
    unsafe_allow_html=True,
)

# -------------------------
# MODE SELECTION CARDS
# -------------------------
col_left, col_right = st.columns(2)
with col_left:
    st.markdown(
        """
    <div class="feature-card">
        <h3>🔍 Extraction Mode</h3>
        <p style='color: #8B949E;'>Smart crawling and content cleaning.</p>
        <ul style='color: #C9D1D9;'>
            <li><b>Deep Crawl</b>: Extract text from any website URL.</li>
            <li><b>Tensorlake Logic</b>: Handles JS-heavy content.</li>
            <li><b>Clean Extraction</b>: Removes ads and clutter.</li>
            <li><b>Metadata Parsing</b>: Captures titles and context.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
with col_right:
    st.markdown(
        """
    <div class="feature-card">
        <h3>🎙️ Generation Mode</h3>
        <p style='color: #8B949E;'>AI Scripting and Voice Synthesis.</p>
        <ul style='color: #C9D1D9;'>
            <li><b>Gemini Summarization</b>: Professional podcast scripts.</li>
            <li><b>Natural Voices</b>: ElevenLabs high-quality synthesis.</li>
            <li><b>Stability Control</b>: AI voice parameter tuning.</li>
            <li><b>Instant Audio</b>: Direct MP3 download ready.</li>
        </ul>
    </div>
    """,
        unsafe_allow_html=True,
    )
st.markdown("<br>", unsafe_allow_html=True)

# -------------------------
# INPUT SECTION
# -------------------------
article_url = st.text_input(
    "Enter the article URL:", placeholder="https://www.tensorlake.ai/blog/toon-vs-json"
)

# -------------------------
# GENERATE BUTTON
# -------------------------
if st.button("🚀 Generate Podcast Audio"):
    if not article_url:
        st.error("❌ Please provide an article URL.")
    elif not st.session_state.gemini_api_key or not st.session_state.elevenlabs_api_key:
        st.error("❌ Please configure your API keys in the sidebar.")
    else:
        with st.status("🎬 Running Tensorlake pipeline...", expanded=True) as status:
            try:
                input_data = CrawlInput(
                    url=article_url,
                    max_depth=max_depth,
                    max_links=1,
                )
                request = run_local_application(podcast_agent, input_data)
                audio_file = request.output()
                # 🔧 FIX: convert bytearray → bytes for Streamlit
                audio_bytes = bytes(audio_file.content)
                st.audio(audio_bytes, format="audio/mp3")

                status.update(
                    label="✅ Podcast generated successfully!",
                    state="complete",
                )

                st.markdown("---")
                st.success("🎉 Your podcast is ready!")

                res_col1, res_col2 = st.columns([2, 1])
                with res_col1:
                    st.subheader("🎧 Podcast Audio")
                    st.audio(audio_bytes, format="audio/mp3")
                with res_col2:
                    st.subheader("📥 Actions")
                    st.download_button(
                        label="Download MP3",
                        data=audio_bytes,
                        file_name="podcast.mp3",
                        mime="audio/mpeg",
                        type="primary",
                    )


            except Exception as e:
                status.update(label="❌ Pipeline failed", state="error")
                st.error(f"Error: {str(e)}")
