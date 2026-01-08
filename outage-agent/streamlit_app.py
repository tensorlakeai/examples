"""
Streamlit UI for Outage Agent - Final Clean Version
Bullet lists now use plain <li> without * or •
"""

import streamlit as st
import os
from datetime import datetime
import re
import json
from tensorlake.applications import run_remote_application, Request

# ========================= PAGE CONFIG =========================
st.set_page_config(
    page_title="Outage Agent",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========================= CUSTOM CSS =========================
st.markdown("""
<style>
    .stApp {
        background: #0f0f1a;
        color: #e0e0e0;
    }
    h1, h2, h3, h4 {
        color: #ffffff !important;
        font-weight: 600;
    }
    section[data-testid="stSidebar"] {
        background-color: #0f0f1a;
        border-right: 1px solid #1e1e2e;
    }
    .hero-banner {
        background: linear-gradient(135deg, #667eea, #764ba2);
        padding: 2.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 2rem 0;
        color: white;
        box-shadow: 0 8px 20px rgba(0,0,0,0.3);
    }
    .mode-card {
        background: #1a1a2e;
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #2a2a3a;
        margin: 1rem 0;
        box-shadow: 0 6px 16px rgba(0,0,0,0.4);
        height: 100%;
    }
    .mode-card h3 {
        color: #ffffff;
        margin-bottom: 1rem;
    }
    .mode-card ul {
        opacity: 0.9;
        font-size: 0.95rem;
        line-height: 1.7;
        padding-left: 1.2rem;
    }
    .feature-card {
        background: #1e1e2e;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        height: 3.2em;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #5a67d8, #6b46c1);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
    }
    .stTextInput > div > div > input,
    .stTextArea > div > div > textarea {
        background-color: #1a1a2e;
        color: #e0e0e0;
        border: 1px solid #3a3a5a;
        border-radius: 12px;
    }
    .stSuccess { background-color: #166534 !important; border-radius: 12px; padding: 1rem; }
    .stError { background-color: #7f1d1d !important; border-radius: 12px; padding: 1rem; }
    .stInfo { background-color: #1e3a8a; border-left: 4px solid #3b82f6; border-radius: 10px; }
    .stWarning { background-color: #713f12; border-left: 4px solid #f59e0b; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# ========================= SESSION STATE =========================
if "history" not in st.session_state:
    st.session_state.history = []
if "alert_text" not in st.session_state:
    st.session_state.alert_text = ""

# ========================= SIDEBAR =========================
with st.sidebar:
    try:
        st.image("Assets/Logo.png", use_container_width=True)
    except:
        st.markdown("<h2 style='color:#667eea; text-align:center;'>🚨 Tensorlake</h2>", unsafe_allow_html=True)
    
    st.markdown("---")
    
    st.markdown("### 🔑 API Configuration")
    tensorlake_key = st.text_input(
        "Tensorlake API Key",
        type="password",
        placeholder="Enter your key",
        help="Required to access your deployed Outage Agent",
        label_visibility="collapsed"
    )
    
    if tensorlake_key:
        os.environ["TENSORLAKE_API_KEY"] = tensorlake_key.strip()
        st.success("Key entered")
    else:
        pass  # No message when empty

    st.markdown("---")
    
    st.markdown("### 📜 Recent Analyses")
    if st.session_state.history:
        for i, entry in enumerate(reversed(st.session_state.history[-8:])):
            escalated = "🚨" if entry.get('escalated', False) else "✓"
            time_str = entry['timestamp'].split(" ")[1][:5]
            preview = entry['alert'][:40] + "..." if len(entry['alert']) > 40 else entry['alert']
            with st.expander(f"{escalated} {time_str} • {preview}"):
                if st.button("↻ Reuse Alert", key=f"reuse_{i}", width="stretch"):
                    st.session_state.alert_text = entry['alert']
                    st.rerun()
    else:
        st.info("No analyses yet")

    if st.session_state.history:
        if st.button("🗑️ Clear History", width="stretch"):
            st.session_state.history = []
            st.rerun()

    st.markdown("---")

    st.markdown("### 🔑 Key Capabilities")
    st.markdown("""
    <div class="feature-card">
    <ul>
    <li><strong>Groq Intelligence</strong>: Lightning-fast reasoning with Llama 3.3 70B</li>
    <li><strong>Exa Integration</strong>: Real-time external incident correlation</li>
    <li><strong>Tensorlake Orchestration</strong>: Durable, observable workflows</li>
    <li><strong>Smart Escalation</strong>: Only wakes humans when truly needed</li>
    <li><strong>Structured Decisions</strong>: JSON output for automation</li>
    <li><strong>Future Memory</strong>: Learns from past incidents over time</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Developed with ❤️ by Arindam")

# ========================= MAIN CONTENT =========================
st.markdown("<h1 style='text-align:center; color:white; margin-bottom:0;'>🚨 Outage Agent with Tensorlake</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:#aaa; font-size:1.1rem;'>Your AI-Powered On-Call Engineer</p>", unsafe_allow_html=True)

st.markdown("""
<div class="hero-banner">
    <h2>✨ Intelligent Incident Response</h2>
    <p style="font-size:1.1rem; opacity:0.9; max-width:800px; margin:0 auto;">
    Submit any alert. The agent analyzes severity, infers root cause using Groq, searches external knowledge via Exa when needed, 
    and intelligently decides whether to escalate — all orchestrated reliably on Tensorlake Cloud.
    </p>
</div>
""", unsafe_allow_html=True)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="mode-card">
        <h3>✓ Handled Automatically</h3>
        <p><strong>No human escalation needed</strong></p>
        <ul>
        <li>Known incident pattern with high confidence</li>
        <li>Clear root cause identified</li>
        <li>Safe, standard mitigation steps recommended</li>
        <li>Verification metric provided for monitoring</li>
        <li>You can stay asleep — just follow the guidance</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="mode-card">
        <h3>🚨 Escalation Required</h3>
        <p><strong>Human review needed now</strong></p>
        <ul>
        <li>Low confidence in analysis</li>
        <li>Unfamiliar or novel error pattern</li>
        <li>Critical impact (e.g., data loss, full outage)</li>
        <li>Contradictory signals or ambiguity</li>
        <li>On-call engineer should investigate immediately</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

st.markdown("### Enter Alert")
alert_text = st.text_area(
    label="Alert Description",
    value=st.session_state.alert_text,
    placeholder="e.g., Unusual spike in authentication failures. Failed logins increased 800% in 10 minutes. Auth service affected.",
    height=140,
    label_visibility="collapsed"
)

col1, col2 = st.columns([1, 4])
with col1:
    run_btn = st.button("🚀 Analyze Alert", type="primary", width="stretch")
with col2:
    clear_btn = st.button("Clear", width="stretch")

if clear_btn:
    st.session_state.alert_text = ""
    st.rerun()

# ========================= EXECUTION =========================
if run_btn:
    if not alert_text.strip():
        st.warning("Please enter an alert description.")
        st.stop()

    current_key = os.getenv("TENSORLAKE_API_KEY", "")
    if not current_key:
        st.error("Tensorlake API key required.")
        st.stop()

    with st.spinner("Agent analyzing on Tensorlake Cloud..."):
        try:
            request = run_remote_application("outage_agent", alert_text.strip())
            output = request.output()
            output_str = str(output)

            json_match = re.search(r'\{.*"incident_id".*\}', output_str, re.DOTALL)
            json_data = {}
            escalated = False
            if json_match:
                try:
                    clean_json = re.sub(r'```json\s*|```', '', json_match.group(0))
                    json_data = json.loads(clean_json)
                    escalated = json_data.get("should_escalate", False)
                except:
                    escalated = "escalation required" in output_str.lower()

            st.session_state.history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "alert": alert_text.strip(),
                "response": output_str,
                "escalated": escalated
            })

            st.markdown("---")
            if escalated:
                st.error("🚨 **ESCALATION REQUIRED**")
                st.markdown("""
                <div style="background:#7f1d1d; padding:1.2rem; border-radius:12px; border-left:5px solid #dc2626; color:white; margin:1rem 0;">
                <strong>Human review required immediately.</strong><br>
                The agent has low confidence or detected a critical/unknown issue that needs on-call attention.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.success("✓ **Handled Automatically**")
                st.markdown("""
                <div style="background:#166534; padding:1.2rem; border-radius:12px; border-left:5px solid #10b981; color:white; margin:1rem 0;">
                <strong>No escalation needed.</strong><br>
                Known pattern with high confidence. Follow the recommended actions and monitor as advised.
                </div>
                """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### Summary")
            summary = output_str.split('\n\n')[0].strip()
            st.info(summary)

            if json_data:
                cols = st.columns(4)
                with cols[0]: st.metric("Service", json_data.get("service", "N/A"))
                with cols[1]: st.metric("Severity", json_data.get("severity", "unknown").capitalize())
                with cols[2]: st.metric("Status", json_data.get("status", "unknown").capitalize())
                with cols[3]: st.metric("Escalation", "Yes" if escalated else "No")

                st.markdown("**Root Cause**")
                st.write(json_data.get("root_cause", "Not determined"))

                if json_data.get("actions_taken"):
                    st.markdown("**Recommended Actions**")
                    for a in json_data["actions_taken"]:
                        st.write(f"{a}")

                if json_data.get("next_recommendation"):
                    st.markdown("**Next Steps**")
                    st.warning(json_data["next_recommendation"])

                with st.expander("📋 Raw Structured Output"):
                    st.json(json_data)
            else:
                st.code(output_str)

        except Exception as e:
            error_str = str(e).lower()
            if "authorization" in error_str or "unauthorized" in error_str or "invalid api key" in error_str:
                st.error("Invalid API key")
            else:
                st.error("Analysis failed")
                with st.expander("Error Details"):
                    st.exception(e)

st.markdown("---")
st.caption("Running on Tensorlake Cloud • Production-Ready • Built for On-Call")