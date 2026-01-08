# 🚨 Outage Agent by Tensorlake

A production-ready serverless AI agent deployed on Tensorlake Cloud that simulates an always-on on-call engineer. The agent automatically analyzes alerts, investigates incidents using semantic search and web browsing, and provides intelligent escalation decisions with structured output.

## 🌟 Features

- **🤖 Intelligent Analysis**: Uses Groq LLM (llama-3.3-70b-versatile) for fast inference and decision-making
- **🔍 Evidence Gathering**: 
  - Semantic search via Exa for similar outages and root causes
  - HTTP GET tool for checking status pages and documentation
- **⚡ Smart Escalation**: Only escalates when truly necessary (high impact, ongoing issues, or unclear causes)
- **📊 Structured Output**: Provides clear summaries with JSON format for systems integration
- **💻 Modern Web UI**: Beautiful Streamlit interface with:
  - Real-time alert analysis
  - Analysis history with one-click reuse
  - Color-coded results with detailed breakdowns
  - Compact, expandable history items

## 🏗️ Architecture

- **Platform**: Tensorlake Applications (deployed to Tensorlake Cloud)
- **LLM**: Groq (llama-3.3-70b-versatile)
- **Tools**: Exa search, HTTP GET (all implemented as `@function` decorators)
- **Orchestration**: LangChain for prompt templates and agent loops
- **UI**: Streamlit for interactive web interface

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt --upgrade
```

### 2. Set Up Secrets

```bash
# Set Tensorlake API Key
export TENSORLAKE_API_KEY="your_tensorlake_api_key"

# Set service API keys
tensorlake secrets set GROQ_API_KEY "your_groq_api_key"
tensorlake secrets set EXA_API_KEY "your_exa_api_key"
```

### 3. Test Locally

**Option A: Run the agent locallyt **
```bash
python outage_agent.py
```

**Option B: Use the Streamlit web interface (Recommended)**
```bash
streamlit run streamlit_app.py
```
Then open your browser to `http://localhost:8501`

### 4. Deploy to Tensorlake Cloud

```bash
tensorlake deploy outage_agent.py
```

### 5. Run the agent deployed to Tensorlake Cloud

```bash
python test_remote_outage_agent.py
```

## Usage

### Running locally

```python
from tensorlake.applications import run_local_application, Request

alert = "Alert: High error rate detected in API service"
request: Request = run_local_application(outage_agent, alert)
output = request.output()
print(output)
```

### Running on Tensorlake Cloud

```python
from tensorlake.applications import run_remote_application

alert = "Alert: Service degradation detected"
request = run_remote_application("outage_agent", alert)
output = request.output()
print(output)
```

### Web Interface (Streamlit)

The easiest way to interact with the agent is through the Streamlit web interface:

```bash
streamlit run streamlit_app.py
```

**Features:**
- Enter alerts in a clean text input
- Real-time analysis with progress indicators
- Color-coded results with gradient cards
- Analysis history with one-click reuse
- Sidebar configuration for API keys

### Command Line

**Using the test script:**
```bash
# Test with default alert
python test_remote_outage_agent.py

# Test with custom alert
python test_remote_outage_agent.py "Alert: Database connection pool exhausted"
```


## Agent Behavior

The agent follows intelligent decision-making rules:

1. **Analyze**: Quickly analyzes the alert description
2. **Gather Evidence**: Uses tools to find similar incidents and check relevant URLs
3. **Smart Decision Making**: 
   - **Handles** expected, scheduled, known, or already resolved issues
   - **Handles** transient issues that have recovered
   - **Handles** routine, low-impact problems with analysis and recommendations
   - **Escalates** only when:
     - High impact on users (data loss, complete outage, critical service)
     - Issue is ongoing/not resolved
     - No clear explanation from alert description
     - Tools reveal conflicting or worrying information
4. **Tool Failure Handling**: When tools fail, relies on alert description rather than automatically escalating
5. **Structured Summary**: Provides clear output with:
   - What happened
   - What was done (or why nothing was done)
   - What should be monitored next
   - Escalation reason (if escalated)

## Project Structure

```
.
├── outage_agent.py              # Main application with @application decorator
├── streamlit_app.py             # Streamlit web interface (recommended)
├── test_remote_outage_agent.py  # Remote testing script
├── outage_agent_cli.py          # Optional CLI wrapper (requires click)
├── requirements.txt             # Python dependencies
├── DEPLOYMENT.md               # Detailed deployment instructions
├── QUICKSTART.md               # Quick start guide
├── Assets/
│   └── Logo.png                # Tensorlake logo for UI
└── README.md                   # This file
```

## Configuration

### LLM Model

The agent uses Groq's `llama-3.3-70b-versatile` model by default. To change the model, edit the `outage_agent()` function in `outage_agent.py`:

```python
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # Change to "mixtral-8x7b-instant" or other available models
    groq_api_key=os.environ["GROQ_API_KEY"],
    temperature=0.1,
)
```

### Tool Settings

- **Exa Search**: Default `num_results=5`
- **HTTP GET**: Default `timeout=10` seconds

## API Keys

The following API keys are required:

- **TENSORLAKE_API_KEY**: Your Tensorlake Cloud API key
- **GROQ_API_KEY**: Groq API key for LLM inference
- **EXA_API_KEY**: Exa API key for semantic search

See `DEPLOYMENT.md` for detailed setup instructions.

## Troubleshooting

See `DEPLOYMENT.md` for common issues and troubleshooting steps.

## 📝 Output Format

The agent provides structured output in three formats:

1. **One-Paragraph Summary**: Human-readable summary for quick understanding
2. **Structured JSON**: Machine-readable format with:
   - Incident ID, service, severity, status
   - Root cause analysis with confidence score
   - Actions taken and verification metrics
   - Escalation decision and recommendations
3. **Slack-Friendly Format**: Condensed version for notifications

## 🙏 Acknowledgments

- Built with [Tensorlake](https://tensorlake.ai) for serverless AI deployment
- Powered by [Groq](https://groq.com) for fast LLM inference
- Uses [Exa](https://exa.ai) for semantic search
- UI built with [Streamlit](https://streamlit.io)

## 📧 Support

For issues and questions:
- Check the [Tensorlake documentation](https://docs.tensorlake.ai)

