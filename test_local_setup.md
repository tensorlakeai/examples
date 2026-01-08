# Testing Outage Agent Locally

## Quick Setup for Local Testing

### 1. Set Required Environment Variables

For local testing, you need to set these environment variables in your terminal:

**Windows PowerShell:**
```powershell
$env:GROQ_API_KEY = "your_groq_api_key_here"
$env:EXA_API_KEY = "your_exa_api_key_here"
```

**Windows CMD:**
```cmd
set GROQ_API_KEY=your_groq_api_key_here
set EXA_API_KEY=your_exa_api_key_here
```

**Linux/Mac:**
```bash
export GROQ_API_KEY="your_groq_api_key_here"
export EXA_API_KEY="your_exa_api_key_here"
```

### 2. Run the Agent Locally

```bash
python outage_agent.py
```

### 3. Verify Setup

The agent will:
1. Extract alert information
2. Gather internal context
3. Reason with Groq (requires GROQ_API_KEY)
4. Fetch external knowledge with Exa (requires EXA_API_KEY, if needed)
5. Make a decision
6. Store in memory

If you see a `KeyError` for `GROQ_API_KEY` or `EXA_API_KEY`, make sure you've set them in your current terminal session.

### Notes

- **Local testing**: Uses `run_local_application()` - runs on your machine
- **Remote testing**: Uses `run_remote_application()` - runs on Tensorlake Cloud (requires TENSORLAKE_API_KEY)
- Environment variables set in the terminal are only available for that session
- For persistent setup, consider using a `.env` file or system environment variables

