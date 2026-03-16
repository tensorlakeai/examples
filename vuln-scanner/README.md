# Vulnerability Scanner & Auto-Patcher

Scans a GitHub repo for security vulnerabilities using distributed Claude-powered agents on [Tensorlake](https://docs.tensorlake.ai), then generates patches for confirmed findings.

## Quick Start

### Prerequisites

- Python 3.10+
- An [Anthropic API key](https://console.anthropic.com/)
- A [Tensorlake account](https://cloud.tensorlake.ai)

### 1. Install dependencies

```bash
pip install tensorlake anthropic pydantic requests
```

### 2. Set up credentials

```bash
export ANTHROPIC_API_KEY=<your-anthropic-key>

# Either login interactively or set the API key
tensorlake login
# or
export TENSORLAKE_API_KEY=<your-key>
```

### 3. Run

```bash
python vuln_scanner.py
```

You'll be prompted for the max number of files to scan (default: 10).

### Deploy to Tensorlake

```bash
tensorlake secrets set ANTHROPIC_API_KEY <your-anthropic-key>
tensorlake deploy vuln_scanner.py

curl https://api.tensorlake.ai/applications/scan_and_patch \
  -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
  --json '{"repo_url": "https://github.com/juice-shop/juice-shop", "branch": "main", "max_files": 20}'
```

## Architecture

```
scan_and_patch (entry point)
│
├─► fetch_repo_files                    Fetches source files from GitHub
│
├─► detect_vulnerabilities.map(files)   MAP across all files
│     │
│     └─► Per file, 4 parallel detectors:
│           ├─ sqli_detector            SQL Injection (CWE-89)
│           ├─ xss_detector             Cross-Site Scripting (CWE-79)
│           ├─ ssrf_detector            SSRF (CWE-918)
│           └─ auth_detector            Auth Bypass (CWE-287/862)
│
├─► aggregate_findings.reduce(...)      REDUCE: merge per-file results
│
├─► triage_findings                     Adversarial reviewer rejects false positives
│
├─► generate_single_patch.map(vulns)    MAP: generate patches in isolated sandboxes
│
└─► compile_report                      Final structured output
```

## How It Works

Each `@function()` runs in its own isolated container. For every source file, 4 specialist detectors launch in parallel via `.future()`. `.map()` fans scanning across all files, and `.reduce()` aggregates the results. A triage agent then reviews findings adversarially to reject false positives before patches are generated.

For a repo with 50 source files, this spawns **50 x 4 = 200 parallel detector containers**. Without Tensorlake that's ~100 minutes of sequential LLM calls — with it, ~30 seconds.
