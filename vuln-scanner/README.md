# Vulnerability Scanner

Scans a GitHub repo for SQLi, XSS, SSRF, and auth-bypass vulnerabilities using parallel Claude detectors. Triages false positives, generates patches, and validates each patch against the project's real test suite.

## Setup

```bash
./setup.sh
```

Then put your `ANTHROPIC_API_KEY` in `.env`.

## Run

```bash
python vuln_scanner.py
```

Defaults to OWASP/NodeGoat. Press enter through the prompts to use defaults.

## How it works

The agent runs locally and drives Claude through detect → triage → patch, but every operation that touches untrusted code — cloning the repo, installing `npm`/`pip` dependencies, applying patches, running the project's real test suite — happens inside a [Tensorlake](https://cloud.tensorlake.ai/) sandbox. Tensorlake gives us a disposable VM built once from a declarative `Image` (`ubuntu-systemd` + `git`/`node`/`npm`/`python3`/`pytest`), per-scan isolation with explicit CPU/memory/disk/timeout budgets, and a simple `run` / `read_file` / `write_file` API so the local driver can manipulate the workspace without SSH or Docker plumbing.

1. Clones the target repo into the sandbox.
2. For each source file, runs 4 specialist Claude detectors in parallel.
3. A triage agent classifies findings as `confirmed` / `likely` / `false_positive`.
4. Generates a patch for each surviving finding.
5. Applies each patch in the sandbox, runs `npm test` / `pytest`, reverts.
6. Reports pass/fail per patch.

## Requirements

- Python 3.10+
- [Tensorlake account](https://cloud.tensorlake.ai/)
- [Anthropic API key](https://console.anthropic.com/)

### Dependency installation

The scanner installs Node dependencies with pnpm 11.28.5, a strict 24-hour
release delay, required timestamps and disabled hooks. Existing longer delays
are preserved. npm locks are imported before the frozen guarded install; npm
test commands remain supported. This may reveal undeclared dependencies that
previously depended on npm hoisting. Installation failure stops validation.
Paid scans and Linux image validation are required before merging this change.
