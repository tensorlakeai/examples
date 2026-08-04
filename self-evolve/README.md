# Self-Evolving Bug-Fix Agent

This folder contains a small self-evolution loop for a coding agent. The orchestrator mutates the agent's genome (`system_prompt.md` and optional helper tools), evaluates each candidate in Tensorlake sandboxes, and keeps candidates that improve performance on the benchmark.

## What Is Included

- `orchestrator.py` - runs the planner, evaluator, critic, and final held-out evaluation.
- `runner.py` - creates/reuses Tensorlake snapshots and evaluates genomes in sandbox forks.
- `agent_harness.py` - runs inside each sandbox as the candidate coding agent.
- `bench/` - benchmark repo, tests, and task split.
- `genomes/gen_0/` - base genome required to start a fresh run.

Generated genomes such as `genomes/gen_1/`, `genomes/gen_2/`, etc. are run outputs. Include them only if you want the recipient to inspect your previous evolution history.

## Requirements

- Python 3.10+
- Tensorlake Sandbox access configured for your machine/account
- An OpenAI API key
- Python packages:
  - `tensorlake`
  - `openai-agents`
  - `pytest`
  - `pydantic`

## Setup

From this example directory:

```bash
cd self-evolve
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Set your API key in the shell before running:

```bash
export OPENAI_API_KEY="your_openai_key"
```

If your Tensorlake account requires an environment variable or login command, configure it before running the scripts.

Optional model overrides:

```bash
export ORCHESTRATOR_MODEL="gpt-4.1"
export CANDIDATE_MODEL="gpt-4.1-mini"
```

## Run

Build or reuse the Tensorlake snapshot:

```bash
python runner.py snapshot
```

Evaluate one genome manually:

```bash
python runner.py eval genomes/gen_0 --split train --parallel 1
```

Run the full self-evolution loop:

```bash
python orchestrator.py --rounds 6 --parallel 1
```

The run writes generated genomes and summaries under:

```text
genomes/gen_*/
logs/
.snapshot_id
.snapshot_meta.json
```

## Zip For Sharing

Recommended: ship only source, benchmark files, the base genome, and this README.

Include:

```text
self_evolve/README.md
self_evolve/agent_harness.py
self_evolve/orchestrator.py
self_evolve/runner.py
self_evolve/bench/
self_evolve/genomes/gen_0/
```

Do not include:

```text
self_evolve/.env
self_evolve/.venv/
self_evolve/__pycache__/
self_evolve/.pytest_cache/
self_evolve/logs/
self_evolve/.snapshot_id
self_evolve/.snapshot_meta.json
self_evolve/genomes/gen_1/
self_evolve/genomes/gen_2/
self_evolve/genomes/gen_3/
self_evolve/genomes/gen_4/
self_evolve/genomes/gen_5/
self_evolve/genomes/gen_6/
```

From the repository root, create a clean zip by naming only the paths you want to send:

```bash
zip -r self_evolve_share.zip \
  self_evolve/README.md \
  self_evolve/agent_harness.py \
  self_evolve/orchestrator.py \
  self_evolve/runner.py \
  self_evolve/bench \
  self_evolve/genomes/gen_0
```

If you want to include your evolved genomes for review, add the specific `self_evolve/genomes/gen_N` folders you want to share.
