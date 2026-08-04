"""Self-evolving agent orchestrator.

Karpathy-style loop, but the artifact being mutated is the candidate
agent's *scaffolding* (system prompt + extra Python tools), not weights.

  ┌──────────────────────────────────────────────────────────────────┐
  │  PER ROUND                                                       │
  │                                                                  │
  │   Planner (ORCHESTRATOR_MODEL) ──[ProposedMutation]──┐           │
  │     reads:                                           ▼           │
  │       - current genome files               apply mutation        │
  │       - recent run logs                     → genomes/gen_N/     │
  │                                                                  │
  │   Runner (Tensorlake)                                            │
  │     forks N sandboxes from warm snapshot                         │
  │     evaluates candidate on TRAIN tasks                           │
  │                                                                  │
  │   Critic (ORCHESTRATOR_MODEL) ──[Verdict {keep, reason}]──> base │
  └──────────────────────────────────────────────────────────────────┘

After all rounds: evaluate the final base on the held-out EVAL tasks for
the headline score, write a markdown summary.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

from agents import Agent, Runner, set_tracing_disabled
from pydantic import BaseModel, Field

set_tracing_disabled(True)  # silence the broken-pipe spam from the tracing exporter

sys.path.insert(0, str(Path(__file__).parent))
from runner import (
    DEFAULT_CANDIDATE_MODEL,
    SELF_EVOLVE,
    evaluate_genome,
    prepare_snapshot,
    score,
)


GENOMES_DIR = SELF_EVOLVE / "genomes"
LOGS_DIR = SELF_EVOLVE / "logs"
LOGS_DIR.mkdir(exist_ok=True)
RUNS_LOG = LOGS_DIR / "runs.jsonl"
SUMMARY_MD = LOGS_DIR / "summary.md"

BENCH = json.loads((SELF_EVOLVE / "bench" / "tasks.json").read_text())


# ---------- structured outputs from Planner / Critic --------------------------


class ProposedMutation(BaseModel):
    """A single mutation to apply to the genome.

    Either edit the system prompt or add/replace code in tools.py — exactly
    one of `new_system_prompt` and `new_tools_py` should be non-empty per
    round. (Both is allowed but discouraged for clean attribution.)
    """
    new_system_prompt: str | None = Field(
        default=None,
        description="If set, replace genome/system_prompt.md with this content.",
    )
    new_tools_py: str | None = Field(
        default=None,
        description="If set, replace genome/tools.py with this content. Must be valid Python; "
                    "tool functions must be named `tool_<name>` and have type-annotated args.",
    )
    rationale: str = Field(description="Why this mutation should improve the score.")


class Verdict(BaseModel):
    keep: bool = Field(description="True if the candidate genome is at least as good as the base.")
    reason: str


# ---------- agents ------------------------------------------------------------

ORCHESTRATOR_MODEL = os.environ.get("ORCHESTRATOR_MODEL", "gpt-4.1")

PLANNER_INSTRUCTIONS = """\
You are the Planner in a self-evolving agent loop.

Your job: propose ONE mutation to a candidate coding agent's scaffolding to
make it better at fixing a Python bug. The candidate agent has these built-in
tools: read_file, write_file, list_dir, run_pytest, run_bash. You can either:

  (a) Rewrite its system prompt (clearer instructions, better workflow), or
  (b) Add a Python helper to its `tools.py` (a function named `tool_<name>`
      with type-annotated args; its docstring is the tool description).

You are shown the current genome and the recent run logs. Look at where the
candidate spends tool calls or fails — that's the lever. Propose a SMALL,
targeted change. Do not rewrite everything.
"""

CRITIC_INSTRUCTIONS = """\
You are the Critic. Decide whether to keep a mutated candidate genome.

You will see:
  - the base genome's score on the train tasks
  - the candidate genome's score on the same train tasks
  - per-task pass/fail and short summaries

Keep the candidate if it is *strictly better* OR *equal but simpler*.
Discard if it regressed on any task without compensating gains. Be conservative;
the loop reverts to the base when you discard.
"""


def make_planner() -> Agent:
    return Agent(
        name="Planner",
        instructions=PLANNER_INSTRUCTIONS,
        model=ORCHESTRATOR_MODEL,
        output_type=ProposedMutation,
    )


def make_critic() -> Agent:
    return Agent(
        name="Critic",
        instructions=CRITIC_INSTRUCTIONS,
        model=ORCHESTRATOR_MODEL,
        output_type=Verdict,
    )


def run_with_retry(agent: Agent, prompt: str, attempts: int = 5, base_delay: float = 8.0):
    """Runner.run_sync with TPM-aware retry/backoff."""
    import openai
    last = None
    for i in range(attempts):
        try:
            return Runner.run_sync(agent, prompt)
        except openai.RateLimitError as e:
            last = e
            wait = base_delay * (2 ** i)
            print(f"[retry] rate-limited; sleeping {wait:.0f}s ({i+1}/{attempts})")
            time.sleep(wait)
    raise last  # type: ignore[misc]


# ---------- genome ops --------------------------------------------------------


def read_genome(path: Path) -> dict:
    return {
        "system_prompt.md": (path / "system_prompt.md").read_text(),
        "tools.py": (path / "tools.py").read_text(),
    }


def fork_genome(base: Path, dest: Path, mutation: ProposedMutation) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(base, dest)
    if mutation.new_system_prompt:
        (dest / "system_prompt.md").write_text(mutation.new_system_prompt)
    if mutation.new_tools_py:
        (dest / "tools.py").write_text(mutation.new_tools_py)


# ---------- logging -----------------------------------------------------------


def log_event(kind: str, payload: dict) -> None:
    rec = {"ts": datetime.utcnow().isoformat(timespec="seconds"), "kind": kind, **payload}
    with RUNS_LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")


def task_summary(results: list[dict], total_tests: int) -> str:
    lines = []
    for r in sorted(results, key=lambda x: x["task_id"]):
        ag = r["agent"]
        pt = r["pytest"]
        lines.append(
            f"  - {r['task_id']:<14} "
            f"agent={ag.get('stop_reason','?'):<18} "
            f"tests={pt['passed']}/{total_tests} "
            f"({pt.get('summary','')})"
        )
    return "\n".join(lines)


# ---------- main loop ---------------------------------------------------------


def run_loop(rounds: int = 6, parallel: int = 1, model: str = DEFAULT_CANDIDATE_MODEL) -> None:
    meta = prepare_snapshot()
    snapshot_id = meta["snapshot_id"]
    total_tests = meta["total_tests"]
    train = BENCH["train"]
    eval_tasks = BENCH["eval"]

    base = GENOMES_DIR / "gen_0"
    print(f"\n[orchestrator] base genome: {base.name}  (total_tests={total_tests})")
    base_results = evaluate_genome(snapshot_id, base, train, parallel=parallel, model=model)
    base_score = score(base_results, total_tests)
    print(f"[orchestrator] base score on train: {base_score:.3f}")
    print(task_summary(base_results, total_tests))
    log_event("base_eval", {"genome": base.name, "score": base_score, "results": base_results})

    planner = make_planner()
    critic = make_critic()

    for r in range(1, rounds + 1):
        print(f"\n{'='*60}\n[round {r}/{rounds}]\n{'='*60}")

        # ---- Planner ----
        genome_files = read_genome(base)
        planner_input = (
            f"## Current genome\n\n"
            f"### system_prompt.md\n```\n{genome_files['system_prompt.md']}\n```\n\n"
            f"### tools.py\n```python\n{genome_files['tools.py']}\n```\n\n"
            f"## Base score on train: {base_score:.3f}\n\n"
            f"## Per-task results (most recent)\n{task_summary(base_results, total_tests)}\n\n"
            f"Propose one targeted mutation."
        )
        plan = run_with_retry(planner, planner_input)
        mutation: ProposedMutation = plan.final_output
        print(f"[planner] {mutation.rationale[:200]}")
        log_event("plan", {"round": r, "mutation": mutation.model_dump()})

        # ---- Apply ----
        cand = GENOMES_DIR / f"gen_{r}"
        fork_genome(base, cand, mutation)

        # ---- Eval candidate ----
        cand_results = evaluate_genome(snapshot_id, cand, train, parallel=parallel, model=model)
        cand_score = score(cand_results, total_tests)
        print(f"[round {r}] candidate score: {cand_score:.3f} (base {base_score:.3f})")
        print(task_summary(cand_results, total_tests))
        log_event("cand_eval", {"round": r, "genome": cand.name, "score": cand_score, "results": cand_results})

        # ---- Critic ----
        critic_input = (
            f"Base genome ({base.name}) score on train: {base_score:.3f}\n"
            f"{task_summary(base_results, total_tests)}\n\n"
            f"Candidate genome ({cand.name}) score on train: {cand_score:.3f}\n"
            f"{task_summary(cand_results, total_tests)}\n\n"
            f"Mutation rationale: {mutation.rationale}"
        )
        # Cooldown so the candidate evals' TPM burst clears before Critic asks.
        time.sleep(15)
        verdict = run_with_retry(critic, critic_input).final_output
        print(f"[critic] keep={verdict.keep}  — {verdict.reason[:200]}")
        log_event("verdict", {"round": r, "keep": verdict.keep, "reason": verdict.reason})

        if verdict.keep:
            base = cand
            base_results = cand_results
            base_score = cand_score

    # ---- Final eval on held-out ----
    print(f"\n{'='*60}\n[final] eval on held-out tasks\n{'='*60}")
    final = evaluate_genome(snapshot_id, base, eval_tasks, parallel=parallel, model=model)
    final_score = score(final, total_tests)
    print(f"[final] {base.name} score on eval: {final_score:.3f}")
    print(task_summary(final, total_tests))
    log_event("final_eval", {"genome": base.name, "score": final_score, "results": final})

    SUMMARY_MD.write_text(
        f"# Self-evolve run summary\n\n"
        f"- Started: {datetime.utcnow().isoformat(timespec='seconds')}Z\n"
        f"- Rounds: {rounds}\n"
        f"- Final base genome: `{base.name}`\n"
        f"- Train score (last): {base_score:.3f}\n"
        f"- **Held-out eval score: {final_score:.3f}**\n\n"
        f"## Final per-task (eval)\n```\n{task_summary(final, total_tests)}\n```\n"
    )
    print(f"\n[orchestrator] summary -> {SUMMARY_MD}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--parallel", type=int, default=1)
    ap.add_argument("--model", default=DEFAULT_CANDIDATE_MODEL,
                    help="Model used by the candidate agent inside the sandbox")
    args = ap.parse_args()
    run_loop(rounds=args.rounds, parallel=args.parallel, model=args.model)
