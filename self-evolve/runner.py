"""Host-side runner — builds the warm snapshot, forks sandboxes per eval task.

Public surface used by orchestrator.py:
    prepare_snapshot()                      -> {"snapshot_id": str, "total_tests": int}
    evaluate_genome(snap_id, genome, tasks) -> list[task_result]

Each task_result:
    {
      "task_id": str,
      "agent": {finished, iterations, tool_calls, edited_files, stop_reason, last_text, error},
      "pytest": {passed, failed, errors, summary},
      "duration_s": float,
    }

Why it looks the way it does:
  - Tensorlake snapshot the warm "deps + repo + pip install" state ONCE,
    then fork a fresh sandbox per task. This is the whole reason we're
    not on Modal/E2B — sub-300ms forks make population-based eval cheap.
  - Tasks within a genome eval run in parallel threads (one sandbox each).
"""
from __future__ import annotations

import io
import json
import os
import re
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tensorlake.sandbox import Sandbox


SELF_EVOLVE = Path(__file__).resolve().parent
BENCH_REPO = SELF_EVOLVE / "bench" / "repo"
HARNESS = SELF_EVOLVE / "agent_harness.py"
SNAPSHOT_CACHE = SELF_EVOLVE / ".snapshot_id"        # legacy: just the id
SNAPSHOT_META = SELF_EVOLVE / ".snapshot_meta.json"  # current: id + total_tests

DEFAULT_CANDIDATE_MODEL = os.environ.get("CANDIDATE_MODEL", "gpt-4.1-mini")


# ---------- helpers -----------------------------------------------------------


def _tar_dir(src: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for p in sorted(src.rglob("*")):
            if any(part in {"__pycache__", ".pytest_cache", ".git"} for part in p.parts):
                continue
            tf.add(p, arcname=str(p.relative_to(src)))
    return buf.getvalue()


def _push_dir(sandbox: Sandbox, local: Path, remote: str) -> None:
    sandbox.run("bash", ["-lc", f"mkdir -p {remote}"], timeout=10)
    sandbox.write_file(f"{remote}/.upload.tar", _tar_dir(local))
    sandbox.run(
        "bash",
        ["-lc", f"cd {remote} && tar xf .upload.tar && rm .upload.tar"],
        timeout=60,
    )


def _parse_pytest_summary(out: str) -> dict:
    last = None
    for line in out.splitlines()[::-1]:
        if " passed" in line or " failed" in line or " error" in line:
            last = line.strip()
            break
    passed = failed = errors = 0
    if last:
        m = re.search(r"(\d+) passed", last)
        if m: passed = int(m.group(1))
        m = re.search(r"(\d+) failed", last)
        if m: failed = int(m.group(1))
        m = re.search(r"(\d+) error", last)
        if m: errors = int(m.group(1))
    return {"passed": passed, "failed": failed, "errors": errors, "summary": last or ""}


# ---------- snapshot ----------------------------------------------------------


def prepare_snapshot(force: bool = False) -> dict:
    """Build (or reuse) the warm sandbox snapshot.

    Returns {"snapshot_id": str, "total_tests": int}. `total_tests` is the
    full-suite test count derived from a pytest sanity run inside the
    snapshot — used by the orchestrator for scoring instead of hardcoding.
    """
    if not force and SNAPSHOT_META.exists():
        try:
            meta = json.loads(SNAPSHOT_META.read_text())
            if meta.get("snapshot_id") and meta.get("total_tests"):
                print(f"[runner] reusing cached snapshot {meta['snapshot_id']} (total_tests={meta['total_tests']})")
                return meta
        except Exception:
            pass

    print("[runner] building base snapshot (one-time, ~2 min)...")
    t0 = time.time()
    sb = Sandbox.create(
        image="tensorlake/ubuntu-minimal",
        cpus=2.0,
        memory_mb=4096,
        timeout_secs=1500,
    )
    try:
        sb.run(
            "bash",
            ["-lc",
             "apt-get update -qq && DEBIAN_FRONTEND=noninteractive "
             "apt-get install -y -qq python3 python3-pip python3-venv git ca-certificates"],
            timeout=600,
        )
        sb.run(
            "bash",
            ["-lc", "pip3 install --break-system-packages -q openai-agents pytest"],
            timeout=300,
        )
        _push_dir(sb, BENCH_REPO, "/workspace/repo")
        sb.run(
            "bash",
            ["-lc", "cd /workspace/repo && pip3 install --break-system-packages -e . -q"],
            timeout=300,
        )
        # Sanity run also gives us the total test count for scoring.
        sanity = sb.run(
            "bash",
            ["-lc", "cd /workspace/repo && python3 -m pytest -q --tb=no 2>&1 | tail -5"],
            timeout=60,
        )
        sanity_summary = _parse_pytest_summary(sanity.stdout)
        total_tests = sanity_summary["passed"] + sanity_summary["failed"] + sanity_summary["errors"]
        if total_tests <= 0:
            raise RuntimeError(
                f"could not determine total_tests from sanity output: {sanity.stdout!r}"
            )
        print(f"[runner] sanity: {sanity_summary['summary']} (total_tests={total_tests})")
        snap = sb.checkpoint(wait=True, timeout=600)
        meta = {"snapshot_id": snap.snapshot_id, "total_tests": total_tests}
        SNAPSHOT_META.write_text(json.dumps(meta, indent=2))
        SNAPSHOT_CACHE.write_text(snap.snapshot_id)  # keep legacy file in sync
        print(f"[runner] snapshot {snap.snapshot_id}  ({time.time()-t0:.1f}s)")
        return meta
    finally:
        sb.terminate()


# ---------- eval --------------------------------------------------------------


def evaluate_one(
    snapshot_id: str,
    genome_dir: Path,
    task: dict,
    max_iters: int = 20,
    model: str = DEFAULT_CANDIDATE_MODEL,
    openai_key: str | None = None,
) -> dict:
    openai_key = openai_key or os.environ["OPENAI_API_KEY"]
    t0 = time.time()
    sb = Sandbox.create(
        snapshot_id=snapshot_id,
        cpus=2.0,
        memory_mb=4096,
        timeout_secs=900,
    )
    try:
        # Reset repo to its snapshotted state so prior agent edits don't leak.
        # (The snapshot fork is already pristine, but be defensive.)
        sb.run("bash", ["-lc", "rm -rf /workspace/genome && mkdir -p /workspace/genome"], timeout=10)
        _push_dir(sb, genome_dir, "/workspace/genome")
        sb.write_file("/workspace/agent_harness.py", HARNESS.read_bytes())

        task_doc = {
            "shown_test": task["shown_test"],
            "repo_dir": "/workspace/repo",
            "genome_dir": "/workspace/genome",
            "max_iters": max_iters,
            "model": model,
        }
        sb.write_file("/workspace/task.json", json.dumps(task_doc).encode())

        run = sb.run(
            "bash",
            ["-lc", "cd /workspace && python3 agent_harness.py task.json"],
            env={"OPENAI_API_KEY": openai_key},
            timeout=720,
        )

        try:
            agent_result = json.loads(sb.read_file("/workspace/result.json").value.decode())
        except Exception as e:
            agent_result = {
                "finished": False,
                "stop_reason": "missing_result",
                "error": f"{type(e).__name__}: {e}",
                "harness_stdout_tail": (run.stdout or "")[-600:],
                "harness_stderr_tail": (run.stderr or "")[-600:],
            }

        pyt = sb.run(
            "bash",
            ["-lc",
             "cd /workspace/repo && python3 -m pytest -q --tb=line 2>&1 | tail -30"],
            timeout=120,
        )
        suite = _parse_pytest_summary(pyt.stdout)
        return {
            "task_id": task["id"],
            "agent": agent_result,
            "pytest": suite,
            "pytest_tail": pyt.stdout[-1500:],
            "duration_s": round(time.time() - t0, 1),
        }
    finally:
        sb.terminate()


def evaluate_genome(
    snapshot_id: str,
    genome_dir: Path,
    tasks: list[dict],
    parallel: int = 3,
    max_iters: int = 20,
    model: str = DEFAULT_CANDIDATE_MODEL,
) -> list[dict]:
    """Evaluate `genome_dir` on each task. Returns list of task_results."""
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=parallel) as ex:
        futs = {
            ex.submit(evaluate_one, snapshot_id, genome_dir, t, max_iters, model): t
            for t in tasks
        }
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                results.append(fut.result())
            except Exception as e:
                results.append({
                    "task_id": t["id"],
                    "agent": {"finished": False, "stop_reason": "runner_exception", "error": f"{type(e).__name__}: {e}"},
                    "pytest": {"passed": 0, "failed": 0, "errors": 0, "summary": ""},
                    "duration_s": 0.0,
                })
    return results


def score(results: list[dict], total_tests: int) -> float:
    """Mean fraction of full-suite tests passing across tasks."""
    if not results:
        return 0.0
    return sum(r["pytest"]["passed"] / total_tests for r in results) / len(results)


# ---------- CLI for quick manual probes --------------------------------------


def _cli() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("snapshot")
    e = sub.add_parser("eval")
    e.add_argument("genome", help="path to genome dir")
    e.add_argument("--split", choices=["train", "eval", "all"], default="train")
    e.add_argument("--parallel", type=int, default=3)
    e.add_argument("--model", default=DEFAULT_CANDIDATE_MODEL)
    args = ap.parse_args()

    if args.cmd == "snapshot":
        prepare_snapshot()
        return

    meta = prepare_snapshot()
    bench = json.loads((SELF_EVOLVE / "bench" / "tasks.json").read_text())
    tasks = bench["train"] if args.split == "train" else bench["eval"] if args.split == "eval" else bench["train"] + bench["eval"]
    out = evaluate_genome(meta["snapshot_id"], Path(args.genome), tasks, parallel=args.parallel, model=args.model)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    _cli()
