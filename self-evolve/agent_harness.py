"""Candidate-agent harness — runs INSIDE the Tensorlake sandbox.

Built on the OpenAI Agents SDK (`agents` package). The Runner handles the
tool-call loop, tracing, and turn limits; we just declare tools and the
initial input.

Invocation:
    python /workspace/agent_harness.py /workspace/task.json

task.json schema:
    {
      "shown_test":   "tests/test_normalize.py::test_v4_dotted",
      "repo_dir":     "/workspace/repo",
      "genome_dir":   "/workspace/genome",
      "max_iters":    20,
      "model":        "gpt-4.1-mini"
    }

Outputs (written to /workspace/result.json):
    {
      "finished":        bool,
      "iterations":      int,
      "tool_calls":      int,
      "last_text":       str,
      "edited_files":    [str, ...],
      "stop_reason":     "done" | "end_turn_no_done" | "budget" | "error",
      "error":           str | null
    }
"""
from __future__ import annotations

import importlib.util
import inspect
import json
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Callable

from agents import Agent, Runner, function_tool
from agents.exceptions import MaxTurnsExceeded
from agents.items import ToolCallItem, MessageOutputItem


# Module-level state — set in main() before tools are invoked. Tools close
# over these so the @function_tool decorator can introspect plain signatures.
REPO_DIR: Path = Path("/workspace/repo")
EDITED: set[str] = set()


def _truncate(s: str, n: int = 4096) -> str:
    return s if len(s) <= n else s[:n] + f"\n[...truncated {len(s) - n} chars]"


# ---------- built-in tools ---------------------------------------------------


@function_tool
def read_file(path: str) -> str:
    """Read a UTF-8 text file from the repo. Path is relative to repo root."""
    p = (REPO_DIR / path).resolve()
    if not str(p).startswith(str(REPO_DIR)):
        return "ERROR: path escapes repo"
    try:
        return _truncate(p.read_text())
    except Exception as e:
        return f"ERROR: {e}"


@function_tool
def write_file(path: str, content: str) -> str:
    """Overwrite (or create) a UTF-8 text file in the repo."""
    p = (REPO_DIR / path).resolve()
    if not str(p).startswith(str(REPO_DIR)):
        return "ERROR: path escapes repo"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    rel = str(p.relative_to(REPO_DIR))
    EDITED.add(rel)
    return f"wrote {len(content)} bytes to {rel}"


@function_tool
def list_dir(path: str = ".") -> str:
    """List entries in a directory under the repo root."""
    p = (REPO_DIR / path).resolve()
    if not str(p).startswith(str(REPO_DIR)):
        return "ERROR: path escapes repo"
    try:
        return "\n".join(sorted(e.name + ("/" if e.is_dir() else "") for e in p.iterdir()))
    except Exception as e:
        return f"ERROR: {e}"


@function_tool
def run_pytest(node_id: str = "") -> str:
    """Run pytest in the repo. Pass an optional node id (e.g. tests/test_x.py::test_y) to scope."""
    cmd = ["python3", "-m", "pytest", "-x", "--tb=short", "-q"]
    if node_id:
        cmd.append(node_id)
    proc = subprocess.run(cmd, cwd=REPO_DIR, capture_output=True, text=True, timeout=120)
    return f"exit={proc.returncode}\n{_truncate((proc.stdout or '') + (proc.stderr or ''))}"


@function_tool
def run_bash(cmd: str) -> str:
    """Run a bash command in the repo root. Use sparingly; prefer dedicated tools."""
    proc = subprocess.run(["bash", "-lc", cmd], cwd=REPO_DIR, capture_output=True, text=True, timeout=120)
    return f"exit={proc.returncode}\n{_truncate((proc.stdout or '') + (proc.stderr or ''))}"


BUILTIN_TOOLS = [read_file, write_file, list_dir, run_pytest, run_bash]


# ---------- genome-defined tools --------------------------------------------


def _load_genome_tools(genome_dir: Path) -> list[Callable]:
    """Load any function in genome/tools.py whose name starts with `tool_`,
    strip the prefix, and wrap it with @function_tool."""
    tools_py = genome_dir / "tools.py"
    if not tools_py.exists():
        return []
    spec = importlib.util.spec_from_file_location("genome_tools", tools_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    wrapped = []
    for name, fn in inspect.getmembers(mod, inspect.isfunction):
        if not name.startswith("tool_"):
            continue
        # Rename so the model sees the natural tool name.
        fn.__name__ = name[len("tool_"):]
        wrapped.append(function_tool(fn))
    return wrapped


# ---------- main ------------------------------------------------------------


RESULT_PATH = Path("/workspace/result.json")


def _write_result(stop: str, *, iterations: int = 0, tool_calls: int = 0,
                  last_text: str = "", err: str | None = None) -> None:
    RESULT_PATH.write_text(json.dumps({
        "finished": stop == "done",
        "iterations": iterations,
        "tool_calls": tool_calls,
        "last_text": last_text,
        "edited_files": sorted(EDITED),
        "stop_reason": stop,
        "error": err,
    }, indent=2))


def _count_items(items) -> tuple[int, int]:
    tool_calls = sum(1 for it in items if isinstance(it, ToolCallItem))
    iterations = sum(1 for it in items if isinstance(it, MessageOutputItem))
    return iterations, tool_calls


def main() -> None:
    global REPO_DIR

    # Write a skeleton result first so the runner never sees `missing_result`
    # if we crash before the SDK call returns.
    _write_result("error", err="harness_started_but_did_not_finish")

    task = json.loads(Path(sys.argv[1]).read_text())
    REPO_DIR = Path(task["repo_dir"])
    genome_dir = Path(task["genome_dir"])
    max_iters = int(task.get("max_iters", 20))
    model = task.get("model", "gpt-4.1-mini")

    system_prompt = (genome_dir / "system_prompt.md").read_text()
    tools = BUILTIN_TOOLS + _load_genome_tools(genome_dir)

    user_prompt = (
        f"The failing test is: `{task['shown_test']}`.\n\n"
        f"Repo root: `{REPO_DIR}`.\n\n"
        "Investigate, fix the underlying bug, then run the FULL test suite "
        "(`run_pytest` with no node_id). Reply `DONE` once everything is green."
    )

    agent = Agent(
        name="BugFixer",
        instructions=system_prompt,
        tools=tools,
        model=model,
    )

    # Retry the whole run on transient API errors. Each retry restarts the
    # agent loop from scratch — we keep this simple because EDITED persists
    # across attempts within the same harness process.
    import openai
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            result = Runner.run_sync(agent, user_prompt, max_turns=max_iters)
            iterations, tool_calls = _count_items(result.new_items)
            final = result.final_output
            last_text = (final if isinstance(final, str) else str(final))[:1000]
            stop = "done" if "DONE" in last_text.upper() else "end_turn_no_done"
            _write_result(stop, iterations=iterations, tool_calls=tool_calls, last_text=last_text)
            return
        except MaxTurnsExceeded as e:
            # Hit the turn budget — not an error. Pull whatever progress we can.
            iterations = tool_calls = 0
            items = getattr(e, "run_data", None) or getattr(e, "items", None) or []
            try:
                iterations, tool_calls = _count_items(items)
            except Exception:
                pass
            _write_result("budget", iterations=iterations, tool_calls=tool_calls,
                          err=f"max_turns={max_iters}")
            return
        except (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError) as e:
            last_exc = e
            wait = 8.0 * (2 ** attempt)
            print(f"[harness] transient API error ({type(e).__name__}); retry in {wait:.0f}s",
                  file=sys.stderr, flush=True)
            time.sleep(wait)
        except Exception as e:
            traceback.print_exc(file=sys.stderr)
            _write_result("error", err=f"{type(e).__name__}: {e}")
            return

    # Exhausted retries on transient errors.
    _write_result("error", err=f"retries_exhausted: {type(last_exc).__name__}: {last_exc}")


if __name__ == "__main__":
    main()
