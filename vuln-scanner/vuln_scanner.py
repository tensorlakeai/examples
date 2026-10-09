"""
Vulnerability Scanner & Auto-Patcher (sandbox-only edition)
============================================================

A plain Python agent — no Tensorlake Applications/orchestration framework.
The agent runs locally; it uses ONE Tensorlake sandbox as its workspace:

    Local process (this script)                Tensorlake sandbox
    ─────────────────────────                   ──────────────────
    Drives the pipeline                         /workspace/repo (cloned once)
    Calls Claude (anthropic SDK)                Holds source files
    Threads detectors in parallel    ◄──read───  ...
    Triage / patch gen (Claude)
    Apply + test patches             ───write──► patched files
                                                 npm test / pytest runs here

Run:
    export TENSORLAKE_API_KEY=...
    export ANTHROPIC_API_KEY=...
    python vuln_scanner.py
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

from pydantic import BaseModel, Field
from anthropic import Anthropic
from tensorlake.sandbox import Sandbox
from tensorlake import Image

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


def _bootstrap_tensorlake_env() -> None:
    """The Tensorlake image builder reads org/project/auth from env vars only —
    it ignores ./.tensorlake/config.toml and ~/.config/tensorlake/credentials.toml
    that `tl login` and `tl init` write. This bridges the gap so the user
    doesn't have to manually `export` after running the CLI."""
    from pathlib import Path

    cfg_path = Path(".tensorlake/config.toml")
    if cfg_path.exists():
        text = cfg_path.read_text()
        m = re.search(r'organization\s*=\s*"([^"]+)"', text)
        if m and not os.environ.get("TENSORLAKE_ORGANIZATION_ID"):
            os.environ["TENSORLAKE_ORGANIZATION_ID"] = m.group(1)
        m = re.search(r'project\s*=\s*"([^"]+)"', text)
        if m and not os.environ.get("TENSORLAKE_PROJECT_ID"):
            os.environ["TENSORLAKE_PROJECT_ID"] = m.group(1)

    cred_path = Path.home() / ".config" / "tensorlake" / "credentials.toml"
    if cred_path.exists() and not os.environ.get("TENSORLAKE_PAT") and not os.environ.get("TENSORLAKE_API_KEY"):
        text = cred_path.read_text()
        m = re.search(r'token\s*=\s*"([^"]+)"', text)
        if m:
            os.environ["TENSORLAKE_PAT"] = m.group(1)


_bootstrap_tensorlake_env()

# ---------------------------------------------------------------------------
# Sandbox image — built once, reused across runs
# ---------------------------------------------------------------------------

AGENT_IMAGE_NAME = "vuln-agent"

agent_image = (
    Image(name=AGENT_IMAGE_NAME, base_image="ubuntu-systemd")
    .run(
        "apt-get update -qq && "
        "apt-get install -y -qq git curl ca-certificates python3 python3-pip"
    )
    .run("curl -fsSL https://deb.nodesource.com/setup_24.x | bash - && apt-get install -y nodejs")
    .run("npm install --global --ignore-scripts pnpm@11.28.5")
    .run("python3 -m pip install --break-system-packages -q pytest PyYAML")
)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class SourceFile(BaseModel):
    path: str
    content: str
    language: str = "python"


class Vulnerability(BaseModel):
    id: str = ""
    file_path: str
    line_number: int
    vuln_type: str
    severity: str
    description: str
    evidence: str
    cwe_id: str = ""


class TriagedFindings(BaseModel):
    confirmed: list[Vulnerability] = Field(default_factory=list)
    rejected_as_fp: int = 0
    triage_reasoning: str = ""


class Patch(BaseModel):
    vuln_id: str
    file_path: str
    original_code: str
    patched_code: str
    explanation: str
    validation_passed: bool = False
    validation_output: str = ""


class FinalReport(BaseModel):
    repo_url: str
    total_files_scanned: int
    total_vulns_detected: int
    false_positives_rejected: int
    confirmed_vulns: int
    patches_generated: int
    vulnerabilities: list[Vulnerability]
    patches: list[Patch]
    triage_reasoning: str = ""


# ---------------------------------------------------------------------------
# JSON extraction (LLMs sometimes wrap JSON in fences or chatter)
# ---------------------------------------------------------------------------


def _first_text_block(response) -> str:
    """Anthropic responses can technically contain non-text blocks (tool_use,
    refusals, etc). Pull the first text block, or empty string if none."""
    for block in response.content:
        if getattr(block, "type", None) == "text" and getattr(block, "text", None):
            return block.text
    return ""


def _extract_json(raw: str) -> Any:
    raw = raw.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return None


# ---------------------------------------------------------------------------
# Sandbox helpers — clone, walk, read, patch+test
# ---------------------------------------------------------------------------

REPO_DIR = "/workspace/repo"

SOURCE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".go": "go", ".rb": "ruby", ".java": "java", ".php": "php",
}

SKIP_DIRS = {
    "test", "tests", "node_modules", ".git", "vendor", "dist", "build",
    "__pycache__", ".github", ".vscode", "docs", "examples", "bench",
    "scripts", "patches",
}


def _log(step: str, msg: str) -> None:
    print(f"  [{step}] {msg}", flush=True)


def clone_repo(sandbox: Sandbox, repo_url: str, branch: str) -> None:
    _log("clone", f"{repo_url} @ {branch}")
    sandbox.run("bash", ["-lc", f"rm -rf {REPO_DIR}"], timeout=30)
    result = sandbox.run(
        "git", ["clone", "--depth=1", "--branch", branch, repo_url, REPO_DIR],
        timeout=300,
    )
    if result.exit_code != 0:
        raise RuntimeError(f"git clone failed: {result.stderr}")
    _log("clone", "complete")


def collect_source_files(sandbox: Sandbox, max_files: int) -> list[SourceFile]:
    """Use `find` inside the sandbox to enumerate source files, then pull them."""
    prune = " -o ".join(f"-name {d}" for d in SKIP_DIRS)
    exts = " -o ".join(f"-name '*{ext}'" for ext in SOURCE_EXTENSIONS)
    find_cmd = (
        f"cd {REPO_DIR} && find . -type d \\( {prune} \\) -prune -o "
        f"-type f \\( {exts} \\) -print"
    )
    result = sandbox.run("bash", ["-lc", find_cmd], timeout=60)
    rels = [
        (line[2:] if line.startswith("./") else line)
        for line in (result.stdout or "").splitlines()
        if line.strip()
    ]
    if max_files > 0:
        rels = rels[:max_files]
    _log("scan", f"{len(rels)} source files to pull")

    files: list[SourceFile] = []
    for rel in rels:
        ext = os.path.splitext(rel)[1]
        try:
            data = sandbox.read_file(f"{REPO_DIR}/{rel}").value
        except Exception as e:
            _log("scan", f"could not read {rel}: {e}")
            continue
        files.append(SourceFile(
            path=rel,
            content=data.decode("utf-8", errors="replace")[:50_000],
            language=SOURCE_EXTENSIONS.get(ext, "text"),
        ))
    _log("scan", f"pulled {len(files)} files into local memory")
    return files


# ---------------------------------------------------------------------------
# Specialist detectors — plain functions, parallelized via ThreadPoolExecutor
# ---------------------------------------------------------------------------

DETECTOR_SYSTEM_PROMPT = """You are a security vulnerability detector specializing in {vuln_type}.
Analyze the provided source code and identify vulnerabilities.

Respond with a JSON array of findings. Each finding must have:
- "line_number": int
- "severity": "critical" | "high" | "medium" | "low"
- "description": str
- "evidence": str (1-3 lines of vulnerable code)
- "cwe_id": str

If none found, respond with an empty array: []
Respond ONLY with the JSON array."""

DETECTORS = [
    ("sqli", "SQL Injection: string concatenation in queries, unsanitized user input "
             "in SQL, missing parameterized queries, ORM raw query misuse."),
    ("xss", "XSS: unescaped user input in HTML, innerHTML with user data, "
            "template injection, unsafe dangerouslySetInnerHTML."),
    ("ssrf", "SSRF: user-controlled URLs in HTTP clients, missing URL allowlisting, "
             "internal network access via user input."),
    ("auth_bypass", "Auth bypass: missing authentication checks, broken access control, "
                    "IDOR, missing authorization middleware, JWT validation issues, "
                    "hardcoded credentials."),
]


def _run_detector(client: Anthropic, file: SourceFile, vuln_type: str, focus: str) -> list[Vulnerability]:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=DETECTOR_SYSTEM_PROMPT.format(vuln_type=vuln_type),
        messages=[{
            "role": "user",
            "content": (
                f"File: {file.path} (language: {file.language})\n\n"
                f"Focus on: {focus}\n\n"
                f"```{file.language}\n{file.content}\n```"
            ),
        }],
    )
    findings = _extract_json(_first_text_block(response))
    if not isinstance(findings, list):
        return []
    return [
        Vulnerability(
            file_path=file.path,
            line_number=f.get("line_number", 0),
            vuln_type=vuln_type,
            severity=f.get("severity", "medium"),
            description=f.get("description", ""),
            evidence=f.get("evidence", ""),
            cwe_id=f.get("cwe_id", ""),
        )
        for f in findings
    ]


def scan_files(client: Anthropic, files: list[SourceFile], max_workers: int = 16) -> list[Vulnerability]:
    """Fan out (file × detector) pairs to a thread pool. Each task is an LLM call."""
    if not files:
        _log("scan", "WARNING: no source files to scan — check collect_source_files")
        return []

    tasks = [(f, vt, focus) for f in files for (vt, focus) in DETECTORS]
    _log("scan", f"{len(files)} files × {len(DETECTORS)} detectors = {len(tasks)} LLM calls")

    # Collect (key, vulns) pairs from each future. ID assignment happens
    # single-threaded after the pool drains so IDs are deterministic.
    results: list[tuple[str, str, list[Vulnerability]]] = []
    done_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(_run_detector, client, f, vt, focus): (f.path, vt)
                   for (f, vt, focus) in tasks}
        for fut in as_completed(futures):
            path, vt = futures[fut]
            try:
                vulns = fut.result()
            except Exception as e:
                _log("scan", f"detector {vt} on {path} failed: {e}")
                vulns = []
            results.append((path, vt, vulns))
            done_count += 1
            if done_count % 10 == 0 or done_count == len(tasks):
                total_so_far = sum(len(v) for _, _, v in results)
                _log("scan", f"{done_count}/{len(tasks)} detector calls done, {total_so_far} vulns so far")

    # Stable ID assignment: sort by (path, vuln_type) so re-runs produce same IDs.
    results.sort(key=lambda r: (r[0], r[1]))
    all_vulns: list[Vulnerability] = []
    per_key_counter: dict[tuple[str, str], int] = {}
    for path, vt, vulns in results:
        for v in vulns:
            key = (v.file_path, v.vuln_type)
            idx = per_key_counter.get(key, 0)
            v.id = f"{v.file_path}:{v.vuln_type}:{idx}"
            per_key_counter[key] = idx + 1
            all_vulns.append(v)
    return all_vulns


# ---------------------------------------------------------------------------
# Triage — manager agent rebalances detector findings
# ---------------------------------------------------------------------------


def triage(client: Anthropic, vulns: list[Vulnerability]) -> TriagedFindings:
    if not vulns:
        return TriagedFindings(triage_reasoning="No vulnerabilities to triage.")

    _log("triage", f"reviewing {len(vulns)} findings")
    vulns_json = json.dumps([v.model_dump() for v in vulns], indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=8192,
        system=(
            "You are a security reviewer triaging detector findings.\n"
            "Classify each finding as one of:\n"
            '  - "confirmed": clear evidence of exploitability in the snippet\n'
            '  - "likely": pattern looks vulnerable but exploitability is not provable from the snippet alone\n'
            '  - "false_positive": safe pattern, framework already escapes/parameterizes, or test/example code\n'
            "Default to \"likely\" when uncertain — a human will review the survivors.\n"
            "Only mark \"false_positive\" when you can cite the specific reason it is safe.\n\n"
            "Respond with JSON:\n"
            '{"confirmed_ids": ["..."], "likely_ids": ["..."], "false_positive_ids": ["..."], "reasoning": "per-finding notes"}'
        ),
        messages=[{"role": "user", "content": f"Triage these findings:\n\n{vulns_json}"}],
    )

    parsed = _extract_json(_first_text_block(response))
    if not isinstance(parsed, dict):
        _log("triage", "parse failed — keeping all findings")
        return TriagedFindings(confirmed=vulns, triage_reasoning="Triage parse failed; kept all.")

    keep_ids = set(parsed.get("confirmed_ids", [])) | set(parsed.get("likely_ids", []))
    fp_ids = set(parsed.get("false_positive_ids", []))
    confirmed = [v for v in vulns if v.id in keep_ids]
    _log("triage", f"{len(confirmed)} kept (confirmed+likely), {len(fp_ids)} rejected as FP")

    # Claude sometimes returns reasoning as a dict (per-finding notes) instead
    # of a string. Coerce to a printable form either way.
    reasoning_raw = parsed.get("reasoning", "")
    if isinstance(reasoning_raw, dict):
        reasoning_str = "\n".join(f"  {k}: {v}" for k, v in reasoning_raw.items())
    elif isinstance(reasoning_raw, list):
        reasoning_str = "\n".join(str(x) for x in reasoning_raw)
    else:
        reasoning_str = str(reasoning_raw)

    return TriagedFindings(
        confirmed=confirmed,
        rejected_as_fp=len(fp_ids),
        triage_reasoning=reasoning_str,
    )


# ---------------------------------------------------------------------------
# Patch generation — one LLM call per vuln, threaded
# ---------------------------------------------------------------------------


def _generate_patch(client: Anthropic, vuln: Vulnerability) -> Patch:
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=(
            "You are a security engineer generating patches.\n"
            "Provide:\n"
            "1. The exact original vulnerable code (drop-in replaceable)\n"
            "2. The patched code (minimal, targeted fix)\n"
            "3. A brief explanation\n\n"
            "Respond with JSON:\n"
            '{"original_code": "...", "patched_code": "...", "explanation": "..."}'
        ),
        messages=[{
            "role": "user",
            "content": (
                f"File: {vuln.file_path}\nLine: {vuln.line_number}\n"
                f"Type: {vuln.vuln_type}\nSeverity: {vuln.severity}\n"
                f"Description: {vuln.description}\nEvidence:\n```\n{vuln.evidence}\n```"
            ),
        }],
    )
    data = _extract_json(_first_text_block(response))
    if not isinstance(data, dict):
        data = {"original_code": vuln.evidence, "patched_code": "", "explanation": "parse failed"}
    return Patch(
        vuln_id=vuln.id,
        file_path=vuln.file_path,
        original_code=data.get("original_code", vuln.evidence),
        patched_code=data.get("patched_code", ""),
        explanation=data.get("explanation", ""),
    )


def generate_patches(client: Anthropic, vulns: list[Vulnerability], max_workers: int = 8) -> list[Patch]:
    if not vulns:
        return []
    _log("patch", f"generating {len(vulns)} patches in parallel")
    patches: list[Patch] = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_generate_patch, client, v) for v in vulns]
        for fut in as_completed(futures):
            try:
                patches.append(fut.result())
            except Exception as e:
                _log("patch", f"patch generation failed: {e}")
    return patches


# ---------------------------------------------------------------------------
# Patch validation — apply each patch in the sandbox, run tests, revert
# ---------------------------------------------------------------------------


def _detect_project(sandbox: Sandbox) -> tuple[str, str]:
    listing = sandbox.run("bash", ["-lc", f"ls {REPO_DIR}"], timeout=10)
    files = set((listing.stdout or "").split())
    if "package.json" in files and "pnpm-lock.yaml" in files:
        return "node", f"cd {REPO_DIR} && pnpm test"
    if "package.json" in files:
        return "node", f"cd {REPO_DIR} && npm test --silent"
    if "requirements.txt" in files or "pyproject.toml" in files:
        return "python", f"cd {REPO_DIR} && pytest -q"
    return "unknown", ""


def _install_deps(sandbox: Sandbox, project_type: str) -> bool:
    """Returns True on successful install. Logs a warning and returns False on failure
    so the caller can decide whether to continue or short-circuit validation."""
    _log("validate", f"installing {project_type} deps...")
    if project_type == "node":
        helper = Path(__file__).with_name("guarded_node_install.py").read_bytes()
        sandbox.write_file("/tmp/guarded_node_install.py", helper)
        result = sandbox.run(
            "python3", ["-I", "/tmp/guarded_node_install.py", REPO_DIR],
            timeout=900,
        )
    elif project_type == "python":
        result = sandbox.run(
            "bash", ["-lc",
                     f"cd {REPO_DIR} && (pip install --break-system-packages -q -r requirements.txt 2>/dev/null || "
                     f"pip install --break-system-packages -q .)"],
            timeout=900,
        )
    else:
        return False

    if result.exit_code != 0:
        tail = (result.stderr or result.stdout or "")[-500:]
        _log("validate", f"WARNING: dep install exited {result.exit_code}. tail:\n{tail}")
        return False
    return True


def validate_patches(sandbox: Sandbox, patches: list[Patch]) -> list[Patch]:
    if not patches:
        return []

    project_type, test_cmd = _detect_project(sandbox)
    _log("validate", f"project_type={project_type}")
    if project_type == "unknown":
        for p in patches:
            p.validation_output = "no test runner detected"
        return patches

    deps_ok = _install_deps(sandbox, project_type)
    if not deps_ok:
        for p in patches:
            p.validation_output = "dependency install failed — see [validate] log above"
        return patches

    for p in patches:
        target = f"{REPO_DIR}/{p.file_path}"
        try:
            original_bytes = sandbox.read_file(target).value
        except Exception as e:
            p.validation_output = f"could not read {p.file_path}: {e}"
            continue

        patched = original_bytes.decode("utf-8", errors="replace")
        if p.original_code and p.original_code in patched:
            patched = patched.replace(p.original_code, p.patched_code, 1)
        else:
            patched = p.patched_code

        try:
            sandbox.write_file(target, patched.encode("utf-8"))
            _log("validate", f"running tests for {p.vuln_id}...")
            result = sandbox.run("bash", ["-lc", test_cmd], timeout=600)
            p.validation_passed = result.exit_code == 0
            output = (result.stdout or "")
            if result.stderr:
                output += f"\n[stderr]\n{result.stderr}"
            p.validation_output = (output or f"exit_code={result.exit_code}")[-4000:]
        finally:
            sandbox.write_file(target, original_bytes)

    return patches


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def scan_and_patch(repo_url: str, branch: str, max_files: int) -> FinalReport:
    if not os.getenv("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY not set")
    if not (os.getenv("TENSORLAKE_API_KEY") or os.getenv("TENSORLAKE_PAT")):
        raise RuntimeError("Neither TENSORLAKE_API_KEY nor TENSORLAKE_PAT is set")

    client = Anthropic()

    _log("sandbox", f"creating sandbox from image '{AGENT_IMAGE_NAME}'...")
    sandbox = Sandbox.create(
        name="vuln-scan",
        image=AGENT_IMAGE_NAME,
        cpus=2.0,
        memory_mb=4096,
        disk_mb=12_288,
        timeout_secs=3600,
    )
    _log("sandbox", f"created: {sandbox.sandbox_id}")

    try:
        clone_repo(sandbox, repo_url, branch)
        files = collect_source_files(sandbox, max_files)
        all_vulns = scan_files(client, files)
        triaged = triage(client, all_vulns)
        patches = generate_patches(client, triaged.confirmed)
        patches = validate_patches(sandbox, patches)

        return FinalReport(
            repo_url=repo_url,
            total_files_scanned=len(files),
            total_vulns_detected=len(all_vulns),
            false_positives_rejected=triaged.rejected_as_fp,
            confirmed_vulns=len(triaged.confirmed),
            patches_generated=len(patches),
            vulnerabilities=triaged.confirmed,
            patches=patches,
            triage_reasoning=triaged.triage_reasoning,
        )
    finally:
        _log("sandbox", "terminating...")
        try:
            sandbox.terminate()
        except Exception as e:
            _log("sandbox", f"WARNING: terminate failed (sandbox may leak until timeout): {e}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    B, R, C, D = "\033[1m", "\033[0m", "\033[36m", "\033[2m"
    G, Y = "\033[32m", "\033[33m"
    SEV_COLOR = {"critical": "\033[31m", "high": "\033[33m", "medium": "\033[35m", "low": "\033[2m"}

    print(f"\n{B}  VULNERABILITY SCANNER & AUTO-PATCHER{R}")
    print(f"{D}  Sandbox-only edition · powered by Tensorlake + Claude{R}\n")

    # Build the image if it isn't already registered. Idempotent + fast on rebuilds.
    if "--build-image" in sys.argv or os.getenv("BUILD_IMAGE") == "1":
        print(f"  {D}Building sandbox image '{AGENT_IMAGE_NAME}'...{R}")
        agent_image.build(registered_name=AGENT_IMAGE_NAME)
        print(f"  {G}Image built.{R}\n")
        return

    repo_url = input(f"  Repo URL {D}(default: OWASP/NodeGoat){R}: ").strip() or "https://github.com/OWASP/NodeGoat"
    branch = input(f"  Branch {D}(default: master){R}: ").strip() or "master"
    max_files_in = input(f"  Max files {D}(default: 10){R}: ").strip()
    max_files = int(max_files_in) if max_files_in else 10

    t0 = time.time()
    report = scan_and_patch(repo_url, branch, max_files)
    dt = time.time() - t0

    print(f"\n{B}{'=' * 60}{R}")
    print(f"{B}  SCAN COMPLETE{R}  {C}{report.repo_url}{R}  {D}({dt:.1f}s){R}")
    print(f"{B}{'=' * 60}{R}")
    print(f"  Files scanned        {B}{report.total_files_scanned}{R}")
    print(f"  Vulns detected       {B}{report.total_vulns_detected}{R}")
    print(f"  False positives      {D}{report.false_positives_rejected}{R}")
    print(f"  Confirmed/likely     {Y}{report.confirmed_vulns}{R}")
    print(f"  Patches generated    {G}{report.patches_generated}{R}")
    print()

    if report.triage_reasoning:
        print(f"  {B}TRIAGE NOTES{R}")
        print(f"  {'-' * 56}")
        for line in report.triage_reasoning.splitlines():
            print(f"  {D}{line}{R}")
        print()

    if report.vulnerabilities:
        print(f"  {B}VULNERABILITIES{R}")
        print(f"  {'-' * 56}")
        for v in report.vulnerabilities:
            sc = SEV_COLOR.get(v.severity, D)
            print(f"  {sc}■ {v.severity.upper():8s}{R}  {Y}{v.vuln_type:12s}{R}  {v.file_path}:{v.line_number}")
            print(f"             {D}{v.description}{R}")
        print()

    if report.patches:
        print(f"  {B}PATCHES{R}")
        print(f"  {'-' * 56}")
        for p in report.patches:
            mark = f"{G}✓{R}" if p.validation_passed else f"{Y}!{R}"
            status = f"{G}tests passed{R}" if p.validation_passed else f"{Y}tests failed{R}"
            print(f"  {mark} {p.file_path}  {D}({p.vuln_id}){R}  {status}")
            print(f"    {p.explanation}")
        print()

    print(f"{B}{'=' * 60}{R}\n")


if __name__ == "__main__":
    main()
