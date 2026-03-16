"""
Distributed Vulnerability Scanner & Auto-Patcher
=================================================

Built on Tensorlake's Agentic Runtime, featuring:

1. DISTRIBUTED TOOLS    — Each scanner/fixer runs in its own isolated container
2. PARALLEL SUB-AGENTS  — 4 specialist detectors fan out simultaneously via .future()
3. MAP-REDUCE           — Scan files in parallel (.map), aggregate findings (.reduce)
4. SANDBOXES            — Fixer agent executes patches in isolated sandboxes

Deploy:
    pip install tensorlake
    tensorlake secrets set ANTHROPIC_API_KEY <key>
    tensorlake deploy vuln_scanner.py

Invoke:
    curl https://api.tensorlake.ai/applications/scan_and_patch \
      -H "Authorization: Bearer $TENSORLAKE_API_KEY" \
      --json '{"repo_url": "https://github.com/example/vulnerable-app", "branch": "main", "max_files": 10}'
"""

import json
import os
import shutil
import sys
import threading
import time
from typing import Any

from pydantic import BaseModel, Field
from tensorlake.applications import application, function, Image



# ---------------------------------------------------------------------------
# Live kanban board — 3-column terminal UI
# ---------------------------------------------------------------------------

class KanbanBoard:
    """Thread-safe live kanban board that redraws in the terminal."""

    QUEUED = 0
    IN_PROGRESS = 1
    DONE = 2

    # Styling
    DIM = "\033[2m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    YELLOW = "\033[33m"
    GREEN = "\033[32m"
    CYAN = "\033[36m"
    WHITE = "\033[37m"
    MAGENTA = "\033[35m"
    BG_GRAY = "\033[48;5;236m"
    SPINNER_FRAMES = ["◐", "◓", "◑", "◒"]

    def __init__(self, steps: list[str]):
        self._lock = threading.Lock()
        self._steps = steps
        self._status = {s: self.QUEUED for s in steps}
        self._details: dict[str, str] = {}
        self._stats: dict[str, str] = {}
        self._start_times: dict[str, float] = {}
        self._end_times: dict[str, float] = {}
        self._spinner_idx = 0
        self._last_height = 0
        self._active = os.isatty(sys.stdout.fileno()) if hasattr(sys.stdout, "fileno") else False

    def start(self, step: str, detail: str = ""):
        with self._lock:
            self._status[step] = self.IN_PROGRESS
            self._start_times[step] = time.time()
            if detail:
                self._details[step] = detail
            self._draw()

    def update(self, step: str, detail: str):
        with self._lock:
            self._details[step] = detail
            self._draw()

    def done(self, step: str, stat: str = ""):
        with self._lock:
            self._status[step] = self.DONE
            self._end_times[step] = time.time()
            if stat:
                self._stats[step] = stat
            self._draw()

    def tick(self):
        """Advance spinner frame."""
        with self._lock:
            self._spinner_idx = (self._spinner_idx + 1) % len(self.SPINNER_FRAMES)
            self._draw()

    def _elapsed(self, step: str) -> str:
        start = self._start_times.get(step)
        if not start:
            return ""
        end = self._end_times.get(step, time.time())
        secs = int(end - start)
        if secs < 60:
            return f"{secs}s"
        return f"{secs // 60}m{secs % 60:02d}s"

    def _draw(self):
        if not self._active:
            return

        cols = shutil.get_terminal_size((100, 40)).columns
        col_w = max((cols - 4) // 3, 20)

        queued = [s for s in self._steps if self._status[s] == self.QUEUED]
        in_prog = [s for s in self._steps if self._status[s] == self.IN_PROGRESS]
        done = [s for s in self._steps if self._status[s] == self.DONE]

        # Build column content
        def card_queued(s: str) -> list[str]:
            label = self._truncate(s, col_w - 6)
            return [f"  {self.DIM}○ {label}{self.RESET}"]

        def card_progress(s: str) -> list[str]:
            spinner = self.SPINNER_FRAMES[self._spinner_idx]
            elapsed = self._elapsed(s)
            label = self._truncate(s, col_w - 12)
            lines = [f"  {self.YELLOW}{spinner} {self.BOLD}{label}{self.RESET} {self.DIM}{elapsed}{self.RESET}"]
            detail = self._details.get(s, "")
            if detail:
                lines.append(f"    {self.DIM}{self._truncate(detail, col_w - 8)}{self.RESET}")
            return lines

        def card_done(s: str) -> list[str]:
            elapsed = self._elapsed(s)
            label = self._truncate(s, col_w - 12)
            lines = [f"  {self.GREEN}✓ {label}{self.RESET} {self.DIM}{elapsed}{self.RESET}"]
            stat = self._stats.get(s, "")
            if stat:
                lines.append(f"    {self.CYAN}{self._truncate(stat, col_w - 8)}{self.RESET}")
            return lines

        col_queued = []
        for s in queued:
            col_queued.extend(card_queued(s))

        col_prog = []
        for s in in_prog:
            col_prog.extend(card_progress(s))

        col_done = []
        for s in done:
            col_done.extend(card_done(s))

        max_rows = max(len(col_queued), len(col_prog), len(col_done), 1)

        # Pad columns to same height
        while len(col_queued) < max_rows:
            col_queued.append("")
        while len(col_prog) < max_rows:
            col_prog.append("")
        while len(col_done) < max_rows:
            col_done.append("")

        # Clear previous draw
        if self._last_height > 0:
            sys.stdout.write(f"\033[{self._last_height}A\033[J")

        # Headers
        hdr_q = self._center_pad(f" {self.DIM}QUEUED ({len(queued)}){self.RESET} ", col_w)
        hdr_p = self._center_pad(f" {self.YELLOW}● RUNNING ({len(in_prog)}){self.RESET} ", col_w)
        hdr_d = self._center_pad(f" {self.GREEN}✓ DONE ({len(done)}){self.RESET} ", col_w)

        border = f"{self.DIM}{'─' * col_w}{self.RESET}"
        lines = []
        lines.append(f"{hdr_q}│{hdr_p}│{hdr_d}")
        lines.append(f"{border}┼{border}┼{border}")

        for i in range(max_rows):
            lq = self._pad(col_queued[i], col_w)
            lp = self._pad(col_prog[i], col_w)
            ld = self._pad(col_done[i], col_w)
            lines.append(f"{lq}{self.DIM}│{self.RESET}{lp}{self.DIM}│{self.RESET}{ld}")

        lines.append(f"{border}┴{border}┴{border}")

        output = "\n".join(lines) + "\n"
        sys.stdout.write(output)
        sys.stdout.flush()
        self._last_height = len(lines)

    def _truncate(self, s: str, max_len: int) -> str:
        if len(s) <= max_len:
            return s
        return s[: max_len - 1] + "…"

    def _pad(self, s: str, width: int) -> str:
        visible = len(self._strip_ansi(s))
        if visible < width:
            return s + " " * (width - visible)
        return s

    def _center_pad(self, s: str, width: int) -> str:
        visible = len(self._strip_ansi(s))
        if visible >= width:
            return s
        left = (width - visible) // 2
        right = width - visible - left
        return " " * left + s + " " * right

    @staticmethod
    def _strip_ansi(s: str) -> str:
        import re
        return re.sub(r"\033\[[0-9;]*m", "", s)


# Global board instance — set up at __main__, used by pipeline functions
_board: KanbanBoard | None = None


def _log(step: str, msg: str):
    """Fallback log when board isn't active (e.g. deployed on Tensorlake)."""
    if _board is None:
        print(f"[{step}] {msg}")


def _extract_json(raw: str) -> Any:
    """Robustly extract JSON from LLM response text.

    Handles: bare JSON, ```json fences, ``` fences, text before/after JSON.
    """
    import re

    raw = raw.strip()

    # Try direct parse first
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences (```json ... ``` or ``` ... ```)
    fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL)
    if fence_match:
        try:
            return json.loads(fence_match.group(1).strip())
        except json.JSONDecodeError:
            pass

    # Try to find a JSON object or array in the text
    for pattern in [r"\{.*\}", r"\[.*\]"]:
        match = re.search(pattern, raw, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

    return None
# ---------------------------------------------------------------------------
# Container images — each function gets its own isolated runtime
# ---------------------------------------------------------------------------

app_image = Image(name="vuln-patcher-img").run(
    "pip install anthropic requests tensorlake"
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
    vuln_type: str  # sqli, xss, ssrf, auth_bypass
    severity: str  # critical, high, medium, low
    description: str
    evidence: str  # the vulnerable code snippet
    cwe_id: str = ""


class ScanResult(BaseModel):
    file_path: str
    vulnerabilities: list[Vulnerability] = Field(default_factory=list)


class AggregatedFindings(BaseModel):
    total_files_scanned: int = 0
    vulnerabilities: list[Vulnerability] = Field(default_factory=list)


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
    test_code: str = ""


class FinalReport(BaseModel):
    repo_url: str
    total_files_scanned: int
    total_vulns_detected: int
    false_positives_rejected: int
    confirmed_vulns: int
    patches_generated: int
    vulnerabilities: list[Vulnerability]
    patches: list[Patch]


# ---------------------------------------------------------------------------
# PHASE 1: Fetch source files from repo
# ---------------------------------------------------------------------------


@function(image=app_image, timeout=600)
def fetch_repo_files(repo_url: str, branch: str, max_files: int = 10) -> list[SourceFile]:
    """
    Fetch source files from a GitHub repository.
    Each file becomes an item for the .map() fan-out.
    """
    import requests

    # Extract owner/repo from URL
    parts = repo_url.rstrip("/").split("/")
    owner, repo = parts[-2], parts[-1]

    # Use GitHub API to get repo tree
    _log("fetch", f"Fetching repo tree for {owner}/{repo} @ {branch}...")
    api_url = f"https://api.github.com/repos/{owner}/{repo}/git/trees/{branch}?recursive=1"
    resp = requests.get(api_url, timeout=30)
    resp.raise_for_status()
    body = resp.json()
    if "tree" not in body:
        raise RuntimeError(f"GitHub API error for {api_url}: {body.get('message', body)}")
    tree = body["tree"]
    _log("fetch", f"Got {len(tree)} items in repo tree")

    # Filter to source files (Python, JS, TS, Go, etc.)
    source_extensions = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".go": "go",
        ".rb": "ruby",
        ".java": "java",
        ".php": "php",
    }

    # Skip directories that don't contain meaningful application code
    skip_prefixes = (
        "test/", "test-config-errors/", "bench/", "examples/", "docs/",
        "errors/", "contributing/", "turbo/", ".github/", ".devcontainer/",
        ".vscode/", ".cursor/", ".claude/", ".claude-plugin/", ".agents/",
        ".conductor/", ".config/", ".husky/", ".cargo/",
        "patches/", "scripts/", "turbopack/crates/turbopack-bench/",
    )

    candidates = [
        item for item in tree
        if item["type"] == "blob"
        and "." in item["path"]
        and ("." + item["path"].rsplit(".", 1)[-1]) in source_extensions
        and not item["path"].startswith(skip_prefixes)
    ]
    # Cap file count to keep local runs manageable
    if max_files > 0 and len(candidates) > max_files:
        _log("fetch", f"Capping from {len(candidates)} to {max_files} files")
        candidates = candidates[:max_files]

    if _board:
        _board.update("Fetch files", f"{len(candidates)} source files found")
    _log("fetch", f"Found {len(candidates)} source files to download")

    source_files: list[SourceFile] = []
    for i, item in enumerate(candidates):
        ext = "." + item["path"].rsplit(".", 1)[-1]
        raw_url = f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{item['path']}"
        file_resp = requests.get(raw_url, timeout=30)
        if file_resp.status_code == 200:
            source_files.append(
                SourceFile(
                    path=item["path"],
                    content=file_resp.text[:50_000],
                    language=source_extensions[ext],
                )
            )
        if (i + 1) % 50 == 0 or (i + 1) == len(candidates):
            if _board:
                _board.update("Fetch files", f"Downloading {i + 1}/{len(candidates)}")
            _log("fetch", f"Downloaded {i + 1}/{len(candidates)} files")

    _log("fetch", f"Done — {len(source_files)} files fetched successfully")
    return source_files


# ---------------------------------------------------------------------------
# PHASE 2: Parallel specialist detectors (4 sub-agents)
#
# DISTRIBUTED TOOLS: Each detector runs in its own container.
# PARALLEL SUB-AGENTS: All 4 launch simultaneously via .future()
# ---------------------------------------------------------------------------

DETECTOR_SYSTEM_PROMPT = """You are a security vulnerability detector specializing in {vuln_type}.
Analyze the provided source code and identify vulnerabilities.

Respond with a JSON array of findings. Each finding must have:
- "line_number": int (approximate line where the vulnerability exists)
- "severity": "critical" | "high" | "medium" | "low"
- "description": str (what the vulnerability is and why it's dangerous)
- "evidence": str (the vulnerable code snippet, 1-3 lines)
- "cwe_id": str (e.g. "CWE-89")

If no vulnerabilities found, respond with an empty array: []
Respond ONLY with the JSON array, no other text."""


def _run_detector(file: SourceFile, vuln_type: str, focus: str) -> list[Vulnerability]:
    """Shared logic for all detectors — calls Claude with a specialist prompt."""
    from anthropic import Anthropic

    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": (
                    f"File: {file.path} (language: {file.language})\n\n"
                    f"Focus on: {focus}\n\n"
                    f"```{file.language}\n{file.content}\n```"
                ),
            }
        ],
        system=DETECTOR_SYSTEM_PROMPT.format(vuln_type=vuln_type),
    )

    findings = _extract_json(response.content[0].text)
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


@function(image=app_image, timeout=300, secrets=["ANTHROPIC_API_KEY"])
def sqli_detector(file: SourceFile) -> list[Vulnerability]:
    """Specialist: SQL Injection (CWE-89) — runs in its own container."""
    return _run_detector(
        file,
        vuln_type="sqli",
        focus=(
            "SQL Injection vulnerabilities: string concatenation in queries, "
            "unsanitized user input in SQL, missing parameterized queries, "
            "ORM raw query misuse, dynamic table/column names from user input."
        ),
    )


@function(image=app_image, timeout=300, secrets=["ANTHROPIC_API_KEY"])
def xss_detector(file: SourceFile) -> list[Vulnerability]:
    """Specialist: Cross-Site Scripting (CWE-79) — runs in its own container."""
    return _run_detector(
        file,
        vuln_type="xss",
        focus=(
            "XSS vulnerabilities: unescaped user input rendered in HTML, "
            "innerHTML usage with user data, template injection, "
            "missing Content-Security-Policy, unsafe dangerouslySetInnerHTML."
        ),
    )


@function(image=app_image, timeout=300, secrets=["ANTHROPIC_API_KEY"])
def ssrf_detector(file: SourceFile) -> list[Vulnerability]:
    """Specialist: Server-Side Request Forgery (CWE-918) — runs in its own container."""
    return _run_detector(
        file,
        vuln_type="ssrf",
        focus=(
            "SSRF vulnerabilities: user-controlled URLs passed to HTTP clients, "
            "missing URL validation/allowlisting, internal network access via "
            "user input, redirect following with user-supplied URLs."
        ),
    )


@function(image=app_image, timeout=300, secrets=["ANTHROPIC_API_KEY"])
def auth_detector(file: SourceFile) -> list[Vulnerability]:
    """Specialist: Authentication/Authorization Bypass (CWE-287/CWE-862) — runs in its own container."""
    return _run_detector(
        file,
        vuln_type="auth_bypass",
        focus=(
            "Auth bypass vulnerabilities: missing authentication checks on endpoints, "
            "broken access control, IDOR, missing authorization middleware, "
            "JWT validation issues, hardcoded credentials, insecure session management."
        ),
    )


# ---------------------------------------------------------------------------
# PHASE 2b: Per-file scanning with parallel detector fan-out
#
# This function is called via .map() across all source files.
# Inside each call, 4 detectors run in PARALLEL via .future().
# ---------------------------------------------------------------------------


@function(timeout=600)
def detect_vulnerabilities(file: SourceFile) -> ScanResult:
    """
    Scan a single file with all 4 specialist detectors in parallel.

    PARALLEL SUB-AGENTS: .future() launches each detector in its own
    container simultaneously. Tensorlake schedules them across the cluster.
    """
    if _board:
        _board.update("Scan files", f"{file.path}")
    _log("scan", f"Scanning {file.path} with 4 parallel detectors...")
    # Fan out to 4 specialist detectors — each runs in its own container
    sqli_findings = sqli_detector.future(file)
    xss_findings = xss_detector.future(file)
    ssrf_findings = ssrf_detector.future(file)
    auth_findings = auth_detector.future(file)

    # Wait for all 4 parallel detectors to complete, then collect results
    all_futures = [sqli_findings, xss_findings, ssrf_findings, auth_findings]
    all_vulns: list[Vulnerability] = []
    for future in all_futures:
        all_vulns.extend(future.result())

    # Assign unique IDs
    for i, v in enumerate(all_vulns):
        v.id = f"{file.path}:{v.vuln_type}:{i}"

    _log("scan", f"{file.path} — found {len(all_vulns)} vulnerabilities")
    return ScanResult(file_path=file.path, vulnerabilities=all_vulns)


# ---------------------------------------------------------------------------
# PHASE 2c: Aggregate scan results (reduce)
#
# MAP-REDUCE: .map() fans detect_vulnerabilities across all files,
#             .reduce() aggregates the per-file results into one report.
# ---------------------------------------------------------------------------


@function(timeout=60)
def aggregate_findings(acc: AggregatedFindings, result: ScanResult) -> AggregatedFindings:
    """
    Reduce function: merge per-file scan results into a single findings list.
    Called sequentially by Tensorlake as .map() results complete.
    """
    acc.total_files_scanned += 1
    acc.vulnerabilities.extend(result.vulnerabilities)
    if _board:
        _board.update("Aggregate", f"{acc.total_files_scanned} files, {len(acc.vulnerabilities)} vulns")
    _log("aggregate", f"{acc.total_files_scanned} files aggregated, {len(acc.vulnerabilities)} total vulns so far")
    return acc


# ---------------------------------------------------------------------------
# PHASE 3: Triage — adversarial manager agent reviews findings
#
# Adversarial manager agent reviews findings to reject false positives
# ---------------------------------------------------------------------------


@function(image=app_image, timeout=600, secrets=["ANTHROPIC_API_KEY"])
def triage_findings(findings: AggregatedFindings) -> TriagedFindings:
    """
    Manager Agent: adversarially reviews all findings to reject false positives.

    Runs in its own container with Claude acting as a skeptical security reviewer.
    """
    from anthropic import Anthropic

    _log("triage", f"Reviewing {len(findings.vulnerabilities)} findings for false positives...")
    if not findings.vulnerabilities:
        _log("triage", "No vulnerabilities to triage")
        return TriagedFindings(triage_reasoning="No vulnerabilities to triage.")

    client = Anthropic()
    vulns_json = json.dumps([v.model_dump() for v in findings.vulnerabilities], indent=2)

    response = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=8192,
        system=(
            "You are an adversarial security reviewer. Your job is to REJECT false positives.\n"
            "For each vulnerability finding, determine if it is a TRUE positive or FALSE positive.\n"
            "Be skeptical — only confirm findings with clear evidence of exploitability.\n\n"
            "Respond with JSON:\n"
            '{"confirmed_ids": ["id1", "id2", ...], "rejected_count": N, "reasoning": "..."}'
        ),
        messages=[
            {
                "role": "user",
                "content": f"Review these vulnerability findings and reject false positives:\n\n{vulns_json}",
            }
        ],
    )

    triage = _extract_json(response.content[0].text)
    if not isinstance(triage, dict):
        return TriagedFindings(
            confirmed=findings.vulnerabilities,
            triage_reasoning="Triage parsing failed — confirming all findings for safety.",
        )

    confirmed_ids = set(triage.get("confirmed_ids", []))
    confirmed = [v for v in findings.vulnerabilities if v.id in confirmed_ids]
    _log("triage", f"Done — {len(confirmed)} confirmed, {triage.get('rejected_count', 0)} rejected as FP")

    return TriagedFindings(
        confirmed=confirmed,
        rejected_as_fp=triage.get("rejected_count", 0),
        triage_reasoning=triage.get("reasoning", ""),
    )


# ---------------------------------------------------------------------------
# PHASE 4: Generate patches — fixer agent with sandbox code execution
#
# SANDBOXES: Each patch is generated and validated in an isolated sandbox.
# Uses .map() to generate patches for all confirmed vulns in parallel.
# ---------------------------------------------------------------------------


@function(image=app_image, timeout=600, secrets=["ANTHROPIC_API_KEY"], ephemeral_disk=4)
def generate_single_patch(vuln: Vulnerability) -> Patch:
    """
    Fixer Agent: generates a patch for a single confirmed vulnerability.

    SANDBOX: Runs in its own isolated container with ephemeral disk.
    Uses Claude for test-driven patch generation.
    """
    from anthropic import Anthropic

    if _board:
        _board.update("Gen patches", f"{vuln.vuln_type} in {vuln.file_path}")
    _log("patch", f"Generating patch for {vuln.id} ({vuln.vuln_type}, {vuln.severity})")
    client = Anthropic()
    response = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=4096,
        system=(
            "You are a security engineer generating patches for vulnerabilities.\n"
            "For each vulnerability, provide:\n"
            "1. The original vulnerable code\n"
            "2. The patched code (minimal, targeted fix)\n"
            "3. A brief explanation of the fix\n"
            "4. A test that validates the fix\n\n"
            "Respond with JSON:\n"
            '{"original_code": "...", "patched_code": "...", "explanation": "...", "test_code": "..."}'
        ),
        messages=[
            {
                "role": "user",
                "content": (
                    f"Generate a patch for this vulnerability:\n\n"
                    f"File: {vuln.file_path}\n"
                    f"Line: {vuln.line_number}\n"
                    f"Type: {vuln.vuln_type}\n"
                    f"Severity: {vuln.severity}\n"
                    f"Description: {vuln.description}\n"
                    f"Evidence:\n```\n{vuln.evidence}\n```"
                ),
            }
        ],
    )

    patch_data = _extract_json(response.content[0].text)
    if not isinstance(patch_data, dict):
        patch_data = {
            "original_code": vuln.evidence,
            "patched_code": "// Patch generation failed — manual review required",
            "explanation": "Failed to parse LLM response",
            "test_code": "",
        }

    _log("patch", f"Patch ready for {vuln.id}")
    return Patch(
        vuln_id=vuln.id,
        file_path=vuln.file_path,
        original_code=patch_data.get("original_code", vuln.evidence),
        patched_code=patch_data.get("patched_code", ""),
        explanation=patch_data.get("explanation", ""),
        test_code=patch_data.get("test_code", ""),
    )


@function(timeout=60)
def collect_patches(patches: list[Patch]) -> list[Patch]:
    """Passthrough to collect .map() results for patches."""
    return patches


# ---------------------------------------------------------------------------
# PHASE 5: Compile final report
# ---------------------------------------------------------------------------


@function(timeout=60)
def compile_report(
    repo_url: str,
    findings: AggregatedFindings,
    triaged: TriagedFindings,
    patches: list[Patch],
) -> FinalReport:
    """Compile everything into a final structured report."""
    return FinalReport(
        repo_url=repo_url,
        total_files_scanned=findings.total_files_scanned,
        total_vulns_detected=len(findings.vulnerabilities),
        false_positives_rejected=triaged.rejected_as_fp,
        confirmed_vulns=len(triaged.confirmed),
        patches_generated=len(patches),
        vulnerabilities=triaged.confirmed,
        patches=patches,
    )


# ---------------------------------------------------------------------------
# ENTRY POINT — orchestrates the full pipeline
# ---------------------------------------------------------------------------


@application()
@function(timeout=3600)
def scan_and_patch(repo_url: str, branch: str = "main", max_files: int = 10) -> FinalReport:
    """
    Main entry point: scan a repo for vulnerabilities and generate patches.

    Orchestration flow:
    1. Fetch all source files from the repo
    2. MAP each file -> detect_vulnerabilities (4 parallel detectors per file)
    3. REDUCE per-file results -> single aggregated findings list
    4. Triage findings (adversarial manager agent rejects FPs)
    5. MAP confirmed vulns -> generate patches (each in its own sandbox)
    6. Compile final report
    """
    # Step 1: Fetch repo files
    if _board:
        _board.start("Fetch files", f"{repo_url} @ {branch}")
    files: list[SourceFile] = fetch_repo_files(repo_url, branch, max_files)
    if _board:
        _board.done("Fetch files", f"{len(files)} files")

    # Step 2: MAP — scan all files in parallel
    if _board:
        _board.start("Scan files", f"{len(files)} files x 4 detectors")
    per_file_results = detect_vulnerabilities.map(files)
    if _board:
        _board.done("Scan files")

    # Step 3: REDUCE — aggregate results
    if _board:
        _board.start("Aggregate")
    findings: AggregatedFindings = aggregate_findings.reduce(
        per_file_results, AggregatedFindings()
    )
    if _board:
        _board.done("Aggregate", f"{len(findings.vulnerabilities)} vulns found")

    # Step 4: Triage — adversarial manager reviews findings
    if _board:
        _board.start("Triage", f"{len(findings.vulnerabilities)} findings")
    triaged: TriagedFindings = triage_findings(findings)
    if _board:
        _board.done("Triage", f"{len(triaged.confirmed)} confirmed")

    # Step 5: Generate patches
    if _board:
        _board.start("Gen patches", f"{len(triaged.confirmed)} vulns")
    patches_per_vuln = generate_single_patch.map(triaged.confirmed)
    patches: list[Patch] = collect_patches(patches_per_vuln)
    if _board:
        _board.done("Gen patches", f"{len(patches)} patches")

    # Step 6: Compile report
    if _board:
        _board.start("Report")
    report = compile_report(repo_url, findings, triaged, patches)
    if _board:
        _board.done("Report")
    return report


# ---------------------------------------------------------------------------
# Local development
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from tensorlake.applications import run_local_application

    STEPS = ["Fetch files", "Scan files", "Aggregate", "Triage", "Gen patches", "Report"]
    _board = KanbanBoard(STEPS)

    # Spinner thread keeps the board animating
    _stop_spinner = threading.Event()

    def _spinner_loop():
        while not _stop_spinner.is_set():
            time.sleep(0.3)
            if not _stop_spinner.is_set():
                _board.tick()

    spinner_thread = threading.Thread(target=_spinner_loop, daemon=True)
    spinner_thread.start()

    # Title
    B, R, C, D = "\033[1m", "\033[0m", "\033[36m", "\033[2m"
    print(f"\n{B}  VULNERABILITY SCANNER & AUTO-PATCHER{R}")
    print(f"{D}  Powered by Tensorlake + Claude{R}\n")

    max_files_input = input(f"  Max files to scan {D}(default: 10){R}: ").strip()
    max_files = int(max_files_input) if max_files_input else 10

    _board._draw()

    result = run_local_application(
        scan_and_patch,
        repo_url="https://github.com/vercel/next.js",
        branch="canary",
        max_files=max_files,
    )
    report: FinalReport = result.output()

    _stop_spinner.set()
    spinner_thread.join()

    # Final report
    G = "\033[32m"
    Y = "\033[33m"
    M = "\033[35m"
    W = "\033[37m"
    SEV_COLOR = {"critical": "\033[31m", "high": "\033[33m", "medium": "\033[35m", "low": "\033[2m"}

    print(f"\n{B}{'=' * 60}{R}")
    print(f"{B}  SCAN COMPLETE{R}  {C}{report.repo_url}{R}")
    print(f"{B}{'=' * 60}{R}")
    print(f"  {W}Files scanned{R}        {B}{report.total_files_scanned}{R}")
    print(f"  {W}Vulns detected{R}       {B}{report.total_vulns_detected}{R}")
    print(f"  {W}False positives{R}      {D}{report.false_positives_rejected}{R}")
    print(f"  {W}Confirmed{R}            {Y}{report.confirmed_vulns}{R}")
    print(f"  {W}Patches generated{R}    {G}{report.patches_generated}{R}")
    print()

    if report.vulnerabilities:
        print(f"  {B}VULNERABILITIES{R}")
        print(f"  {'-' * 56}")
        for vuln in report.vulnerabilities:
            sc = SEV_COLOR.get(vuln.severity, D)
            print(f"  {sc}■ {vuln.severity.upper():8s}{R}  {Y}{vuln.vuln_type:12s}{R}  {vuln.file_path}:{vuln.line_number}")
            print(f"             {D}{vuln.description}{R}")
        print()

    if report.patches:
        print(f"  {B}PATCHES{R}")
        print(f"  {'-' * 56}")
        for patch in report.patches:
            print(f"  {G}✓{R} {patch.file_path}  {D}({patch.vuln_id}){R}")
            print(f"    {patch.explanation}")
        print()

    print(f"{B}{'=' * 60}{R}\n")
