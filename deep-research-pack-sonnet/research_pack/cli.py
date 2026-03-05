"""
Research Pack CLI — research-pack <command> [options]

Commands:
    run     Run a deep research task on a topic
    status  Check status of a previous run
    open    Print paths to report.md and library.json for a run
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

import typer
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)
from rich.table import Table

from research_pack.utils.ids import new_run_id
from research_pack.utils.logging import setup_logging

app = typer.Typer(
    name="research-pack",
    help="Deep Research Pack — generate research reports with citations, powered by Tensorlake.",
    add_completion=False,
    no_args_is_help=True,
)

console = Console()

_DEFAULT_OUT = Path.home() / ".research_pack" / "runs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_run_dir(run_id: str, out_base: Path) -> Optional[Path]:
    candidate = out_base / run_id
    if candidate.exists():
        return candidate
    local = Path.cwd() / run_id
    if local.exists():
        return local
    return None


def _load_run_json(run_dir: Path) -> Optional[dict]:
    p = run_dir / "run.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def _print_summary(result: dict, run_dir: Path) -> None:
    stats = result.get("stats", {})
    sources = result.get("sources", [])

    kept = [s for s in sources if not s.get("duplicate_of") and not s.get("fetch_error")]
    domains: dict[str, int] = {}
    for s in kept:
        try:
            import tldextract
            ext = tldextract.extract(s["url"])
            dom = ext.registered_domain or s["url"].split("/")[2]
        except Exception:
            parts = s["url"].split("/")
            dom = parts[2] if len(parts) > 2 else s["url"]
        domains[dom] = domains.get(dom, 0) + 1
    top_domains = sorted(domains.items(), key=lambda x: -x[1])[:5]

    table = Table(box=box.ROUNDED, show_header=False, title="[bold cyan]Run Summary[/bold cyan]")
    table.add_column("Key", style="bold cyan", width=24)
    table.add_column("Value", style="white")

    table.add_row("Run ID", result.get("run_id", "—"))
    table.add_row("Topic", result.get("topic", "—"))
    table.add_row("Status", result.get("status", "—"))
    table.add_row("Sources fetched", str(stats.get("fetched_count", 0)))
    table.add_row("Sources kept", str(stats.get("kept_count", 0)))
    table.add_row("Duplicates removed", str(stats.get("duplicates_count", 0)))
    table.add_row("Failures", str(stats.get("failures_count", 0)))
    if top_domains:
        table.add_row("Top domains", ", ".join(f"{d} ({c})" for d, c in top_domains))
    if result.get("report_path"):
        table.add_row("Report", result["report_path"])
    if result.get("library_path"):
        table.add_row("Library", result["library_path"])

    console.print()
    console.print(table)


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.command("run")
def cmd_run(
    topic: str = typer.Argument(..., help="Research topic to investigate"),
    out: Optional[Path] = typer.Option(
        None, "--out", "-o", help="Output directory (default: ~/.research_pack/runs)"
    ),
    depth: int = typer.Option(
        1, "--depth", "-d", min=0, max=2,
        help="Crawl depth: 0=seeds only, 1=seeds+1 hop, 2=+2 hops"
    ),
    max_sources: int = typer.Option(
        20, "--max-sources", "-n", help="Max sources to keep after dedup"
    ),
    max_pages_per_domain: int = typer.Option(
        3, "--max-pages-per-domain", help="Max pages to crawl per domain"
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Skip crawling and LLM calls (structure test)"
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose logging"),
) -> None:
    """Run a deep research task on TOPIC and produce a report + source library."""

    setup_logging(logging.DEBUG if verbose else logging.WARNING)

    out_base = out or _DEFAULT_OUT
    out_base.mkdir(parents=True, exist_ok=True)

    if not offline and not os.environ.get("OPENAI_API_KEY"):
        console.print(
            Panel(
                "[bold red]OPENAI_API_KEY is not set.[/bold red]\n\n"
                "Export it before running:\n"
                "  [cyan]export OPENAI_API_KEY=sk-ant-...[/cyan]\n\n"
                "Or use [cyan]--offline[/cyan] to test the output structure.",
                title="Missing API Key",
                border_style="red",
            )
        )
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold white]Topic:[/bold white]       {topic}\n"
            f"[bold white]Depth:[/bold white]       {depth}  "
            f"[bold white]Max sources:[/bold white] {max_sources}  "
            f"[bold white]Offline:[/bold white]     {offline}\n"
            f"[bold white]Output:[/bold white]      {out_base}",
            title="[bold cyan]Research Pack[/bold cyan]",
            border_style="cyan",
        )
    )

    from research_pack.workflow import RunConfig, run_research

    run_id = new_run_id()
    config = RunConfig(
        topic=topic,
        out_dir=out_base,
        depth=depth,
        max_sources=max_sources,
        max_pages_per_domain=max_pages_per_domain,
        offline=offline,
        run_id=run_id,
    )

    stages = [
        "Planning research strategy",
        "Crawling sources",
        "Deduplicating",
        "Enriching sources with AI",
        "Writing report",
        "Done!",
    ]

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=30),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
        transient=False,
    ) as progress:
        ptask = progress.add_task(stages[0], total=5)

        def on_progress(msg: str, current: int, total: int) -> None:
            label = stages[min(current, len(stages) - 1)]
            progress.update(ptask, description=f"[cyan]{label}[/cyan]", completed=current)

        try:
            result = run_research(config, progress=on_progress)
        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted by user. Partial artifacts may exist.[/yellow]")
            raise typer.Exit(1)
        except Exception as exc:
            console.print(f"\n[bold red]Run failed:[/bold red] {exc}")
            if verbose:
                import traceback
                traceback.print_exc()
            raise typer.Exit(1)

        progress.update(ptask, description="[bold green]Done![/bold green]", completed=5)

    _print_summary(result.model_dump(mode="json"), out_base / run_id)

    console.print(
        f"\n[bold green]Report:[/bold green]  {result.report_path}"
        f"\n[bold green]Library:[/bold green] {result.library_path}"
        f"\n[bold green]Run ID:[/bold green]  {result.run_id}"
        "\n"
        "\nTo revisit later:"
        f"\n  [cyan]research-pack status {result.run_id}[/cyan]"
        f"\n  [cyan]research-pack open   {result.run_id}[/cyan]"
    )


@app.command("status")
def cmd_status(
    run_id: str = typer.Argument(..., help="Run ID to check"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output base directory"),
) -> None:
    """Check the status and summary of a previous run."""
    out_base = out or _DEFAULT_OUT
    run_dir = _find_run_dir(run_id, out_base)
    if not run_dir:
        console.print(f"[red]Run not found:[/red] {run_id}")
        console.print(f"Searched in: {out_base}")
        raise typer.Exit(1)

    data = _load_run_json(run_dir)
    if not data:
        console.print(f"[red]run.json not found or unreadable in:[/red] {run_dir}")
        raise typer.Exit(1)

    _print_summary(data, run_dir)


@app.command("open")
def cmd_open(
    run_id: str = typer.Argument(..., help="Run ID to open"),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Output base directory"),
) -> None:
    """Print paths to report.md and library.json for a completed run."""
    out_base = out or _DEFAULT_OUT
    run_dir = _find_run_dir(run_id, out_base)
    if not run_dir:
        console.print(f"[red]Run not found:[/red] {run_id}")
        raise typer.Exit(1)

    data = _load_run_json(run_dir)
    report = data.get("report_path") if data else None
    library = data.get("library_path") if data else None

    report_path = Path(report) if report else run_dir / "report.md"
    library_path = Path(library) if library else run_dir / "library.json"

    if report_path.exists():
        console.print(f"[bold cyan]Report:[/bold cyan]  {report_path}")
    else:
        console.print(f"[yellow]Report not found.[/yellow]  Expected: {report_path}")

    if library_path.exists():
        console.print(f"[bold cyan]Library:[/bold cyan] {library_path}")
    else:
        console.print(f"[yellow]Library not found.[/yellow]  Expected: {library_path}")


if __name__ == "__main__":
    app()
