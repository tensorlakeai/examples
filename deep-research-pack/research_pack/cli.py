"""CLI for Research Pack — built with Typer + Rich."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn

from research_pack.utils.text import extract_domain

app = typer.Typer(
    name="deeprp",
    help="Deep Research Pack — AI-powered research synthesis.",
    add_completion=False,
)
console = Console()

# Default output root
_DEFAULT_OUT = os.environ.get("RESEARCH_PACK_OUT", "./output")


# ---------------------------------------------------------------------------
# research-pack run
# ---------------------------------------------------------------------------


@app.command()
def run(
    topic: str = typer.Argument(..., help="Research topic or question"),
    out: str = typer.Option(_DEFAULT_OUT, "--out", "-o", help="Output directory root"),
    depth: int = typer.Option(1, "--depth", "-d", help="Crawl depth (1 or 2)"),
    max_sources: int = typer.Option(
        20, "--max-sources", "-n", help="Maximum seed URLs to crawl"
    ),
    max_pages_per_domain: int = typer.Option(
        5, "--max-pages-per-domain", help="Maximum pages per domain"
    ),
    offline: bool = typer.Option(
        False, "--offline", help="Skip LLM calls (test pipeline mechanics only)"
    ),
) -> None:
    """Run a deep research pipeline on TOPIC."""

    # Quick pre-flight check
    if not offline and not os.environ.get("OPENAI_API_KEY"):
        console.print(
            "[bold red]Error:[/bold red] OPENAI_API_KEY is not set. "
            "Export it or pass --offline to skip LLM calls."
        )
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[bold blue]Research Pack[/bold blue]\n\n"
            f"[dim]Topic:[/dim]  {topic}\n"
            f"[dim]Depth:[/dim]  {depth}   "
            f"[dim]Max sources:[/dim] {max_sources}   "
            f"[dim]Offline:[/dim] {offline}",
            border_style="blue",
        )
    )

    from research_pack.workflow import run_pipeline

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=False,
    ) as progress:
        task = progress.add_task("[bold]Starting\u2026[/bold]", total=None)

        def _on_progress(stage: str, detail: str) -> None:
            label = {
                "planning": "[cyan]Plan[/cyan]",
                "crawling": "[yellow]Crawl[/yellow]",
                "deduplicating": "[magenta]Dedupe[/magenta]",
                "enriching": "[green]Enrich[/green]",
                "writing": "[blue]Write[/blue]",
                "done": "[bold green]Done[/bold green]",
            }.get(stage, stage)
            progress.update(task, description=f"{label} {detail}")

        result = run_pipeline(
            topic=topic,
            out_dir=out,
            depth=depth,
            max_sources=max_sources,
            max_pages_per_domain=max_pages_per_domain,
            offline=offline,
            progress_callback=_on_progress,
        )

    # -- Summary table -----------------------------------------------------
    console.print()
    table = Table(title="Run Summary", border_style="dim")
    table.add_column("Metric", style="cyan", no_wrap=True)
    table.add_column("Value", style="green")

    table.add_row("Run ID", result.run_id)
    table.add_row("Sources fetched", str(result.stats.fetched_count))
    table.add_row("Sources kept", str(result.stats.kept_count))
    table.add_row("Duplicates removed", str(result.stats.duplicates_count))
    table.add_row("Failures", str(result.stats.failures_count))

    # top domains
    domains: dict[str, int] = {}
    for s in result.sources:
        if not s.duplicate_of:
            d = extract_domain(s.url)
            domains[d] = domains.get(d, 0) + 1
    top = sorted(domains.items(), key=lambda x: -x[1])[:5]
    if top:
        table.add_row(
            "Top domains",
            ", ".join(f"{d} ({c})" for d, c in top),
        )

    console.print(table)
    console.print()
    console.print(f"[bold green]Report:[/bold green]  {result.report_path}")
    console.print(f"[bold green]Library:[/bold green] {result.library_path}")
    console.print(
        f"[bold green]Run log:[/bold green] {Path(result.report_path).parent / 'run.json'}"
    )


# ---------------------------------------------------------------------------
# research-pack status
# ---------------------------------------------------------------------------


@app.command()
def status(
    run_id: str = typer.Argument(..., help="Run ID to look up"),
    out: str = typer.Option(_DEFAULT_OUT, "--out", "-o", help="Output directory root"),
) -> None:
    """Check the status of a previous run."""
    run_json = _find_run(run_id, out)
    if run_json is None:
        console.print(f"[red]Run {run_id!r} not found in {out}[/red]")
        raise typer.Exit(code=1)

    data = json.loads(run_json.read_text())
    stats = data.get("stats", {})
    console.print(
        Panel(
            f"[bold]Run:[/bold]    {data['run_id']}\n"
            f"[bold]Topic:[/bold]  {data['topic']}\n"
            f"[bold]Status:[/bold] complete\n"
            f"[bold]Fetched:[/bold] {stats.get('fetched_count', '?')}  "
            f"[bold]Kept:[/bold] {stats.get('kept_count', '?')}  "
            f"[bold]Dupes:[/bold] {stats.get('duplicates_count', '?')}  "
            f"[bold]Failures:[/bold] {stats.get('failures_count', '?')}",
            title="Run Status",
            border_style="green",
        )
    )


# ---------------------------------------------------------------------------
# research-pack open
# ---------------------------------------------------------------------------


@app.command(name="open")
def open_cmd(
    run_id: str = typer.Argument(..., help="Run ID to look up"),
    out: str = typer.Option(_DEFAULT_OUT, "--out", "-o", help="Output directory root"),
) -> None:
    """Print paths to report and library for a completed run."""
    run_json = _find_run(run_id, out)
    if run_json is None:
        console.print(f"[red]Run {run_id!r} not found in {out}[/red]")
        raise typer.Exit(code=1)

    data = json.loads(run_json.read_text())
    console.print(f"[bold]Report:[/bold]  {data.get('report_path', 'n/a')}")
    console.print(f"[bold]Library:[/bold] {data.get('library_path', 'n/a')}")
    run_dir = run_json.parent
    console.print(f"[bold]Run dir:[/bold] {run_dir}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _find_run(run_id: str, out_dir: str) -> Path | None:
    """Locate ``run.json`` for a given *run_id* under *out_dir*."""
    root = Path(out_dir)
    if not root.exists():
        return None

    # direct match
    candidate = root / run_id / "run.json"
    if candidate.exists():
        return candidate

    # search
    for p in sorted(root.iterdir()):
        if p.is_dir() and run_id in p.name:
            rj = p / "run.json"
            if rj.exists():
                return rj
    return None


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
