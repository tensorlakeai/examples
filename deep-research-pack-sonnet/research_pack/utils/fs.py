"""Filesystem helpers for artifact management."""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime


def make_output_dir(base: Path, run_id: str) -> Path:
    """Create and return the output directory for a run."""
    out = base / run_id
    (out / "artifacts" / "sources").mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "html").mkdir(parents=True, exist_ok=True)
    (out / "artifacts" / "logs").mkdir(parents=True, exist_ok=True)
    return out


def write_text_artifact(out_dir: Path, source_id: str, text: str) -> str:
    """Write extracted text and return relative path."""
    path = out_dir / "artifacts" / "sources" / f"{source_id}.txt"
    path.write_text(text, encoding="utf-8")
    return str(path.relative_to(out_dir))


def write_html_artifact(out_dir: Path, source_id: str, html: str) -> str:
    """Write raw HTML and return relative path."""
    path = out_dir / "artifacts" / "html" / f"{source_id}.html"
    path.write_text(html, encoding="utf-8")
    return str(path.relative_to(out_dir))


def write_json(path: Path, obj) -> None:
    """Serialize pydantic model or dict to JSON file."""
    if hasattr(obj, "model_dump"):
        data = obj.model_dump(mode="json")
    else:
        data = obj
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")


def write_log(out_dir: Path, name: str, lines: list[str]) -> None:
    log_path = out_dir / "artifacts" / "logs" / name
    log_path.write_text("\n".join(lines), encoding="utf-8")
