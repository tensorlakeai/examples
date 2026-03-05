"""Structured logging setup for Research Pack."""

from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


class _JSONFormatter(logging.Formatter):
    """Emit each record as a single-line JSON object."""

    def format(self, record: logging.LogRecord) -> str:
        entry = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        if record.exc_info and record.exc_info[0] is not None:
            entry["exc"] = self.formatException(record.exc_info)
        return json.dumps(entry, default=str)


def get_logger(
    name: str = "research_pack",
    log_dir: Optional[Path] = None,
    level: int = logging.INFO,
    console_output: bool = False,
) -> logging.Logger:
    """Return a logger that writes structured JSON.

    If *log_dir* is given a ``run.log`` file handler is added.
    Set *console_output* to ``True`` to also emit to stderr (off by
    default so the CLI Rich output stays clean).
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    logger.propagate = False

    fmt = _JSONFormatter()

    if console_output:
        console = logging.StreamHandler(sys.stderr)
        console.setFormatter(fmt)
        logger.addHandler(console)

    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / "run.log", encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    # If no handlers were added, add a NullHandler to avoid
    # "No handlers could be found" warnings.
    if not logger.handlers:
        logger.addHandler(logging.NullHandler())

    return logger
