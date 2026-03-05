"""Run and source ID generation."""

import uuid


def new_run_id() -> str:
    return f"run_{uuid.uuid4().hex[:12]}"


def source_id(index: int) -> str:
    """Return citation-friendly source ID like S1, S2, ..."""
    return f"S{index}"
