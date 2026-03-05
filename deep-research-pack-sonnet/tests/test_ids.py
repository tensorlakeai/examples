"""Tests for ID generation utilities."""

from research_pack.utils.ids import new_run_id, source_id


def test_run_id_format():
    rid = new_run_id()
    assert rid.startswith("run_")
    assert len(rid) == len("run_") + 12


def test_run_id_unique():
    ids = {new_run_id() for _ in range(100)}
    assert len(ids) == 100


def test_source_id():
    assert source_id(1) == "S1"
    assert source_id(10) == "S10"
    assert source_id(99) == "S99"
