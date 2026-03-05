"""Tests for deduplication logic."""

from research_pack.dedupe.similarity import (
    cosine_similarity,
    find_duplicates,
    tokenize,
)


def test_tokenize():
    tokens = tokenize("Hello, World! 123")
    assert tokens == ["hello", "world", "123"]


def test_cosine_identical():
    v = {"a": 1.0, "b": 2.0}
    assert abs(cosine_similarity(v, v) - 1.0) < 1e-9


def test_cosine_orthogonal():
    v1 = {"a": 1.0}
    v2 = {"b": 1.0}
    assert cosine_similarity(v1, v2) == 0.0


def test_find_duplicates_identical():
    texts = [
        "The quick brown fox jumps over the lazy dog",
        "The quick brown fox jumps over the lazy dog",
    ]
    dupes = find_duplicates(texts, threshold=0.8)
    assert len(dupes) == 1
    assert dupes[0] == (1, 0)


def test_find_duplicates_distinct():
    texts = [
        "Quantum computing uses qubits and superposition for parallel computation",
        "The recipe calls for flour sugar eggs and butter mixed together",
    ]
    dupes = find_duplicates(texts, threshold=0.8)
    assert dupes == []


def test_find_duplicates_near():
    base = "Machine learning is a subset of artificial intelligence that enables systems to learn from data"
    variant = "Machine learning is a branch of artificial intelligence enabling systems to learn from datasets"
    dupes = find_duplicates([base, variant], threshold=0.5)
    assert len(dupes) == 1


def test_find_duplicates_single():
    assert find_duplicates(["only one"]) == []
