"""Tests for near-duplicate detection."""

import pytest
from research_pack.dedupe.similarity import deduplicate


def test_empty():
    result = deduplicate([])
    assert result.kept_indices == []
    assert result.duplicate_map == {}


def test_single():
    result = deduplicate(["hello world this is a document with enough words to be valid"])
    assert result.kept_indices == [0]
    assert result.duplicate_map == {}


def test_identical_documents():
    text = (
        "The quick brown fox jumps over the lazy dog. " * 10
    )
    result = deduplicate([text, text])
    assert len(result.kept_indices) == 1
    assert len(result.duplicate_map) == 1


def test_different_documents():
    text_a = "Python is a programming language known for its simplicity. " * 8
    text_b = "The Great Wall of China is visible from space on a clear day. " * 8
    result = deduplicate([text_a, text_b])
    assert len(result.kept_indices) == 2
    assert len(result.duplicate_map) == 0


def test_near_duplicate():
    # Use a lower threshold so that a very similar pair is caught.
    # One sentence repeated 10x; near_dup differs by ~2 words out of ~80 tokens.
    base = (
        "Machine learning is reshaping how we develop, deploy and maintain "
        "modern software applications at enterprise scale. " * 10
    )
    # Near duplicate: only a few words swapped
    near_dup = base.replace("reshaping", "transforming").replace("maintain", "manage")
    unique = "Quantum physics explores the behavior of subatomic particles. " * 10
    result = deduplicate([base, near_dup, unique], threshold=0.70)
    # near_dup should be flagged as duplicate of base
    assert len(result.kept_indices) == 2
    assert len(result.duplicate_map) == 1


def test_short_docs_not_deduped():
    """Very short documents below token threshold should be kept as-is."""
    short_a = "Hi there"
    short_b = "Hi there"
    result = deduplicate([short_a, short_b])
    # Both are below min_tokens threshold — treated as unique
    assert len(result.kept_indices) == 2
