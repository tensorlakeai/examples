"""Near-duplicate detection: cosine similarity over TF-IDF vectors."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from research_pack.utils.logging import get_logger

logger = get_logger("dedupe")

_THRESHOLD = 0.85  # cosine similarity above this → near-duplicate
_MIN_TOKENS = 30   # ignore very short documents


@dataclass
class DedupeResult:
    kept_indices: list[int]          # indices of documents to keep
    duplicate_map: dict[int, int]    # dup_index → kept_index


def deduplicate(texts: list[str], threshold: float = _THRESHOLD) -> DedupeResult:
    """
    Given a list of document texts, find near-duplicates via TF-IDF cosine similarity.

    Returns:
        kept_indices    – list of indices of unique documents to keep
        duplicate_map   – mapping from duplicate index to the kept document's index
    """
    n = len(texts)
    if n == 0:
        return DedupeResult(kept_indices=[], duplicate_map={})
    if n == 1:
        return DedupeResult(kept_indices=[0], duplicate_map={})

    # Short docs are kept as-is (can't vectorize reliably)
    valid_mask = [len(t.split()) >= _MIN_TOKENS for t in texts]

    # Vectorize valid docs
    valid_texts = [t if valid_mask[i] else " " for i, t in enumerate(texts)]

    try:
        vectorizer = TfidfVectorizer(
            max_features=20_000,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
        )
        matrix = vectorizer.fit_transform(valid_texts)
    except Exception as exc:
        logger.warning("TF-IDF vectorization failed: %s — keeping all", exc)
        return DedupeResult(kept_indices=list(range(n)), duplicate_map={})

    sim = cosine_similarity(matrix)

    kept: set[int] = set()
    dup_map: dict[int, int] = {}

    for i in range(n):
        if i in dup_map:
            continue  # already marked as duplicate
        kept.add(i)
        for j in range(i + 1, n):
            if j in dup_map:
                continue
            # Short docs are never considered duplicates of each other
            if not valid_mask[i] or not valid_mask[j]:
                continue
            if sim[i, j] >= threshold:
                dup_map[j] = i
                logger.debug("Duplicate detected: %d ≈ %d (sim=%.3f)", j, i, sim[i, j])

    kept_indices = sorted(kept - set(dup_map.keys()))
    logger.info(
        "Dedup: %d documents → %d unique, %d duplicates removed",
        n, len(kept_indices), len(dup_map),
    )
    return DedupeResult(kept_indices=kept_indices, duplicate_map=dup_map)
