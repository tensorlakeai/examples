"""Near-duplicate detection via TF-IDF cosine similarity.

No external dependencies — uses only the Python standard library.
"""

from __future__ import annotations

import math
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenisation."""
    return re.findall(r"\b\w+\b", text.lower())


def compute_tfidf(documents: list[str]) -> list[dict[str, float]]:
    """Return a list of sparse TF-IDF vectors (one per document)."""
    doc_tokens = [tokenize(doc) for doc in documents]

    # document frequency
    df: dict[str, int] = {}
    for tokens in doc_tokens:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1

    n_docs = len(documents)
    idf = {tok: math.log(1 + n_docs / (1 + cnt)) for tok, cnt in df.items()}

    vectors: list[dict[str, float]] = []
    for tokens in doc_tokens:
        tf = Counter(tokens)
        n = max(len(tokens), 1)
        vectors.append({tok: (cnt / n) * idf.get(tok, 0.0) for tok, cnt in tf.items()})
    return vectors


def cosine_similarity(v1: dict[str, float], v2: dict[str, float]) -> float:
    """Cosine similarity between two sparse vectors."""
    common = v1.keys() & v2.keys()
    if not common:
        return 0.0
    dot = sum(v1[k] * v2[k] for k in common)
    norm1 = math.sqrt(sum(v * v for v in v1.values()))
    norm2 = math.sqrt(sum(v * v for v in v2.values()))
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot / (norm1 * norm2)


def find_duplicates(
    texts: list[str],
    threshold: float = 0.85,
) -> list[tuple[int, int]]:
    """Find near-duplicate pairs.

    Returns ``[(duplicate_idx, original_idx), ...]``.  The first
    occurrence of a cluster is treated as the *original*.
    """
    if len(texts) <= 1:
        return []

    vectors = compute_tfidf(texts)
    already_dup: set[int] = set()
    duplicates: list[tuple[int, int]] = []

    for i in range(len(texts)):
        if i in already_dup:
            continue
        for j in range(i + 1, len(texts)):
            if j in already_dup:
                continue
            if cosine_similarity(vectors[i], vectors[j]) >= threshold:
                duplicates.append((j, i))
                already_dup.add(j)

    return duplicates
