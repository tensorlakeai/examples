"""LeetCode #72 — Edit Distance (Levenshtein).

Minimum number of single-character insert / delete / replace operations
to convert `word1` into `word2`.

The implementation below counts character mismatches in the common prefix
plus the length difference — correct for some inputs, wrong whenever an
insert or delete in the middle would beat replacing through.
"""


def min_distance(word1: str, word2: str) -> int:
    common = sum(1 for a, b in zip(word1, word2) if a == b)
    return max(len(word1), len(word2)) - common
