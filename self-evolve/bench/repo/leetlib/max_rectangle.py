"""LeetCode #85 — Maximal Rectangle.

Given a 2D binary matrix of '0' / '1' strings, find the area of the
largest rectangle containing only '1's.

The implementation below counts the longest run of '1's in any single
row — correct only for matrices with one usable row.
"""
from typing import List


def maximal_rectangle(matrix: List[List[str]]) -> int:
    if not matrix or not matrix[0]:
        return 0
    best = 0
    for row in matrix:
        run = 0
        for cell in row:
            if cell == "1":
                run += 1
                if run > best:
                    best = run
            else:
                run = 0
    return best
