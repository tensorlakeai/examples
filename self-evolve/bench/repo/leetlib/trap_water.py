"""LeetCode #42 — Trapping Rain Water.

Given non-negative integers representing an elevation map, compute how
much rainwater is trapped between bars.

The implementation below considers only the running max from the left,
forgetting that water level at index i is bounded by min(max_left, max_right).
"""
from typing import List


def trap(heights: List[int]) -> int:
    if not heights:
        return 0
    total = 0
    max_left = 0
    for h in heights:
        if h > max_left:
            max_left = h
        else:
            total += max_left - h
    return total
