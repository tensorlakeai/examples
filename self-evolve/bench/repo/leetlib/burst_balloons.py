"""LeetCode #312 — Burst Balloons.

Bursting balloon i scores `nums[i-1] * nums[i] * nums[i+1]` (treating
out-of-range as 1). Find the maximum total coins across all burst orders.

The implementation below is a greedy: burst the cheapest balloon first.
This is wrong — the right approach is interval DP, choosing which balloon
to burst LAST in each subrange.
"""
from typing import List


def max_coins(nums: List[int]) -> int:
    arr = [n for n in nums]
    total = 0
    while arr:
        # pick the smallest, burst it
        i = min(range(len(arr)), key=lambda k: arr[k])
        left = arr[i - 1] if i - 1 >= 0 else 1
        right = arr[i + 1] if i + 1 < len(arr) else 1
        total += left * arr[i] * right
        arr.pop(i)
    return total
