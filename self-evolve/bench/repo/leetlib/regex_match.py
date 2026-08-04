"""LeetCode #10 — Regular Expression Matching.

`is_match(s, p)` returns True iff `s` is fully matched by pattern `p`, where:
  - `.`  matches any single character
  - `*`  matches zero or more of the preceding element

The match must cover the ENTIRE input (anchored both ends).

The implementation below is a naive greedy 2-pointer that gets the easy
cases right and fails when `*` needs to give back characters.
"""


def is_match(s: str, p: str) -> bool:
    i = j = 0
    while i < len(s) and j < len(p):
        if j + 1 < len(p) and p[j + 1] == "*":
            while i < len(s) and (s[i] == p[j] or p[j] == "."):
                i += 1
            j += 2
        elif p[j] == "." or s[i] == p[j]:
            i += 1
            j += 1
        else:
            return False
    return i == len(s) and j == len(p)
