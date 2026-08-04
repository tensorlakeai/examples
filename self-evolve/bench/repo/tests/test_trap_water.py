from leetlib import trap


def test_trap_v1_empty():
    assert trap([]) == 0


def test_trap_v2_flat():
    assert trap([1, 1, 1, 1]) == 0


def test_trap_v3_simple_valley():
    # trapped = 1 (between the 2s)
    assert trap([2, 0, 2]) == 2


def test_trap_v4_classic_leetcode():
    # canonical example
    assert trap([0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1]) == 6


def test_trap_v5_descending_then_ascending():
    # The buggy left-only impl over-counts on the descending tail.
    assert trap([4, 2, 0, 3, 2, 5]) == 9


def test_trap_v6_max_at_end():
    # Buggy: thinks indices 0..3 trap up to max_left=2, but real max is 5 on the right.
    assert trap([2, 1, 0, 1, 5]) == 5
