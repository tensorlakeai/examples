from leetlib import max_coins


def test_burst_v1_empty():
    assert max_coins([]) == 0


def test_burst_v2_single():
    assert max_coins([5]) == 5  # 1 * 5 * 1


def test_burst_v3_two():
    # Order [3,1,5,8] doesn't apply; just two: greedy and optimal agree.
    assert max_coins([3, 5]) == 1 * 3 * 5 + 1 * 5 * 1  # = 15 + 5 = 20


def test_burst_v4_classic_leetcode():
    # Canonical example. Optimal = 167; greedy gives lower.
    assert max_coins([3, 1, 5, 8]) == 167


def test_burst_v5_descending():
    # [9, 76, 64] — greedy bursts 9 first; optimal bursts 76 last.
    assert max_coins([9, 76, 64]) == 9 * 76 * 64 + 1 * 9 * 64 + 1 * 64 * 1


def test_burst_v6_fives():
    # All same value — every order ties; both greedy and optimal == 5*5*5 + 5*5*1 + 5*1*1
    assert max_coins([5, 5, 5]) == 5 * 5 * 5 + 5 * 5 * 1 + 5 * 1 * 1
