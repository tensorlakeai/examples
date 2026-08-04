from leetlib import min_distance


def test_edit_v1_identical():
    assert min_distance("abc", "abc") == 0


def test_edit_v2_one_replace():
    assert min_distance("abc", "abd") == 1


def test_edit_v3_one_insert():
    assert min_distance("abc", "abdc") == 1


def test_edit_v4_horse_ros():
    # Classic LeetCode example: horse -> ros = 3
    assert min_distance("horse", "ros") == 3


def test_edit_v5_intention_execution():
    # intention -> execution = 5
    assert min_distance("intention", "execution") == 5


def test_edit_v6_one_empty():
    assert min_distance("", "abcde") == 5
