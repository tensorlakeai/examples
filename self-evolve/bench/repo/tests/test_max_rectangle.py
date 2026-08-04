from leetlib import maximal_rectangle


def test_rect_v1_empty():
    assert maximal_rectangle([]) == 0


def test_rect_v2_all_zeros():
    assert maximal_rectangle([["0", "0"], ["0", "0"]]) == 0


def test_rect_v3_single_row():
    assert maximal_rectangle([["1", "1", "0", "1"]]) == 2


def test_rect_v4_classic_leetcode():
    matrix = [
        ["1", "0", "1", "0", "0"],
        ["1", "0", "1", "1", "1"],
        ["1", "1", "1", "1", "1"],
        ["1", "0", "0", "1", "0"],
    ]
    assert maximal_rectangle(matrix) == 6


def test_rect_v5_tall_column():
    # Single tall column of '1's — buggy single-row impl returns 1.
    matrix = [["1"], ["1"], ["1"], ["1"]]
    assert maximal_rectangle(matrix) == 4


def test_rect_v6_2x2_block():
    matrix = [
        ["0", "1", "1", "0"],
        ["0", "1", "1", "0"],
    ]
    assert maximal_rectangle(matrix) == 4
