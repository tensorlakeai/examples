from leetlib import is_match


def test_regex_v1_literal():
    assert is_match("abc", "abc") is True


def test_regex_v2_dot():
    assert is_match("abc", "a.c") is True


def test_regex_v3_star_zero():
    assert is_match("ab", "ab*") is True


def test_regex_v4_a_star_a_classic():
    # Greedy 2-pointer eats all `a`s with `a*`, no chars left for trailing `a`.
    assert is_match("aaa", "a*a") is True


def test_regex_v5_mississippi_neg():
    assert is_match("mississippi", "mis*is*p*.") is False


def test_regex_v6_mississippi_pos():
    assert is_match("mississippi", "mis*is*ip*.") is True
