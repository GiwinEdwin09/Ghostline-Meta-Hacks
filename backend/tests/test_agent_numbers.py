import numpy as np
import pytest

from ghostline.agent.numbers import (
    close_enough,
    matched_numbers,
    round_tree,
    sig_round,
    sorted_known,
    unmatched_numbers,
)

KNOWN = sorted_known([9.81, 0.012346, 0.65, 1.5e-3, 7.9896])


def test_numbers_match_to_the_precision_they_are_written_with():
    assert unmatched_numbers("g is 9.8, or 9.81 m/s^2; RMSE 0.0123.", KNOWN) == []
    assert unmatched_numbers("RMSE 0.0124 and 9.9", KNOWN) == ["0.0124", "9.9"]
    assert unmatched_numbers("about 1.5e-3 and -7.99", KNOWN) == []  # signs are ignored


def test_percentages_and_free_numbers():
    assert unmatched_numbers("the first 65% of the motion, 2 of 3 splits", KNOWN) == []
    assert unmatched_numbers("a 70% drop in 5 s", KNOWN) == ["70%"]
    assert unmatched_numbers("models P1, FF2 and S0 with theta_0 and 10^{-3}", KNOWN) == []
    assert unmatched_numbers("there were 13 runs", KNOWN) == ["13"]  # whole numbers past 12


def test_an_invented_ratio_is_not_a_harness_number():
    assert unmatched_numbers("S1 is 3.4 times worse", KNOWN) == ["3.4"]


def test_matched_numbers_skips_the_free_ones():
    assert matched_numbers("RMSE 7.99 over 3 splits", KNOWN) == ["7.99"]


def test_rounding_and_trees():
    assert sig_round(0.0123456, 3) == 0.0123 and sig_round(0.0, 5) == 0.0
    assert round_tree({"a": [1.234567, True, 3], "b": "x"}, 3) == {"a": [1.23, True, 3], "b": "x"}
    assert close_enough(9.8, 9.81) and not close_enough(9.9, 9.81)
    assert np.all(np.diff(KNOWN) >= 0)


@pytest.mark.parametrize("text", ["", "no numbers here"])
def test_text_without_numbers_passes(text):
    assert unmatched_numbers(text, KNOWN) == []
