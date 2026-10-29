"""Rounding what Wonder is shown, and checking that every number it writes was shown to it.

Part 13: "Every number in `submit_conclusion` must match a harness result within rounding.
Wonder can't introduce a number the harness didn't produce." The orchestrator therefore keeps
the list of numbers it has put in front of Wonder (`known`), and `unmatched_numbers` finds the
numbers in a piece of Wonder's text that are not on it.

A number in text matches a known value when they agree to the precision it was written with
("9.8" matches 9.81 and 9.76, "0.0123" matches 0.012346 but not 0.0124). A trailing "%" divides
by 100. Whole numbers up to 12 are free (counts, "2 of 3 splits", exponents), and so are digits
glued to a letter or underscore (model ids such as P1 or FF2, subscripts). Anything else, such as
a ratio or a difference Wonder computed itself, has to be quoted from the harness instead.
"""

from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

import numpy as np

SHOWN_DIGITS = 5
FREE_INTEGER_MAX = 12
_NUMBER = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)(?!\d)(%?)")


def sig_round(x: float, digits: int = SHOWN_DIGITS) -> float:
    """Round to `digits` significant digits. NaN, inf and zero pass through."""
    if not math.isfinite(x) or x == 0.0:
        return x
    return round(x, digits - 1 - math.floor(math.log10(abs(x))))


def round_tree(value: Any, digits: int = SHOWN_DIGITS) -> Any:
    """A copy of a JSON-like tree with every float rounded to `digits` significant digits."""
    if isinstance(value, bool):
        return value
    if isinstance(value, float):
        return sig_round(value, digits)
    if isinstance(value, dict):
        return {k: round_tree(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [round_tree(v, digits) for v in value]
    return value


def floats_in(value: Any) -> Iterable[float]:
    """Every finite number (int or float, not bool) in a JSON-like tree."""
    if isinstance(value, bool):
        return
    if isinstance(value, int | float):
        if math.isfinite(value):
            yield float(value)
    elif isinstance(value, dict):
        for v in value.values():
            yield from floats_in(v)
    elif isinstance(value, list | tuple):
        for v in value:
            yield from floats_in(v)


def _tolerance(token: str) -> float:
    """Half a unit in the last place the token was written to."""
    mantissa, _, exponent = token.lower().partition("e")
    exp = int(exponent) if exponent else 0
    decimals = len(mantissa.partition(".")[2])
    return 0.5 * 10.0 ** (exp - decimals)


def _is_free(token: str, as_percent: bool) -> bool:
    return (
        not as_percent
        and token.isdigit()
        and (len(token) == 1 or not token.startswith("0"))
        and int(token) <= FREE_INTEGER_MAX
    )


def matches(token: str, known: np.ndarray, *, percent: bool = False) -> bool:
    """Whether `token` agrees, to its own precision, with a value in the sorted array `known`."""
    value = abs(float(token))
    tol = _tolerance(token)
    if percent:
        value, tol = value / 100.0, tol / 100.0
    slack = tol * (1 + 1e-9) + 1e-12 * max(value, 1e-300)
    lo = np.searchsorted(known, value - slack, side="left")
    return bool(lo < known.size and known[lo] <= value + slack)


def sorted_known(values: Iterable[float]) -> np.ndarray:
    return np.sort(np.abs(np.fromiter(values, dtype=float)))


def unmatched_numbers(text: str, known: np.ndarray) -> list[str]:
    """The numbers in `text` that are neither free nor on the known list, as written."""
    bad: list[str] = []
    for match in _NUMBER.finditer(text):
        token, percent = match.group(1), bool(match.group(2))
        if _is_free(token, percent) or matches(token, known, percent=percent):
            continue
        written = token + ("%" if percent else "")
        if written not in bad:
            bad.append(written)
    return bad


def matched_numbers(text: str, known: np.ndarray) -> list[str]:
    """The numbers in `text` (leaving out the free whole numbers) that are on the known list."""
    found: list[str] = []
    for match in _NUMBER.finditer(text):
        token, percent = match.group(1), bool(match.group(2))
        if not _is_free(token, percent) and matches(token, known, percent=percent):
            found.append(token + ("%" if percent else ""))
    return found


def close_enough(claimed: float, actual: float, rel: float = 2e-3) -> bool:
    """A number in a JSON argument against its harness value: equal to a few significant digits,
    so rounding to 3 or 4 digits is fine."""
    scale = max(abs(actual), 1e-300)
    return abs(claimed - actual) <= rel * scale + 1e-12
