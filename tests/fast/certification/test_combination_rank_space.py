"""Fast invariants for fixed-composition rank-space primitives."""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from archimono.assignment.solvers import _combinatorics as combinatorics

_SMALL_PAIRS = [(4, 2), (5, 2), (5, 3)]


@pytest.mark.parametrize(("n", "k"), _SMALL_PAIRS)
def test_unranked_combinations_follow_itertools_lexicographic_order(
    n: int,
    k: int,
) -> None:
    expected = list(combinations(range(n), k))

    actual = [
        combinatorics.unrank_combination(rank, n, k)
        for rank in range(len(expected))
    ]

    assert actual == expected


@pytest.mark.parametrize(("n", "k"), _SMALL_PAIRS)
def test_ranking_an_unranked_combination_recovers_its_original_rank(
    n: int,
    k: int,
) -> None:
    for rank in range(math.comb(n, k)):
        combination = combinatorics.unrank_combination(rank, n, k)

        assert combinatorics.rank_combination(combination, n) == rank


@pytest.mark.parametrize(("n", "k"), _SMALL_PAIRS)
def test_next_combination_matches_the_next_lexicographic_rank(n: int, k: int) -> None:
    total = math.comb(n, k)
    for rank in range(total):
        combination = combinatorics.unrank_combination(rank, n, k)
        expected = (
            combinatorics.unrank_combination(rank + 1, n, k)
            if rank + 1 < total
            else None
        )

        assert combinatorics.next_combination(combination, n) == expected



