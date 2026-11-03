"""Fast tests for combinatorial (un)ranking and rank-space slicing.

These back the HPC brute-force workflow: workers must jump directly to an
arbitrary rank in the ``C(n, k)`` index space and partition that space into
contiguous, non-overlapping, exhaustive slices.
"""

from __future__ import annotations

import math
from itertools import combinations

import pytest

from archimono.assignment.solvers import _combinatorics as comb

# Every (n, k) up to n = 12, including the k = 0 and k = n boundary cases, per
# the sliced brute-force certification spec's unranking round-trip requirement.
_CASES: list[tuple[int, int]] = [(n, k) for n in range(13) for k in range(n + 1)]


class TestUnrankMatchesItertools:
    """``unrank_combination`` reproduces ``itertools.combinations`` order."""

    @pytest.mark.parametrize(("n", "k"), _CASES)
    def test_full_enumeration_matches(self, n: int, k: int) -> None:
        """Unranking every rank reproduces the lexicographic enumeration."""
        expected = list(combinations(range(n), k))
        got = [comb.unrank_combination(r, n, k) for r in range(len(expected))]
        assert got == expected


class TestRankUnrankRoundtrip:
    """``rank_combination`` and ``unrank_combination`` are inverses."""

    @pytest.mark.parametrize(("n", "k"), _CASES)
    def test_roundtrip(self, n: int, k: int) -> None:
        """rank(unrank(r)) == r and unrank(rank(c)) == c for all ranks."""
        total = math.comb(n, k)
        for r in range(total):
            c = comb.unrank_combination(r, n, k)
            assert comb.rank_combination(c, n) == r
            assert comb.unrank_combination(comb.rank_combination(c, n), n, k) == c

    def test_unrank_rejects_out_of_range(self) -> None:
        """A rank outside ``[0, C(n, k))`` is rejected."""
        with pytest.raises(ValueError):
            comb.unrank_combination(math.comb(6, 3), 6, 3)
        with pytest.raises(ValueError):
            comb.unrank_combination(-1, 6, 3)


class TestNextCombination:
    """``next_combination`` advances through the lexicographic sequence."""

    @pytest.mark.parametrize(("n", "k"), [(6, 3), (8, 4)])
    def test_walks_full_sequence(self, n: int, k: int) -> None:
        """Stepping from the first combination reproduces the full order."""
        expected = list(combinations(range(n), k))
        current: tuple[int, ...] | None = expected[0]
        walked: list[tuple[int, ...]] = []
        while current is not None:
            walked.append(current)
            current = comb.next_combination(current, n)
        assert walked == expected

    def test_last_combination_has_no_successor(self) -> None:
        assert comb.next_combination((5, 6, 7), 8) is None


class TestBinomialTable:
    """``binomial_table`` matches ``math.comb`` entrywise."""

    def test_entries_match_math_comb(self) -> None:
        n = 12
        table = comb.binomial_table(n)
        assert table.shape == (n + 1, n + 1)
        for i in range(n + 1):
            for j in range(n + 1):
                assert int(table[i, j]) == math.comb(i, j)

    def test_n66_is_the_supported_int64_boundary(self) -> None:
        """n=66 succeeds and its central coefficient matches math.comb exactly."""
        table = comb.binomial_table(66)
        assert int(table[66, 33]) == math.comb(66, 33) == 7219428434016265740
        assert int(table[66, 33]) > 0

    def test_n67_raises_overflow_before_any_table_write(self) -> None:
        """n=67's central coefficient overflows int64; the table is never built."""
        with pytest.raises(OverflowError) as excinfo:
            comb.binomial_table(67)
        message = str(excinfo.value)
        assert "n=67" in message
        assert "C(67, 33)=14226520737620288370" in message
        assert str(comb._INT64_MAX) in message

    def test_check_int64_rank_capacity_matches_binomial_table_boundary(self) -> None:
        """The standalone capacity check agrees with binomial_table's guard."""
        comb.check_int64_rank_capacity(66)
        with pytest.raises(OverflowError):
            comb.check_int64_rank_capacity(67)



