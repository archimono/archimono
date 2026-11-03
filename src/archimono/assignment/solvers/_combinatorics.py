"""Combinatorial (un)ranking and rank-space slicing for brute-force search.

The fixed-composition brute-force search ranges over all ``C(n, k)`` subsets
of size ``k`` (the ``1``-labelled nodes). To distribute that search across HPC
workers, each worker must jump directly to an arbitrary position in the
lexicographic enumeration without enumerating its predecessors. This module
provides that primitive by counting the subsets skipped at each position.

References:
    Knuth, D. E. (2011). *The Art of Computer Programming, Vol. 4A:
    Combinatorial Algorithms, Part 1*, §7.2.1.3 ("Generating all
    combinations"), Addison-Wesley.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

_INT64_MAX = np.iinfo(np.int64).max


def check_int64_rank_capacity(n: int) -> None:
    """Raise if the rank space for a ground set of size *n* exceeds int64.

    The largest value any ``C(n, k)`` can take is the central coefficient
    ``C(n, n // 2)``; if that fits in signed ``int64``, every rank and every
    binomial-table entry for this *n* does too.

    Raises:
        OverflowError: If ``C(n, n // 2)`` exceeds
            ``np.iinfo(np.int64).max``.
    """
    k = n // 2
    max_coefficient = math.comb(n, k)
    if max_coefficient > _INT64_MAX:
        raise OverflowError(
            f"Combination rank space for n={n} exceeds signed int64 "
            f"capacity: C({n}, {k})={max_coefficient} > {_INT64_MAX}."
        )


def unrank_combination(rank: int, n: int, k: int) -> tuple[int, ...]:
    """Return the ``rank``-th ``k``-subset of ``{0, ..., n-1}`` in lex order.

    Lexicographic order matches :func:`itertools.combinations`, so
    ``unrank_combination(r, n, k)`` equals the ``r``-th tuple it would yield.

    Returns:
        The ``rank``-th ``k``-combination as an ascending tuple of indices.

    Raises:
        ValueError: If ``rank`` is outside ``[0, C(n, k))``.
    """
    total = math.comb(n, k)
    if rank < 0 or rank >= total:
        raise ValueError(
            f"rank must satisfy 0 <= rank < C({n}, {k}) = {total}; got {rank}."
        )
    result: list[int] = []
    remaining = rank
    x = 0
    for i in range(k):
        c = x
        # Count combinations whose next chosen element is exactly ``c``; skip
        # whole blocks until ``remaining`` falls within the current block.
        while True:
            block = math.comb(n - c - 1, k - i - 1)
            if remaining < block:
                break
            remaining -= block
            c += 1
        result.append(c)
        x = c + 1
    return tuple(result)


def rank_combination(combo: Sequence[int], n: int) -> int:
    """Return the lexicographic rank of ``combo`` among ``k``-subsets.

    Inverse of :func:`unrank_combination`: ``rank_combination(c, n)`` recovers
    the position ``r`` such that ``unrank_combination(r, n, len(c)) == c``.

    Args:
        combo: Ascending sequence of distinct indices drawn from
            ``{0, ..., n-1}``.
        n: Size of the ground set.
    """
    k = len(combo)
    rank = 0
    x = 0
    for i, c in enumerate(combo):
        for j in range(x, c):
            rank += math.comb(n - j - 1, k - i - 1)
        x = c + 1
    return rank


def next_combination(combo: Sequence[int], n: int) -> tuple[int, ...] | None:
    """Return the lexicographic successor of ``combo``, or ``None`` if last.

    Advancing from the first combination reproduces the same order as
    :func:`itertools.combinations`, one step at a time, without rebuilding the
    rank.

    Args:
        combo: Ascending sequence of distinct indices drawn from
            ``{0, ..., n-1}``.
        n: Size of the ground set.
    """
    k = len(combo)
    c = list(combo)
    i = k - 1
    while i >= 0 and c[i] == n - k + i:
        i -= 1
    if i < 0:
        return None
    c[i] += 1
    for j in range(i + 1, k):
        c[j] = c[j - 1] + 1
    return tuple(c)


def binomial_table(n: int) -> npt.NDArray[np.int64]:
    """Return Pascal's triangle ``C[i, j] = comb(i, j)`` up to ``n``.

    Precomputing the table lets Numba kernels unrank combinations with array
    lookups instead of calling :func:`math.comb`, which is unsupported in
    nopython mode.

    Returns:
        An ``(n + 1, n + 1)`` integer array with ``C[i, j]`` for ``j <= i`` and
        zeros above the diagonal.

    Raises:
        OverflowError: If the central coefficient ``C(n, n // 2)`` exceeds
            signed ``int64`` capacity (first occurs at ``n = 67``).
    """
    check_int64_rank_capacity(n)
    table = np.zeros((n + 1, n + 1), dtype=np.int64)
    for i in range(n + 1):
        table[i, 0] = 1
        for j in range(1, i + 1):
            table[i, j] = table[i - 1, j - 1] + table[i - 1, j]
    return table
