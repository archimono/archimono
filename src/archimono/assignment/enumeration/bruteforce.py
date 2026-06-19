"""Brute-force orbit enumeration for binary colorings.

Iterates over all C(n, n_b) balanced assignments and keeps one
representative per distinct canonical form.  This is the simplest
correct algorithm but intractable for n > ~20.  Prefer
:class:`~archimono.assignment.enumeration.orderly.OrderlyEnumerator`
for larger instances.
"""

from __future__ import annotations

import math
from itertools import combinations

import archimono.assignment.enumeration.burnside as _burnside_mod
from archimono.assignment.enumeration.burnside import _validate_group
from archimono.assignment.enumeration.canonical import canonical_form


def enumerate_inequivalent(
    n: int,
    n_b: int,
    group: list[tuple[int, ...]],
    *,
    override: bool = False,
) -> list[tuple[int, ...]]:
    """Enumerate one representative per symmetry orbit by brute force.

    Iterates over all C(n, n_b) balanced assignments and keeps the
    first representative encountered for each distinct canonical
    form.

    Args:
        n: Total number of sites.
        n_b: Number of label-1 atoms (= species[1]).
        group: Symmetry group as a list of permutation tuples.
        override: Bypass the safety limit on raw enumeration space.

    Returns:
        One assignment per symmetry-inequivalent orbit.  The
        returned assignments are not necessarily in canonical
        form; they are the first lexicographic representative
        encountered.

    Raises:
        ValueError: If *group* fails structural validation,
            *n_b* is outside ``[0, n]``, or C(n, n_b) exceeds
            the safety limit and *override* is ``False``.
    """
    _validate_group(n, group)
    if not (0 <= n_b <= n):
        raise ValueError(f"n_b={n_b} out of valid range [0, {n}].")

    n_choose_nb = math.comb(n, n_b)
    max_raw = _burnside_mod._MAX_RAW
    if not override and n_choose_nb > max_raw:
        raise ValueError(
            f"Raw enumeration space C({n}, {n_b}) = {n_choose_nb:,} "
            f"exceeds the {max_raw:,}-structure safety limit.  "
            f"Pass override=True only if you have confirmed the "
            f"enumeration is tractable."
        )

    seen: set[tuple[int, ...]] = set()
    representatives: list[tuple[int, ...]] = []

    for combo in combinations(range(n), n_b):
        buf = [0] * n
        for i in combo:
            buf[i] = 1
        assignment = tuple(buf)
        canon = canonical_form(assignment, group)
        if canon not in seen:
            seen.add(canon)
            representatives.append(assignment)

    return representatives
