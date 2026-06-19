"""Burnside's lemma counting and canonical-form orbit enumeration.

Burnside's lemma states that the number of distinct orbits of a finite set X
under a group G acting on it is::

    |X/G| = (1/|G|) × Σ_{g ∈ G} |Fix(g)|

where Fix(g) = { x ∈ X : g·x = x }.

This identity is also called the Cauchy-Frobenius lemma (Burnside attributed
it to Frobenius); "Burnside's lemma" is the conventional name used here.

For binary species assignments on n sites, X = {0,1}^n and G acts by
permuting site indices.  An assignment is fixed by a permutation g if and
only if every cycle of g carries a single species, which reduces the counting
to a subset-sum problem over cycle lengths.

References
----------
Burnside, W. (1897).
    *Theory of Groups of Finite Order*.
    Cambridge University Press.

Pólya, G. (1937).
    Kombinatorische Anzahlbestimmungen für Gruppen, Graphen und chemische
    Verbindungen.
    *Acta Mathematica*, 68, 145–254.
    https://doi.org/10.1007/BF02546665

For the application to crystal-structure enumeration:

    Hart, G. L. W. & Forcade, R. W. (2008).
    Algorithm for generating derivative structures.
    *Physical Review B*, 77(22), 224115.
    https://doi.org/10.1103/PhysRevB.77.224115
"""

from __future__ import annotations

_MAX_RAW = 10_000_000


def _validate_group(n: int, group: list[tuple[int, ...]]) -> None:
    """Check structural invariants of a symmetry group.

    Validates that *group* is non-empty, all permutations have length *n*
    and are bijections on ``range(n)``, the identity is present, and there
    are no duplicates.  Does **not** check closure or inverses (use the
    Burnside divisibility check for that).

    Args:
        n: Expected permutation length.
        group: Symmetry group as a list of permutation tuples.

    Raises:
        ValueError: If any structural invariant is violated.
    """
    if not group:
        raise ValueError("group must contain at least one permutation.")
    for g in group:
        if len(g) != n:
            raise ValueError(
                f"Permutation length {len(g)} does not match n={n}."
            )
        if sorted(g) != list(range(n)):
            raise ValueError(
                f"Permutation {g} is not a bijection on range({n})."
            )
    identity = tuple(range(n))
    group_set = set(group)
    if identity not in group_set:
        raise ValueError(
            "group does not contain the identity permutation; the input "
            "is not a group (did you pass a list of generators or an "
            "orbit?)."
        )
    if len(group_set) != len(group):
        raise ValueError(
            "group contains duplicate permutations; supply each group "
            "element exactly once."
        )


class BurnsideCounter:
    """Counting and enumeration of symmetry-inequivalent binary colorings.

    All methods are static; the class is a namespace for the related
    operations.
    """

    @staticmethod
    def get_cycles(perm: tuple[int, ...]) -> list[int]:
        """Return the cycle lengths of a permutation.

        Args:
            perm: A permutation of ``range(n)`` represented as a tuple such
                that ``perm[i]`` is the image of site *i*.

        Returns:
            Lengths of all cycles (including fixed points, which have
            length 1).

        Raises:
            ValueError: If *perm* is not a bijection on
                ``range(len(perm))``.
        """
        n = len(perm)
        if sorted(perm) != list(range(n)):
            raise ValueError(
                f"perm is not a bijection on range({n}): got {perm}."
            )
        visited = [False] * n
        cycles: list[int] = []
        for start in range(n):
            if not visited[start]:
                length = 0
                i = start
                while not visited[i]:
                    visited[i] = True
                    length += 1
                    i = perm[i]
                cycles.append(length)
        return cycles

    @staticmethod
    def count_fixed(perm: tuple[int, ...], n_b: int) -> int:
        """Count balanced assignments fixed by *perm*.

        An assignment with *n_b* label-1 sites is fixed by *perm* if and only
        if every cycle of *perm* is monochromatic (all-0 or all-1).  The count
        is computed by a knapsack DP over cycle lengths.

        Args:
            perm: Permutation of ``range(n)``.
            n_b: Number of label-1 atoms (= species[1]) (must satisfy
                ``0 <= n_b <= n``).

        Returns:
            Number of n_b-balanced assignments fixed by *perm*.

        Raises:
            ValueError: If *perm* is not a bijection on
                ``range(len(perm))`` or if ``n_b`` is not in
                ``[0, len(perm)]``.
        """
        n = len(perm)
        if not (0 <= n_b <= n):
            raise ValueError(f"n_b={n_b} out of valid range [0, {n}].")
        cycles = BurnsideCounter.get_cycles(perm)
        # dp[k] = number of ways to place exactly k label-1 sites using cycles
        # seen so far, where each cycle must be entirely 0 or entirely 1.
        dp: dict[int, int] = {0: 1}
        for size in cycles:
            new_dp: dict[int, int] = {}
            for j, cnt in dp.items():
                # assign this cycle to label 0
                new_dp[j] = new_dp.get(j, 0) + cnt
                # assign this cycle to label 1
                new_dp[j + size] = new_dp.get(j + size, 0) + cnt
            dp = new_dp
        return dp.get(n_b, 0)

    @staticmethod
    def count(n: int, n_b: int, group: list[tuple[int, ...]]) -> int:
        """Count inequivalent balanced colorings via Burnside's lemma.

        Args:
            n: Total number of sites.
            n_b: Number of label-1 atoms (= species[1]).  Must satisfy
                ``0 <= n_b <= n``.
            group: Symmetry group as a non-empty list of permutation
                tuples of length *n*.

        Returns:
            Number of symmetry-inequivalent balanced assignments.

        Raises:
            ValueError: If *group* fails structural validation
                (empty, wrong length, non-bijection, missing
                identity, or duplicates), *n_b* is out of range,
                or the Burnside sum is not divisible by ``|G|`` (which
                indicates the supplied permutations do not form a
                group).
        """
        _validate_group(n, group)
        if not (0 <= n_b <= n):
            raise ValueError(f"n_b={n_b} out of valid range [0, {n}].")

        total = sum(BurnsideCounter.count_fixed(g, n_b) for g in group)
        if total % len(group) != 0:
            raise ValueError(
                f"Burnside sum {total} is not divisible by |G|={len(group)}. "
                "The supplied permutations likely do not form a group."
            )
        return total // len(group)

    @staticmethod
    def canonical_form(
        assignment: tuple[int, ...],
        group: list[tuple[int, ...]],
    ) -> tuple[int, ...]:
        """Return the lexicographically smallest orbit representative.

        Delegates to :func:`~archimono.assignment.enumeration.canonical.canonical_form`.

        Args:
            assignment: A binary tuple of length *n*
                (0 = ``species[0]``, 1 = ``species[1]``).
            group: Symmetry group as a list of permutation tuples.

        Returns:
            The canonical (lex-minimum) representative of the orbit.
        """
        from archimono.assignment.enumeration.canonical import canonical_form

        return canonical_form(assignment, group)

    @staticmethod
    def enumerate_inequivalent(
        n: int,
        n_b: int,
        group: list[tuple[int, ...]],
        *,
        override: bool = False,
    ) -> list[tuple[int, ...]]:
        """Enumerate one representative per symmetry orbit.

        Delegates to
        :func:`~archimono.assignment.enumeration.bruteforce.enumerate_inequivalent`.

        Args:
            n: Total number of sites.
            n_b: Number of label-1 atoms (= species[1]).
            group: Symmetry group as a list of permutation tuples.
            override: Bypass the safety limit on raw enumeration
                space.

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
        from archimono.assignment.enumeration.bruteforce import (
            enumerate_inequivalent as _enumerate,
        )

        return _enumerate(n, n_b, group, override=override)
