"""Regression tests for the HNF candidate-generation bound (issue #85).

Before the fix, ``valid_supercell_matrices`` bounded its shear parameter as
``0 <= b < d`` instead of ``0 <= b < a``, which produced the right candidate
*count* (``sigma_1(k)``) but the wrong *set* -- duplicating some sublattices
while omitting others (e.g. the triangular-net k=7 shear ``[[7,2],[0,1]]``).
"""

from __future__ import annotations

import itertools

import numpy as np
from numpy.typing import NDArray

from archimono.tilings import registry
from archimono.tilings.base import valid_supercell_matrices


def _sigma_1(k: int) -> int:
    """Sum of divisors of *k* -- the count of index-*k* sublattices."""
    return sum(a for a in range(1, k + 1) if k % a == 0)


def _brute_force_hnf_matrices(k: int) -> list[NDArray[np.intp]]:
    """Enumerate all upper-triangular HNF matrices of determinant *k*.

    Independent sweep over every ``(a, b, d)`` with ``a * d == k`` and
    ``0 <= b < a``, used to check completeness against
    :func:`valid_supercell_matrices`'s internal candidate generator.
    """
    matrices: list[NDArray[np.intp]] = []
    for a in range(1, k + 1):
        if k % a != 0:
            continue
        d = k // a
        for b in range(a):
            matrices.append(np.array([[a, b], [0, d]], dtype=np.intp))
    return matrices


def _are_right_unimodular_equivalent(
    s1: NDArray[np.intp], s2: NDArray[np.intp]
) -> bool:
    """Return whether ``s2 == s1 @ U`` for some integer unimodular ``U``."""
    u = np.linalg.inv(s1.astype(np.float64)) @ s2.astype(np.float64)
    u_round = np.round(u)
    if not np.allclose(u, u_round, atol=1e-9):
        return False
    return abs(round(float(np.linalg.det(u_round)))) == 1


def test_determinant_2_gives_exactly_three_distinct_sublattices() -> None:
    """Ground truth: sigma_1(2) = 3 distinct index-2 column lattices."""
    candidates = _brute_force_hnf_matrices(2)
    assert len(candidates) == 3

    for s1, s2 in itertools.combinations(candidates, 2):
        assert not _are_right_unimodular_equivalent(s1, s2), (
            f"{s1.tolist()} and {s2.tolist()} are equivalent -- not distinct"
        )


def test_triangular_net_admits_seven_site_supercell() -> None:
    """The triangular net has a valid k=7 shear supercell, S=[[7,2],[0,1]]."""
    tiling = registry.get("triangular")
    target = np.array([[7, 2], [0, 1]], dtype=np.intp)

    candidates = valid_supercell_matrices(tiling, target_n=7 * tiling.n_vertices)
    assert any(
        np.array_equal(c, target) for c in candidates
    ), "S=[[7,2],[0,1]] is missing from valid_supercell_matrices(k=7)"

    graph = tiling.graph(target)
    assert graph.number_of_nodes() == 7
    assert graph.number_of_edges() == 21


def test_candidate_count_matches_sigma_1_for_small_indices() -> None:
    """The raw (pre-filter) candidate generator returns exactly sigma_1(k)."""
    for k in range(1, 13):
        candidates = _brute_force_hnf_matrices(k)
        assert len(candidates) == _sigma_1(k), f"k={k}"


def test_candidates_are_pairwise_inequivalent_for_small_indices() -> None:
    """No two raw candidates at the same k are right-unimodular equivalent."""
    for k in range(1, 9):
        candidates = _brute_force_hnf_matrices(k)
        for s1, s2 in itertools.combinations(candidates, 2):
            assert not _are_right_unimodular_equivalent(s1, s2), (
                f"k={k}: {s1.tolist()} ~ {s2.tolist()}"
            )


def test_candidates_are_complete_against_brute_force_sweep() -> None:
    """Every column matrix of determinant k is equivalent to some candidate."""
    for k in range(1, 9):
        candidates = _brute_force_hnf_matrices(k)
        # Brute-force sweep over a bounded window of raw integer matrices
        # with det == k, independent of the HNF-specific generator.
        bound = k + 1
        for a in range(-bound, bound + 1):
            for b in range(-bound, bound + 1):
                for c in range(-bound, bound + 1):
                    for d in range(-bound, bound + 1):
                        s = np.array([[a, b], [c, d]], dtype=np.intp)
                        if round(float(np.linalg.det(s))) != k:
                            continue
                        assert any(
                            _are_right_unimodular_equivalent(s, cand)
                            for cand in candidates
                        ), f"k={k}: {s.tolist()} not covered by any candidate"


def test_valid_supercell_matrices_reconstructs_the_quotient_graph() -> None:
    """Representative matrices build the intended Cartesian quotient graph.

    Uses a tiling with no PBC edge-aliasing degeneracy at small supercells
    (see issue #78) so every candidate's edge count reflects the full
    physical connectivity, independent of that separate defect.
    """
    tiling = registry.get("truncated_trihexagonal")
    target_n = 7 * tiling.n_vertices
    candidates = valid_supercell_matrices(tiling, target_n=target_n)
    assert candidates
    for s in candidates:
        graph = tiling.graph(s)
        assert graph.number_of_nodes() == target_n
        assert graph.number_of_edges() == target_n * tiling.coordination // 2
