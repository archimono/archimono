"""Regression test: orderly canonical generation survives deep site counts.

``OrderlyEnumerator.generate_canonical`` used to recurse once per site, so any
instance with more sites than the Python recursion limit (~1000) crashed with
``RecursionError`` even when the pruned search space was tiny (e.g. n_b=1).
"""

from __future__ import annotations

from archimono.assignment.enumeration.orderly import OrderlyEnumerator


def test_generate_canonical_on_deep_site_count() -> None:
    """1100 sites with the identity group: all C(1100, 1) assignments returned."""
    n = 1100
    identity_group = [tuple(range(n))]
    results = OrderlyEnumerator.generate_canonical(n, 1, identity_group)
    assert len(results) == n
    assert all(sum(row) == 1 for row in results)
