"""Fast smoke tests for enumeration backends.

Verifies enumerators return correct types and non-empty results on trivial inputs.
"""

from __future__ import annotations

from archimono.assignment import IcetEnumerator
from archimono.assignment.enumeration.burnside import BurnsideCounter
from archimono.assignment.enumeration.orderly import OrderlyEnumerator
from archimono.tilings import registry


class TestEnumeratorReturnTypes:
    """Enumerators return lists of valid type on minimal inputs."""

    def test_icet_returns_list(self) -> None:
        """IcetEnumerator.enumerate() returns a list."""
        tiling = registry.get("triangular")
        configs = IcetEnumerator().enumerate(tiling, n_b=2, supercell=(2, 2))
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_orderly_returns_list(self) -> None:
        """OrderlyEnumerator.enumerate() returns a list."""
        tiling = registry.get("triangular")
        configs = OrderlyEnumerator().enumerate(tiling, n_b=2, supercell=(2, 2))
        assert isinstance(configs, list)
        assert len(configs) > 0

    def test_burnside_returns_int(self) -> None:
        """BurnsideCounter.count() returns a positive integer."""
        tiling = registry.get("triangular")
        group = IcetEnumerator().get_symmetry_group(tiling, supercell=(2, 2))
        n = tiling.n_vertices * 2 * 2
        count = BurnsideCounter.count(n, 2, group)
        assert isinstance(count, int)
        assert count > 0


class TestEnumeratorConsistency:
    """Quick sanity: all backends agree on a tiny case."""

    def test_triangular_2x2_nb2_all_agree(self) -> None:
        """All three backends produce the same count on triangular 2x2 n_b=2."""
        tiling = registry.get("triangular")
        supercell = (2, 2)
        n_b = 2
        n = tiling.n_vertices * supercell[0] * supercell[1]

        icet_count = len(
            IcetEnumerator().enumerate(tiling, n_b=n_b, supercell=supercell)
        )
        orderly_count = len(
            OrderlyEnumerator().enumerate(
                tiling, n_b=n_b, supercell=supercell
            )
        )
        group = IcetEnumerator().get_symmetry_group(tiling, supercell=supercell)
        burnside_count = BurnsideCounter.count(n, n_b, group)

        assert icet_count == orderly_count == burnside_count
