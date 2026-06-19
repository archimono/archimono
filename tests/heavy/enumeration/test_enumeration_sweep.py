"""Heavy enumeration tests: all backends x all tilings at 2x2.

Verifies backend agreement across the full tiling set.
"""

from __future__ import annotations

from math import comb

import pytest

from archimono.assignment import IcetEnumerator
from archimono.assignment.enumeration.burnside import BurnsideCounter
from archimono.assignment.enumeration.orderly import OrderlyEnumerator
from archimono.tilings import registry

ALL_TILING_KEYS = sorted(registry.available())

# Build sweep cases: all tilings at 2x2, choosing n_b to keep tractable.
_SWEEP_CASES: list[tuple[str, tuple[int, int], int]] = []
for _key in ALL_TILING_KEYS:
    _tiling = registry.get(_key)
    _n = _tiling.n_vertices * 4
    _n_b = _n // 2
    if comb(_n, _n_b) <= 5000:
        _SWEEP_CASES.append((_key, (2, 2), _n_b))
    else:
        _SWEEP_CASES.append((_key, (2, 2), 2))


class TestAllBackendsAgree:
    """Icet, orderly, and Burnside agree across all tilings at 2x2."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _SWEEP_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}_nb{nb}" for k, s, nb in _SWEEP_CASES],
    )
    def test_three_way_agreement(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """All three enumeration backends agree."""
        tiling = registry.get(key)
        n = tiling.n_vertices * supercell[0] * supercell[1]

        icet_count = len(
            IcetEnumerator().enumerate(tiling, n_b=n_b, supercell=supercell)
        )
        orderly_count = len(
            OrderlyEnumerator().enumerate(
                tiling, n_b=n_b, supercell=supercell
            )
        )
        group = IcetEnumerator().get_symmetry_group(
            tiling, supercell=supercell
        )
        burnside_count = BurnsideCounter.count(n, n_b, group)

        assert icet_count == orderly_count == burnside_count, (
            f"{key} {supercell} n_b={n_b}: "
            f"icet={icet_count}, orderly={orderly_count}, "
            f"burnside={burnside_count}"
        )
