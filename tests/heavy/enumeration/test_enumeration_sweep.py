"""Heavy enumeration tests: all backends x all tilings at their supported size.

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

# triangular and elongated_triangular alias at (2,2) (edge instances collapse
# under PBC -- see Tiling._graph_matrix); their smallest supported supercells
# are (3,3) and (3,1) respectively. Every other tiling is unaffected at
# (2,2). Keyed by vertex_config since ALL_TILING_KEYS includes every
# registered alias (including vertex-config strings) for the same tiling.
_SUPERCELL_BY_VERTEX_CONFIG: dict[str, tuple[int, int]] = {
    "3⁶": (3, 3),  # triangular
    "3³.4²": (3, 1),  # elongated_triangular
}

# Build sweep cases: all tilings at their supported size, choosing n_b to
# keep tractable.
_SWEEP_CASES: list[tuple[str, tuple[int, int], int]] = []
for _key in ALL_TILING_KEYS:
    _tiling = registry.get(_key)
    _supercell = _SUPERCELL_BY_VERTEX_CONFIG.get(_tiling.vertex_config, (2, 2))
    _n = _tiling.n_vertices * _supercell[0] * _supercell[1]
    _n_b = _n // 2
    if comb(_n, _n_b) <= 5000:
        _SWEEP_CASES.append((_key, _supercell, _n_b))
    else:
        _SWEEP_CASES.append((_key, _supercell, 2))


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
