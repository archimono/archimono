"""Contract tests for enumeration ground truth and backend agreement.

Ground truth values verified via brute-force + multi-backend agreement.
"""

from __future__ import annotations

import pytest

from archimono.assignment import IcetEnumerator
from archimono.assignment.enumeration.burnside import BurnsideCounter
from archimono.assignment.enumeration.orderly import OrderlyEnumerator
from archimono.tilings import registry

# Cases where all three backends are tested for agreement.
# (tiling_key, supercell, n_b)
_AGREEMENT_CASES = [
    ("triangular", (2, 2), 2),
    ("kagome", (2, 2), 6),
    ("kagome", (1, 1), 1),
    ("hexagonal", (2, 2), 4),
]


class TestBackendAgreement:
    """All enumeration backends produce the same count."""

    @pytest.mark.parametrize(
        "key,supercell,n_b",
        _AGREEMENT_CASES,
        ids=[f"{k}_{s[0]}x{s[1]}_nb{nb}" for k, s, nb in _AGREEMENT_CASES],
    )
    def test_icet_orderly_burnside_agree(
        self, key: str, supercell: tuple[int, int], n_b: int
    ) -> None:
        """Three backends produce identical counts."""
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

        assert icet_count == orderly_count, (
            f"{key} {supercell} n_b={n_b}: "
            f"icet={icet_count} != orderly={orderly_count}"
        )
        assert icet_count == burnside_count, (
            f"{key} {supercell} n_b={n_b}: "
            f"icet={icet_count} != burnside={burnside_count}"
        )


class TestKagome2x2GroundTruth:
    """Kagome 2x2 reference values (verified via brute-force)."""

    def test_inequivalent_config_count(self) -> None:
        """Total symmetry-inequivalent balanced configs = 30.

        Verified via brute-force, icet, orderly, and Burnside agreement.
        """
        tiling = registry.get("kagome")
        icet_configs = IcetEnumerator().enumerate(
            tiling, n_b=6, supercell=(2, 2)
        )
        orderly_configs = OrderlyEnumerator().enumerate(
            tiling, n_b=6, supercell=(2, 2)
        )
        assert len(icet_configs) == len(orderly_configs)
        assert len(icet_configs) == 30

    def test_near_optimal_window(self) -> None:
        """Configs within 2 bonds of optimum (min_cut >= 14) = 15."""
        tiling = registry.get("kagome")
        near_opt = IcetEnumerator().enumerate(
            tiling, n_b=6, supercell=(2, 2), min_cut=14
        )
        assert len(near_opt) == 15
