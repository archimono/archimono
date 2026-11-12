"""Fast tests for the frontier-dp-value certify memory gate.

The gate estimates peak memory with the reachability-aware
``estimate_peak_memory_bytes(frontier_sizes, n_b)`` estimator and skips
(``status="budget"``) instead of solving when the estimate exceeds the limit.
"""

from __future__ import annotations

import pathlib

from archimono.tilings import registry
from archimono.workflows.certify import _frontier_dp_value_case


def test_zero_budget_skips_every_supercell(tmp_path: pathlib.Path) -> None:
    """With a 0 GB limit every supercell is a budget skip, none is solved."""
    records = _frontier_dp_value_case(
        tiling=registry.get("kagome"),
        tiling_key="kagome",
        tiling_display="kagome",
        target_n=12,
        out_dir=tmp_path,
        memory_limit_gb=0.0,
        stream=None,
    )
    assert records
    assert all(record.status == "budget" for record in records)
    assert all(record.certified_cut_value is None for record in records)


def test_default_budget_solves_small_supercells(tmp_path: pathlib.Path) -> None:
    """With the 60 GB default limit the n=12 kagome cases all certify."""
    records = _frontier_dp_value_case(
        tiling=registry.get("kagome"),
        tiling_key="kagome",
        tiling_display="kagome",
        target_n=12,
        out_dir=tmp_path,
        memory_limit_gb=60.0,
        stream=None,
    )
    assert records
    assert all(record.status == "certified" for record in records)
