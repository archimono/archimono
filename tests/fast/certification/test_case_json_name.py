"""Regression test for certification.case_json_name (audit M5).

Before the fix, ``case_json_name`` appended ``-opt{assignment_index}`` whenever
``n_optimal_assignments > 1``, but the script writes exactly one representative
per case with ``assignment_index=1``, so a 62-optimum case was named
``...-opt1.json`` (misleading). The fix drops the suffix entirely; all optima
live in the ``-labels.npz`` sidecar.
"""

from __future__ import annotations

from archimono.certification import CertificationRecord, case_json_name


def _record(
    n_optimal: int, assignment_index: int, supercell: str
) -> CertificationRecord:
    return CertificationRecord(
        tiling_key="triangular",
        tiling_display="3⁶ (triangular)",
        solver="frontier_dp",
        assignment_index=assignment_index,
        n_optimal_assignments=n_optimal,
        vertex_config="3⁶",
        target_n=12,
        realized_n=12,
        supercell=supercell,
        n_edges=36,
        certified_cut_value=24.0,
        certified_cut_efficiency=24.0 / 36.0,
        runtime_ms=1.0,
        status="certified",
        note="",
        metadata={},
    )


def test_no_opt_suffix_even_with_many_optima() -> None:
    """A multi-optimum case gets no ``-opt`` suffix in its filename."""
    name = case_json_name(_record(n_optimal=62, assignment_index=1, supercell="(2x3)"))
    assert "-opt" not in name
    assert name.startswith("triangular-n12-")
    assert name.endswith(".json")


def test_single_optimum_filename_unchanged() -> None:
    """A single-optimum case keeps the plain ``{tiling}-n{n}-{slug}.json`` name."""
    name = case_json_name(_record(n_optimal=1, assignment_index=1, supercell="(1x6)"))
    assert "-opt" not in name
    assert name.endswith(".json")
