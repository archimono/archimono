"""Contract tests for certify.py provenance fields (audit M1, M2).

M1: heuristic annealing on a frustrated tiling must be ``status="heuristic"``,
not ``"certified"`` (only the exact 2-colouring on bipartite tilings is
certified). M2: the frontier-dp backend must report the real number of optimal
assignments (with ``optima_enumerated``), not a constant 1, and must agree with
the bruteforce backend.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

from archimono.tilings import registry


def _load_certify() -> types.ModuleType:
    """Load scripts/certify.py without requiring scripts to be a package."""
    script = pathlib.Path(__file__).parents[3] / "scripts" / "certify.py"
    spec = importlib.util.spec_from_file_location("certify_script_prov", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_annealing_frustrated_is_heuristic(tmp_path: pathlib.Path) -> None:
    """A frustrated tiling solved by annealing is labelled heuristic, not certified."""
    certify = _load_certify()
    tiling = registry.get("triangular")
    record = certify._annealing_case(
        tiling=tiling,
        tiling_key="triangular",
        tiling_display="3⁶ (triangular)",
        target_n=12,
        out_dir=tmp_path,
    )
    assert record.solver == "annealing"
    assert record.status == "heuristic"
    assert record.metadata["is_exact"] is False


def test_annealing_bipartite_is_certified(tmp_path: pathlib.Path) -> None:
    """A bipartite tiling solved by exact 2-colouring stays certified."""
    certify = _load_certify()
    tiling = registry.get("hexagonal")
    record = certify._annealing_case(
        tiling=tiling,
        tiling_key="hexagonal",
        tiling_display="6³ (hexagonal)",
        target_n=12,
        out_dir=tmp_path,
    )
    assert record.solver == "bipartite_exact"
    assert record.status == "certified"


def test_frontier_dp_reports_real_optima_count(tmp_path: pathlib.Path) -> None:
    """Frontier-dp reports the true optima count and agrees with bruteforce."""
    certify = _load_certify()
    tiling = registry.get("triangular")
    fdp = certify._frontier_dp_case(
        tiling=tiling,
        tiling_key="triangular",
        tiling_display="3⁶ (triangular)",
        target_n=12,
        out_dir=tmp_path / "fdp",
    )
    bf = certify._bruteforce_case(
        tiling=tiling,
        tiling_key="triangular",
        tiling_display="3⁶ (triangular)",
        target_n=12,
        out_dir=tmp_path / "bf",
        one_supercell=False,
    )
    # Small case (n=12 <= guard): optima are enumerated, count is real (>1 here).
    assert all(r.metadata.get("optima_enumerated") for r in fdp)
    assert any(r.n_optimal_assignments > 1 for r in fdp)

    # The two exact backends agree on optima counts per supercell.
    fdp_by_sc = {r.supercell: r.n_optimal_assignments for r in fdp}
    bf_by_sc = {r.supercell: r.n_optimal_assignments for r in bf}
    shared = set(fdp_by_sc) & set(bf_by_sc)
    assert shared, "expected overlapping supercells between backends"
    for sc in shared:
        assert fdp_by_sc[sc] == bf_by_sc[sc], sc
