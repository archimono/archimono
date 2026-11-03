"""End-to-end: a frontier-DP uint8 all_labels ndarray roundtrips through the sidecar.

``FrontierExactSolver(enumerate_all=True)`` now emits
``metadata['all_labels']`` as a node-indexed uint8 ndarray. This test confirms
the array packs into the ``.npz`` sidecar via ``write_case_json`` and is
recovered bit-identically by ``load_case_labels`` — exercising the
``write_case_json`` guard that tolerates a 2-D ndarray.
"""

from __future__ import annotations

import pathlib

import networkx as nx
import numpy as np

from archimono.assignment.solvers.frontier_dp import FrontierExactSolver
from archimono.certification import (
    CertificationRecord,
    load_case_json,
    load_case_labels,
    write_case_json,
)


def test_frontier_all_labels_ndarray_roundtrips_through_sidecar(
    tmp_path: pathlib.Path,
) -> None:
    """Solver uint8 all_labels -> write_case_json sidecar -> load_case_labels."""
    graph = nx.cycle_graph(6)  # even cycle: balanced MAX-CUT has several optima
    result = FrontierExactSolver().solve(
        graph, ["A", "B"], n_b=3, enumerate_all=True
    )
    all_labels = result.metadata["all_labels"]
    assert isinstance(all_labels, np.ndarray)  # produced by the frontier-DP path

    record = CertificationRecord(
        tiling_key="cycle6",
        tiling_display="C6 (test)",
        solver="frontier_dp",
        assignment_index=1,
        n_optimal_assignments=len(all_labels),
        vertex_config="AB",
        target_n=6,
        realized_n=6,
        supercell="(6x1)",
        n_edges=graph.number_of_edges(),
        certified_cut_value=result.cut_value,
        certified_cut_efficiency=None,
        runtime_ms=0.0,
        status="certified",
        metadata={"all_labels": all_labels},
    )

    write_case_json(out_dir=tmp_path, record=record, generated_by="test")
    assert list(tmp_path.glob("*-labels.npz")), "expected a packed-labels sidecar"

    loaded = load_case_json(next(tmp_path.glob("*.json")))
    recovered = load_case_labels(loaded, tmp_path)
    assert recovered == [tuple(int(x) for x in row) for row in all_labels]
