"""Workflow API for solving one binary MAX-CUT assignment."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import networkx as nx
import numpy as np

from archimono.assignment import (
    SOLVERS_BY_NAME,
    AssignmentResult,
    FrontierExactSolver,
    compute_dp_cost,
    estimate_reconstruct_memory_bytes,
)
from archimono.tilings import parse_supercell, registry

__all__ = [
    "SolveConfig",
    "SolveResult",
    "format_json",
    "format_text",
    "run_solve",
    "to_payload",
]

_VALUE_ONLY_UNAVAILABLE = "n/a (value-only solver)"


@dataclass(frozen=True, slots=True)
class SolveConfig:
    """Configuration for one solve workflow run.

    Attributes:
        tiling: Tiling registry key.
        supercell: Diagonal supercell as ``AxB`` or ``A,B``.
        n_b: Optional count of the second species.
        solver: Solver backend name.
        species: Two species symbols; label ``1`` counts toward ``n_b``.
        pbc: Whether to build a periodic graph.
        emit_json: Whether the caller intends to render JSON output.
        memory_limit_gb: ``[frontier-dp only]`` Reject the solve if the
            DP-state and reconstruction memory is estimated to exceed this
            limit.  Does not cover the output array; see ``max_optima``.
        enumerate_all: ``[frontier-dp only]`` If ``True``, enumerate every
            optimal assignment and store them as ``all_labels`` in metadata.
            Default ``False`` returns one optimal witness.
        max_optima: ``[frontier-dp only]`` Maximum number of optimal
            assignments to materialize when ``enumerate_all`` is ``True``.
            Default 10,000.  Raises ``ValueError`` mid-enumeration if
            exceeded.
    """

    tiling: str
    supercell: str = "1x1"
    n_b: int | None = None
    solver: str = "bruteforce"
    species: Sequence[str] = ("A", "B")
    pbc: bool = True
    emit_json: bool = False
    memory_limit_gb: float = 32.0
    enumerate_all: bool = False
    max_optima: int = 10_000


@dataclass(frozen=True, slots=True)
class SolveResult:
    """Result from one solve workflow run.

    Attributes:
        tiling: Tiling registry key.
        supercell: Parsed diagonal supercell.
        pbc: Whether the solved graph was periodic.
        solver: Solver backend name.
        species: Two species symbols used for text rendering.
        labels: Raw integer labels, or ``None`` for value-only solvers.
        cut_value: MAX-CUT value.
        n_frustrated: Number of non-cut edges, if available.
        metadata: Solver metadata.
    """

    tiling: str
    supercell: tuple[int, int]
    pbc: bool
    solver: str
    species: tuple[str, str]
    labels: tuple[int, ...] | None
    cut_value: int | float
    n_frustrated: int | None
    metadata: Mapping[str, Any]


def _json_ready(value: object) -> object:
    """Convert nested metadata to JSON-serializable built-in containers."""
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    return str(value)


def _display_number(value: float | int) -> str:
    """Render an integral numeric value without a trailing decimal."""
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number)


def _json_number(value: float | int) -> int | float:
    """Return an int for integral values, otherwise a float."""
    number = float(value)
    if number.is_integer():
        return int(number)
    return number


def _cut_value(graph: nx.Graph[int], labels: Sequence[int]) -> int:
    """Count unit-weight cut edges for an assignment."""
    return sum(1 for u, v in graph.edges() if labels[int(u)] != labels[int(v)])


def _validate_species(raw_species: Sequence[str]) -> tuple[str, str]:
    """Validate and return the two species symbols."""
    species = tuple(str(item) for item in raw_species)
    if len(species) != 2:
        raise ValueError("--species must contain exactly two symbols.")
    return species[0], species[1]


def _result_from_assignment(
    *,
    config: SolveConfig,
    supercell: tuple[int, int],
    species: tuple[str, str],
    graph: nx.Graph[int],
    result: AssignmentResult,
) -> SolveResult:
    """Build the uniform result for assignment-producing solvers."""
    labels = tuple(int(label) for label in result.labels)
    cut_value = result.cut_value
    if cut_value is None:
        cut_value = float(_cut_value(graph, labels))
    n_frustrated = result.n_frustrated
    if n_frustrated is None:
        n_frustrated = graph.number_of_edges() - int(round(cut_value))

    return SolveResult(
        tiling=config.tiling,
        supercell=supercell,
        pbc=config.pbc,
        solver=config.solver,
        species=species,
        labels=labels,
        cut_value=_json_number(cut_value),
        n_frustrated=n_frustrated,
        metadata=result.metadata,
    )


def _check_frontier_dp_memory_guard(
    graph: nx.Graph[int], memory_limit_gb: float
) -> None:
    """Raise if frontier-DP reconstruction would exceed the memory limit.

    This guard covers DP-state and parent-map memory only.  The output
    array (``all_labels``) is bounded separately by ``max_optima`` on
    :class:`SolveConfig`; see :func:`estimate_reconstruct_memory_bytes`
    for details on what this estimate includes.

    Args:
        graph: The input graph.
        memory_limit_gb: Memory budget in gigabytes.

    Raises:
        ValueError: If the reconstruction memory estimate exceeds the limit.
    """
    n_nodes = graph.number_of_nodes()
    # "geometric" is exact for FrontierExactSolver.solve(), which always runs
    # _best_geometric_order to choose its vertex ordering.
    dp_cost = compute_dp_cost(graph, "geometric")
    estimate_bytes = estimate_reconstruct_memory_bytes(dp_cost, n_nodes)
    limit_bytes = memory_limit_gb * 1e9
    if estimate_bytes > limit_bytes:
        raise ValueError(
            f"Frontier-DP reconstruction over dp_cost={dp_cost:,.0f}, "
            f"n={n_nodes} estimates {estimate_bytes / 1e9:,.0f} GB, exceeding "
            f"the {memory_limit_gb:,.0f} GB feasibility limit. Use "
            f"`certify --solver frontier-dp-value` for a memory-cheap "
            f"value-only alternative."
        )


def run_solve(config: SolveConfig) -> SolveResult:
    """Solve one tiling assignment.

    Raises:
        KeyError: If the tiling or solver name is unknown.
        ValueError: If the supercell, species, or solver inputs are invalid,
            or if the ``frontier-dp`` reconstruction's estimated peak memory
            exceeds ``config.memory_limit_gb``.
    """
    supercell = parse_supercell(config.supercell)
    species = _validate_species(config.species)
    tiling = registry.get(config.tiling)
    graph = tiling.graph(supercell, pbc=config.pbc)

    if config.solver == "frontier-dp":
        _check_frontier_dp_memory_guard(graph, config.memory_limit_gb)
        result = FrontierExactSolver().solve(
            graph,
            species,
            n_b=config.n_b,
            enumerate_all=config.enumerate_all,
            max_optima=config.max_optima if config.enumerate_all else None,
        )
    else:
        solver = SOLVERS_BY_NAME[config.solver]()
        result = solver.solve(graph, species, n_b=config.n_b)
    return _result_from_assignment(
        config=config,
        supercell=supercell,
        species=species,
        graph=graph,
        result=result,
    )


def to_payload(result: SolveResult) -> dict[str, object]:
    """Convert a solve result to the CLI-compatible payload.

    Returns:
        JSON-compatible payload except for nested metadata values, which are
        normalized by :func:`format_json`.
    """
    return {
        "tiling": result.tiling,
        "supercell": [result.supercell[0], result.supercell[1]],
        "pbc": result.pbc,
        "solver": result.solver,
        "labels": None if result.labels is None else list(result.labels),
        "cut_value": result.cut_value,
        "n_frustrated": result.n_frustrated,
        "metadata": result.metadata,
    }


def format_text(result: SolveResult) -> str:
    """Format a solve result as plain text, no trailing newline."""
    payload = to_payload(result)
    lines = [
        f"tiling: {payload['tiling']}",
        f"supercell: {result.supercell[0]}x{result.supercell[1]}",
        f"pbc: {str(payload['pbc']).lower()}",
        f"solver: {payload['solver']}",
    ]
    if result.labels is None:
        lines.append(f"labels: {_VALUE_ONLY_UNAVAILABLE}")
    else:
        lines.append(
            "labels: " + " ".join(result.species[label] for label in result.labels)
        )
    lines.append(f"cut_value: {_display_number(result.cut_value)}")
    if result.n_frustrated is None:
        lines.append(f"n_frustrated: {_VALUE_ONLY_UNAVAILABLE}")
    else:
        lines.append(f"n_frustrated: {result.n_frustrated}")
    metadata = _json_ready(payload["metadata"])
    lines.append(f"metadata: {json.dumps(metadata, sort_keys=True)}")
    return "\n".join(lines)


def format_json(result: SolveResult) -> str:
    """Format a solve result as JSON, no trailing newline."""
    return json.dumps(_json_ready(to_payload(result)), sort_keys=True)
