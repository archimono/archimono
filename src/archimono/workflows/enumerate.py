"""Workflow API for enumerating inequivalent binary structures."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

import networkx as nx

from archimono.assignment import enumeration
from archimono.tilings import parse_supercell, registry

__all__ = [
    "EnumeratedStructure",
    "EnumerationConfig",
    "EnumerationResult",
    "enumerate_structures",
    "format_json",
    "format_text",
    "to_payload",
]


@dataclass(frozen=True, slots=True)
class EnumerationConfig:
    """Configuration for one enumeration workflow run.

    Attributes:
        tiling: Tiling registry key.
        n_b: Required count of the second species.
        supercell: Diagonal supercell as ``AxB`` or ``A,B``.
        min_cut: Minimum cut value to include.
        backend: Enumeration backend name.
        emit_json: Whether the caller intends to render JSON output.
    """

    tiling: str
    n_b: int
    supercell: str = "1x1"
    min_cut: int = 0
    backend: str = "orderly"
    emit_json: bool = False


@dataclass(frozen=True, slots=True)
class EnumeratedStructure:
    """One enumerated binary structure.

    Attributes:
        labels: Raw integer labels for graph nodes.
        cut_value: Unit-weight cut value for the labels.
    """

    labels: tuple[int, ...]
    cut_value: int


@dataclass(frozen=True, slots=True)
class EnumerationResult:
    """Result from one enumeration workflow run.

    Attributes:
        tiling: Tiling registry key.
        supercell: Parsed diagonal supercell.
        n_b: Required count of the second species.
        min_cut: Minimum cut value used during enumeration.
        backend: Enumeration backend name.
        structures: Enumerated structures with cut values.
    """

    tiling: str
    supercell: tuple[int, int]
    n_b: int
    min_cut: int
    backend: str
    structures: tuple[EnumeratedStructure, ...]


def _cut_value(graph: nx.Graph[int], labels: Sequence[int]) -> int:
    """Count unit-weight cut edges for an assignment."""
    return sum(1 for u, v in graph.edges() if labels[int(u)] != labels[int(v)])


def _structure_rows(
    graph: nx.Graph[int],
    structures: Sequence[tuple[int, ...]],
) -> tuple[EnumeratedStructure, ...]:
    """Build labels and cut-value rows for enumerated structures."""
    return tuple(
        EnumeratedStructure(labels=labels, cut_value=_cut_value(graph, labels))
        for labels in structures
    )


def enumerate_structures(config: EnumerationConfig) -> EnumerationResult:
    """Enumerate inequivalent binary structures.

    Raises:
        KeyError: If the tiling or backend name is unknown.
        ValueError: If the supercell or enumerator inputs are invalid.
    """
    supercell = parse_supercell(config.supercell)
    tiling = registry.get(config.tiling)
    structures = enumeration.get(config.backend).enumerate(
        tiling,
        n_b=config.n_b,
        supercell=supercell,
        min_cut=config.min_cut,
    )
    graph = tiling.graph(supercell)
    return EnumerationResult(
        tiling=config.tiling,
        supercell=supercell,
        n_b=config.n_b,
        min_cut=config.min_cut,
        backend=config.backend,
        structures=_structure_rows(graph, structures),
    )


def to_payload(result: EnumerationResult) -> dict[str, object]:
    """Convert an enumeration result to the CLI-compatible JSON payload."""
    structures: list[dict[str, object]] = [
        {"labels": list(structure.labels), "cut_value": structure.cut_value}
        for structure in result.structures
    ]
    return {
        "tiling": result.tiling,
        "supercell": [result.supercell[0], result.supercell[1]],
        "n_b": result.n_b,
        "min_cut": result.min_cut,
        "backend": result.backend,
        "count": len(result.structures),
        "structures": structures,
    }


def format_text(result: EnumerationResult) -> str:
    """Format an enumeration result as plain text, no trailing newline."""
    lines = [f"count: {len(result.structures)}"]
    for structure in result.structures:
        labels = " ".join(str(label) for label in structure.labels)
        lines.append(f"labels: {labels} cut_value: {structure.cut_value}")
    return "\n".join(lines)


def format_json(result: EnumerationResult) -> str:
    """Format an enumeration result as JSON, no trailing newline."""
    return json.dumps(to_payload(result), sort_keys=True)
