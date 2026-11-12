"""Fast output tests for the typed enumeration workflow."""

from __future__ import annotations

import importlib.util
import json
import pathlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

pytestmark = pytest.mark.fast


class _Graph:
    """Minimal graph value used by the workflow's cut calculations."""

    def __init__(self, edges: tuple[tuple[int, int], ...]) -> None:
        self._edges = edges

    def edges(self) -> tuple[tuple[int, int], ...]:
        """Return the fixed unit-weight edges."""
        return self._edges


class _GraphTiling:
    """Deterministic graph provider used to isolate enumeration output."""

    def __init__(self, graph: _Graph) -> None:
        self._graph = graph

    def graph(self, supercell: tuple[int, int]) -> _Graph:
        del supercell
        return self._graph


def _load_workflow(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Load enumeration source with only its lightweight dependency seams."""
    assignment = ModuleType("archimono.assignment")
    assignment.IcetEnumerator = object
    assignment.OrderlyEnumerator = object
    tilings = ModuleType("archimono.tilings")
    tilings.registry = SimpleNamespace(get=lambda key: None)
    tilings.parse_supercell = lambda text: tuple(
        int(part) for part in text.replace("x", ",").split(",")
    )
    tilings_base = ModuleType("archimono.tilings.base")
    tilings_base.Tiling = object
    networkx = ModuleType("networkx")
    networkx.Graph = _Graph
    for name, module in {
        "archimono.assignment": assignment,
        "archimono.tilings": tilings,
        "archimono.tilings.base": tilings_base,
        "networkx": networkx,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    source = (
        pathlib.Path(__file__).resolve().parents[3]
        / "src"
        / "archimono"
        / "workflows"
        / "enumerate.py"
    )
    module_name = "_fast_cli_enumeration_workflow"
    spec = importlib.util.spec_from_file_location(module_name, source)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _stub_enumeration(monkeypatch: pytest.MonkeyPatch) -> tuple[ModuleType, object]:
    """Return a controlled pair of raw label sequences through the workflow."""
    workflow = _load_workflow(monkeypatch)
    graph = _Graph(((0, 1), (1, 2), (2, 3), (3, 0)))
    monkeypatch.setattr(workflow.registry, "get", lambda key: _GraphTiling(graph))
    enumerator = SimpleNamespace(
        enumerate=lambda tiling, n_b, supercell, min_cut: [
            (0, 1, 1, 0),
            (1, 0, 1, 0),
        ]
    )
    monkeypatch.setattr(
        workflow.enumeration,
        "get",
        lambda backend: enumerator,
    )
    return workflow, workflow.enumerate_structures(
        workflow.EnumerationConfig(tiling="fixed", n_b=2)
    )


def test_enumeration_result_recomputes_each_structure_cut_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every enumerated row derives its cut value from the fixed graph edges."""
    _, result = _stub_enumeration(monkeypatch)

    assert [structure.cut_value for structure in result.structures] == [2, 4]


def test_enumeration_text_prints_count_before_structure_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workflow, result = _stub_enumeration(monkeypatch)
    lines = workflow.format_text(result).splitlines()

    assert lines[0] == "count: 2"
    assert len(lines) == 3
    assert all(line.startswith("labels:") for line in lines[1:])


def test_enumeration_json_serializes_raw_integer_labels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Enumeration JSON preserves raw labels rather than display strings."""
    workflow, result = _stub_enumeration(monkeypatch)
    payload = json.loads(workflow.format_json(result))

    labels = payload["structures"][0]["labels"]
    assert labels == [0, 1, 1, 0]
    assert all(isinstance(label, int) for label in labels)


def test_enumeration_json_count_equals_the_number_of_structures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """JSON count covers every serialized row without truncation."""
    workflow, result = _stub_enumeration(monkeypatch)
    payload = json.loads(workflow.format_json(result))

    assert payload["count"] == len(payload["structures"]) == 2
