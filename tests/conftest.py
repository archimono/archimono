"""Shared test infrastructure for the archimono test suite.

Provides:
- Parametrized tiling fixture (all_tilings)
- Small graph fixtures for fast-tier smoke tests
"""

from __future__ import annotations

import pathlib
import sys

import networkx as nx
import pytest

from archimono.tilings import registry

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


ALL_TILING_KEYS = sorted(registry.available())


@pytest.fixture(params=ALL_TILING_KEYS)
def all_tilings(request: pytest.FixtureRequest) -> str:
    """Parametrize over all registered tiling keys (including aliases)."""
    return request.param


@pytest.fixture()
def c4_graph() -> nx.Graph[int]:
    """A 4-node cycle graph (bipartite, planar, trivial)."""
    return nx.cycle_graph(4)


@pytest.fixture()
def petersen_graph() -> nx.Graph[int]:
    """Petersen graph (non-planar, non-bipartite)."""
    return nx.petersen_graph()
