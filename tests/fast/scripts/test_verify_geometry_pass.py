"""Regression test for the package geometry workflow edge-length check."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from archimono import verification
from archimono.tilings import registry
from archimono.tilings.base import Tiling

_KEY = "3.6.3.6"  # kagome


def test_geometry_workflow_uses_expected_edge_length(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Geometry mode passes the physical edge length to every registry lookup."""
    expected = verification._EXPECTED[_KEY]
    expected_edge_length = expected[3]
    monkeypatch.setattr(verification, "_EXPECTED", {_KEY: expected})

    requests: list[tuple[str, float]] = []
    real_get = registry.get

    def record_get(vertex_config: str, edge_length: float = 1.0) -> Tiling:
        requests.append((vertex_config, edge_length))
        return real_get(vertex_config, edge_length)

    monkeypatch.setattr(registry, "get", record_get)
    output = io.StringIO()
    result = verification.run_verification(
        verification.VerificationConfig(
            mode="geometry",
            out_dir=tmp_path,
            output=output,
        )
    )

    assert requests == [(_KEY, expected_edge_length)] * 2
    assert result.ok is True
    assert result.geometry_ok is True
    assert (tmp_path / "overview_geometry.png").is_file()
