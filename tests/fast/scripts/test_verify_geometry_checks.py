"""Fast tests for scripts/verify.py geometry-check error handling (#55)."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from archimono import verification
from archimono.tilings import registry

_KEY = "3.6.3.6"  # kagome


def test_geometry_workflow_reports_coordination_valueerror(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Geometry mode reports coordination errors without aborting its renders."""
    expected = verification._EXPECTED[_KEY]
    monkeypatch.setattr(verification, "_EXPECTED", {_KEY: expected})
    tiling = registry.get(_KEY)

    def raise_value_error(_self: object) -> int:
        raise ValueError("not vertex-transitive")

    monkeypatch.setattr(
        type(tiling),
        "coordination",
        property(raise_value_error),
    )
    output = io.StringIO()
    result = verification.run_verification(
        verification.VerificationConfig(
            mode="geometry",
            out_dir=tmp_path,
            output=output,
        )
    )

    assert result.ok is False
    assert result.geometry_ok is False
    assert "coord=?" in output.getvalue()
    assert "coordination check failed: not vertex-transitive" in output.getvalue()
    assert (tmp_path / "overview_geometry.png").is_file()
