"""Fast tests for scripts/verify.py geometry-check error handling (#55)."""

from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest

from archimono.tilings import registry

_KEY = "3.6.3.6"  # kagome


def _load_verify_module() -> types.ModuleType:
    """Load scripts/verify.py without requiring scripts to be a package."""
    script = pathlib.Path(__file__).parents[3] / "scripts" / "verify.py"
    spec = importlib.util.spec_from_file_location("verify_script_geom", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_geometry_checks_return_coordination() -> None:
    """The checker returns the computed coordination number with no coord error."""
    verify = _load_verify_module()
    tiling = registry.get(_KEY)
    g_vis = tiling.graph(verify._VIS_SC[_KEY])
    g_bp = tiling.graph(verify._BPCHECK_SC[_KEY])

    errors, coord = verify._run_geometry_checks(_KEY, tiling, g_vis, g_bp)

    assert coord == 4  # kagome coordination
    assert not any("coordination" in e for e in errors)


def test_geometry_checks_handle_coordination_valueerror(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A coordination ValueError is recorded (coord=None), not re-raised (#55).

    The status print in ``run_geometry`` previously re-read
    ``tiling.coordination`` unguarded, so a non-vertex-transitive tiling
    aborted the whole loop instead of reporting the error. The checker now
    returns the cached coordination value (``None`` on failure).
    """
    verify = _load_verify_module()
    tiling = registry.get(_KEY)
    g_vis = tiling.graph(verify._VIS_SC[_KEY])
    g_bp = tiling.graph(verify._BPCHECK_SC[_KEY])

    def _raise(_self: object) -> int:
        raise ValueError("not vertex-transitive")

    monkeypatch.setattr(type(tiling), "coordination", property(_raise))

    errors, coord = verify._run_geometry_checks(_KEY, tiling, g_vis, g_bp)

    assert coord is None
    assert any("coordination check failed" in e for e in errors)
