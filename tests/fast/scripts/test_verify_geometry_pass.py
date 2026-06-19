"""Regression test for the verify.py geometry bond-length check (audit H1).

Before the fix, ``verify.py`` built each tiling with the default
``bond_length=1.0`` while comparing edge lengths against the physical Å values
in ``_EXPECTED``, so every tiling failed the bond-length check and the script
exited 1. This pins that, built with the expected bond length, the checks pass.
"""

from __future__ import annotations

import importlib.util
import pathlib
import types

import pytest

from archimono.tilings import registry


def _load_verify() -> types.ModuleType:
    """Load scripts/verify.py without requiring scripts to be a package."""
    script = pathlib.Path(__file__).parents[3] / "scripts" / "verify.py"
    spec = importlib.util.spec_from_file_location("verify_script_pass", script)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("key", ["3.6.3.6", "6³"])
def test_geometry_checks_pass_with_expected_bond_length(key: str) -> None:
    """Built with ``_EXPECTED`` bond length, geometry checks report no errors."""
    verify = _load_verify()
    bond = verify._EXPECTED[key][3]
    tiling = registry.get(key, bond_length=bond)
    g_vis = tiling.graph(verify._VIS_SC[key])
    g_bp = tiling.graph(verify._BPCHECK_SC[key])
    errors, _coord = verify._run_geometry_checks(key, tiling, g_vis, g_bp)
    assert errors == [], f"{key}: unexpected geometry errors {errors}"
