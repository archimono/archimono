"""Regression coverage for the published console-script metadata."""

from __future__ import annotations

import pathlib

import pytest
import tomllib

pytestmark = pytest.mark.fast


def test_project_declares_only_the_archimono_console_script() -> None:
    root = pathlib.Path(__file__).resolve().parents[3]
    with (root / "pyproject.toml").open("rb") as stream:
        project = tomllib.load(stream)["project"]

    assert project["scripts"] == {"archimono": "archimono.cli.__main__:main"}
