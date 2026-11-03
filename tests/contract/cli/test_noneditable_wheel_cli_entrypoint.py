"""Contract coverage for the installed ``archimono`` console entry point."""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import venv

from archimono.tilings import registry

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]


def _venv_executable(venv_dir: pathlib.Path, name: str) -> pathlib.Path:
    """Return an executable path inside a virtual environment.

    Args:
        venv_dir: Root directory of the virtual environment.
        name: Executable base name without an extension.

    Returns:
        Path to the platform-specific virtual-environment executable.
    """
    if os.name == "nt":
        return venv_dir / "Scripts" / f"{name}.exe"
    return venv_dir / "bin" / name


def _subprocess_environment() -> dict[str, str]:
    """Return an environment that cannot import from this checkout.

    Returns:
        Environment for an independently installed subprocess.
    """
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    environment["PYTHONIOENCODING"] = "utf-8"
    return environment


def test_archimono_console_entrypoint_runs_from_noneditable_wheel_install(
    tmp_path: pathlib.Path,
) -> None:
    """An installed wheel provides a working console command outside the checkout."""
    wheel_dir = tmp_path / "wheel"
    build_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "wheel",
            "--no-deps",
            "--no-build-isolation",
            "--wheel-dir",
            str(wheel_dir),
            str(_PROJECT_ROOT),
        ],
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=_subprocess_environment(),
        timeout=120,
    )
    assert build_result.returncode == 0, build_result.stderr

    wheels = list(wheel_dir.glob("archimono-*.whl"))
    assert len(wheels) == 1

    venv_dir = tmp_path / "venv"
    venv.EnvBuilder(with_pip=True, system_site_packages=True).create(venv_dir)
    python = _venv_executable(venv_dir, "python")
    install_result = subprocess.run(
        [str(python), "-m", "pip", "install", "--no-deps", str(wheels[0])],
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=_subprocess_environment(),
        timeout=120,
    )
    assert install_result.returncode == 0, install_result.stderr

    result = subprocess.run(
        [str(_venv_executable(venv_dir, "archimono")), "list-tilings"],
        capture_output=True,
        encoding="utf-8",
        cwd=tmp_path,
        env=_subprocess_environment(),
        timeout=120,
    )

    rows = [tuple(line.split(maxsplit=1)) for line in result.stdout.splitlines()]
    expected = [
        (key, str(registry.get(key).n_vertices)) for key in registry.available()
    ]
    assert result.returncode == 0, result.stderr
    assert rows == expected
