"""Contract smoke tests for the certify CLI command.

Runs each --solver backend on hexagonal n=12 and verifies that at least
one JSON file is written and loadable via archimono.certification.load_case_json.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys

import pytest

from archimono.certification import load_case_json


@pytest.mark.contract
@pytest.mark.parametrize(
    "solver",
    ["bruteforce", "frontier-dp-value", "annealing"],
)
def test_certify_smoke(solver: str, tmp_path: pathlib.Path) -> None:
    """Run the certify command with each solver backend on a minimal case.

    Invokes ``python -m archimono.cli certify --solver {solver} --tilings hexagonal
    --target-sizes 12 --json-out-dir {tmp_path}`` and checks that at
    least one JSON file is written and can be loaded without error.
    """
    markdown_out = tmp_path / f"{solver}.md"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "archimono.cli",
            "certify",
            "--solver",
            solver,
            "--tilings",
            "hexagonal",
            "--target-sizes",
            "12",
            "--json-out-dir",
            str(tmp_path),
            "--markdown-out",
            str(markdown_out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, (
        f"certify.py --solver {solver} exited {result.returncode}\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    json_files = list(tmp_path.glob("*.json"))
    assert json_files, (
        f"No JSON files written to {tmp_path} for --solver {solver}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    assert markdown_out.exists(), (
        f"Markdown output not found at {markdown_out} for --solver {solver}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Each file must be loadable.
    for path in json_files:
        record = load_case_json(path)
        assert record.tiling_key == "hexagonal"
        assert record.target_n == 12
