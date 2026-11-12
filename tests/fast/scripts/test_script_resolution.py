"""Fast tests for cwd-independent CLI command resolution."""

from __future__ import annotations

import argparse
import importlib
import pathlib
import re
import subprocess

import pytest

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_HISTORICAL_SPECS = "docs/sdd/specs/"

_LEGACY_USER_FACING_SCRIPT_NAMES = {
    "verify_tilings.py",
    "visualize_kagome.py",
    "stress_test_tilings.py",
}
_REMOVED_WORKFLOW_SCRIPT_NAMES = {
    "verify_tilings.py",
    "verify_exact_solver.py",
    "visualize_kagome.py",
}
_PYTHON_SCRIPT_INVOCATION = re.compile(
    r"\bpython(?:\d+(?:\.\d+)?)?(?:\s+-[A-Za-z]+)*\s+scripts/[^\s`]+\.py"
)


def _tracked_paths(pathspec: str) -> set[str]:
    """Return repository-tracked paths under one Git pathspec.

    Args:
        pathspec: Repository-relative pathspec to list.

    Returns:
        Set of repository-relative tracked paths.
    """
    result = subprocess.run(
        ["git", "ls-files", "--", pathspec],
        capture_output=True,
        check=True,
        cwd=_PROJECT_ROOT,
        encoding="utf-8",
        text=True,
    )
    return set(result.stdout.splitlines())


def _tracked_nonhistorical_docs() -> list[pathlib.Path]:
    """Return tracked Markdown docs outside historical SDD specifications."""
    return [
        _PROJECT_ROOT / relative_path
        for relative_path in _tracked_paths("docs")
        if relative_path.endswith(".md")
        and not relative_path.startswith(_HISTORICAL_SPECS)
    ]


def _tracked_source_modules() -> list[pathlib.Path]:
    """Return tracked Python source modules, excluding docs and test fixtures."""
    return [
        _PROJECT_ROOT / relative_path
        for relative_path in _tracked_paths("src/archimono")
        if relative_path.endswith(".py")
    ]


def _project_script_entries() -> dict[str, str]:
    """Read ``[project.scripts]`` entries from ``pyproject.toml``.

    Returns:
        Mapping of console script name to import target.
    """
    entries: dict[str, str] = {}
    in_section = False
    for raw_line in (_PROJECT_ROOT / "pyproject.toml").read_text().splitlines():
        line = raw_line.strip()
        if line == "[project.scripts]":
            in_section = True
            continue
        if in_section and line.startswith("["):
            break
        if in_section and line and not line.startswith("#"):
            name, target = line.split("=", maxsplit=1)
            entries[name.strip()] = target.strip().strip('"')
    return entries


def _parser_for(command_module: object) -> argparse.ArgumentParser:
    """Build a command parser through the package CLI seam.

    Args:
        command_module: Imported ``archimono.cli`` command module.

    Returns:
        Configured argument parser.
    """
    parser = argparse.ArgumentParser()
    configure_parser = getattr(command_module, "configure_parser")
    configure_parser(parser)
    return parser


def test_top_level_cli_registers_all_subcommands() -> None:
    cli_main = importlib.import_module("archimono.cli.__main__")
    parser = cli_main._build_parser()

    subparsers = [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]

    assert len(subparsers) == 1
    assert set(subparsers[0].choices) == {
        "verify",
        "certify",
        "benchmark",
        "list-tilings",
        "solve",
        "enumerate",
    }


def test_top_level_cli_normalizes_none_return_to_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_main = importlib.import_module("archimono.cli.__main__")
    benchmark = importlib.import_module("archimono.cli.benchmark")

    monkeypatch.setattr(benchmark, "run", lambda _args: None)

    assert cli_main.main(["benchmark", "numba-jit"]) == 0


def test_top_level_cli_preserves_integer_exit_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cli_main = importlib.import_module("archimono.cli.__main__")
    verify = importlib.import_module("archimono.cli.verify")

    monkeypatch.setattr(verify, "run", lambda _args: 7)

    assert cli_main.main(["verify"]) == 7


def test_verify_default_out_dir_is_project_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    verify = importlib.import_module("archimono.cli.verify")
    parser = _parser_for(verify)
    monkeypatch.chdir(tmp_path)

    args = parser.parse_args([])

    assert args.out_dir == _PROJECT_ROOT / "tmp" / "verify"


def test_certify_default_artifact_paths_are_project_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    certify = importlib.import_module("archimono.cli.certify")
    parser = _parser_for(certify)
    monkeypatch.chdir(tmp_path)

    args = parser.parse_args(["--solver", "annealing"])
    json_out_dir = args.json_out_dir or certify._default_json_out_dir(args.solver)
    markdown_out = args.markdown_out or certify._default_markdown_out(args.solver)

    assert json_out_dir == _PROJECT_ROOT / "tmp" / "annealing-stress"
    assert markdown_out == _PROJECT_ROOT / "tmp" / "stress-test-results.md"
    assert args.bruteforce_md == _PROJECT_ROOT / "docs" / "draft" / (
        "bruteforce-certification.md"
    )


def test_benchmark_default_artifact_paths_are_project_relative(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: pathlib.Path,
) -> None:
    benchmark = importlib.import_module("archimono.cli.benchmark")
    parser = _parser_for(benchmark)
    monkeypatch.chdir(tmp_path)

    frontier_order = parser.parse_args(["frontier-order"])
    report = parser.parse_args(["sample-runs", "--db", "runs.db", "--report"])

    assert frontier_order.md_out == _PROJECT_ROOT / "docs" / "draft" / (
        "frontier-order-comparison.md"
    )
    assert report.report_out == _PROJECT_ROOT / "docs" / "draft" / (
        "solver-characterization.md"
    )


def test_pyproject_exposes_console_script_entry_points() -> None:
    """Installed environments expose only the unified console command."""
    scripts = _project_script_entries()
    expected = {"archimono": "archimono.cli.__main__:main"}

    assert scripts == expected
    for target in scripts.values():
        module_name, function_name = target.split(":", maxsplit=1)
        module = importlib.import_module(module_name)
        assert callable(getattr(module, function_name))



def test_nonhistorical_docs_do_not_refer_to_legacy_python_scripts() -> None:
    """User-facing docs point to package CLI workflows, not removed scripts."""
    violations: dict[str, list[str]] = {}
    for path in _tracked_nonhistorical_docs():
        if path.name == "audit-findings.md":
            continue
        text = path.read_text(encoding="utf-8")
        matches = [
            m for m in _PYTHON_SCRIPT_INVOCATION.findall(text)
            if "export_structures.py" not in m and "sliced_certify.py" not in m
        ]
        matches.extend(
            name for name in _LEGACY_USER_FACING_SCRIPT_NAMES if name in text
        )
        if matches:
            violations[path.relative_to(_PROJECT_ROOT).as_posix()] = matches

    assert violations == {}


def test_source_docs_and_examples_do_not_refer_to_removed_workflow_scripts() -> None:
    """Source docs and examples use supported package CLI commands."""
    violations: dict[str, list[str]] = {}
    for path in _tracked_source_modules():
        text = path.read_text(encoding="utf-8")
        matches = [
            name for name in _REMOVED_WORKFLOW_SCRIPT_NAMES if name in text
        ]
        if matches:
            violations[path.relative_to(_PROJECT_ROOT).as_posix()] = matches

    assert violations == {}


def test_benchmark_numba_import_error_returns_one_and_writes_stderr(
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    benchmark = importlib.import_module("archimono.cli.benchmark")
    benchmark_workflows = importlib.import_module("archimono.workflows.benchmark")

    def raise_missing_numba(_config: object) -> None:
        raise ImportError("numba is required for this benchmark")

    monkeypatch.setattr(benchmark_workflows, "run_numba_jit", raise_missing_numba)
    args = _parser_for(benchmark).parse_args(["numba-jit"])

    assert benchmark.run(args) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "numba is required for this benchmark\n"


def test_script_loader_module_is_removed() -> None:
    """The package CLI no longer dispatches through repository script loading."""
    loader_path = _PROJECT_ROOT / "src" / "archimono" / "cli" / "_script_loader.py"
    assert not loader_path.exists()


def test_bruteforce_slurm_wrapper_uses_standalone_script() -> None:
    """The brute-force wrapper invokes the scripts/sliced_certify.py script."""
    wrapper = (
        _PROJECT_ROOT / "scripts" / "slurm" / "certify_bruteforce_array.sh"
    ).read_text(encoding="utf-8")

    assert wrapper.count("python scripts/sliced_certify.py") == 4
    assert "sliced-certify" not in wrapper
