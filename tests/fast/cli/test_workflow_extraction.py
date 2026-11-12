"""Tests for CLI adapter seams and extracted workflow APIs."""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.util
import inspect
import pathlib
from dataclasses import is_dataclass
from types import ModuleType

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
_CLI_DIR = _REPO_ROOT / "src" / "archimono" / "cli"
_OWNED_CLI_MODULES = (
    "solve",
    "enumerate",
    "list_tilings",
    "verify",
    "certify",
    "benchmark",
)
_BANNED_TOP_LEVEL_IMPORTS = {
    "matplotlib",
    "networkx",
    "numpy",
    "sqlite3",
    "archimono.assignment",
    "archimono.tilings.registry",
    "archimono.verification",
    "archimono.certification",
    "archimono.workflows.certify",
    "archimono.workflows.benchmark",
    "archimono.workflows.solve",
    "archimono.workflows.enumerate",
}


def _import_required(module_name: str) -> ModuleType:
    """Import a module that must exist for the workflow extraction contract."""
    spec = importlib.util.find_spec(module_name)
    assert spec is not None, f"{module_name} is missing"
    return importlib.import_module(module_name)


def _top_level_imports(path: pathlib.Path) -> list[str]:
    """Return import targets from direct module-level import statements."""
    imports: list[str] = []
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for statement in tree.body:
        if (
            isinstance(statement, ast.If)
            and isinstance(statement.test, ast.Name)
            and statement.test.id == "TYPE_CHECKING"
        ):
            continue
        if isinstance(statement, ast.Import):
            imports.extend(alias.name for alias in statement.names)
        elif isinstance(statement, ast.ImportFrom):
            module = statement.module or ""
            imports.append(module)
            imports.extend(f"{module}.{alias.name}" for alias in statement.names)
    return imports


def _is_banned_import(import_name: str) -> bool:
    """Return whether an import violates the lightweight CLI boundary."""
    return any(
        import_name == banned or import_name.startswith(f"{banned}.")
        for banned in _BANNED_TOP_LEVEL_IMPORTS
    )


def test_owned_cli_modules_expose_only_command_seam() -> None:
    """Owned command modules expose only ``configure_parser`` and ``run``."""
    for command_name in _OWNED_CLI_MODULES:
        module = importlib.import_module(f"archimono.cli.{command_name}")

        assert getattr(module, "__all__", None) == ["configure_parser", "run"]
        public_names = {
            name
            for name in vars(module)
            if not name.startswith("_") and name != "annotations"
        }
        assert public_names == {"configure_parser", "run"}


def test_corrected_cli_modules_configure_parsers() -> None:
    for command_name in ("verify", "certify", "benchmark"):
        module = importlib.import_module(f"archimono.cli.{command_name}")
        parser = argparse.ArgumentParser()
        subparser = parser.add_subparsers().add_parser(command_name)

        module.configure_parser(subparser)


def test_owned_cli_modules_have_no_heavy_top_level_imports() -> None:
    """Every CLI source module lazy-loads heavy workflow dependencies."""
    violations: dict[str, list[str]] = {}
    for path in sorted(_CLI_DIR.glob("*.py")):
        banned_imports = [
            import_name
            for import_name in _top_level_imports(path)
            if _is_banned_import(import_name)
        ]
        if banned_imports:
            violations[path.as_posix()] = banned_imports

    assert violations == {}


def test_solve_workflow_exposes_typed_api_and_frontier_dp_assignment_path() -> None:
    """The solve workflow is callable without an ``argparse.Namespace``."""
    module = _import_required("archimono.workflows.solve")

    assert is_dataclass(module.SolveConfig)
    assert is_dataclass(module.SolveResult)
    signature = inspect.signature(module.run_solve)
    assert list(signature.parameters) == ["config"]
    assert signature.parameters["config"].annotation is not argparse.Namespace

    config = module.SolveConfig(
        tiling="kagome",
        supercell="2x2",
        n_b=1,
        solver="frontier-dp",
        species=("A", "B"),
        pbc=True,
        emit_json=False,
    )

    result = module.run_solve(config)
    payload = module.to_payload(result)
    text = module.format_text(result)

    assert result.labels is not None
    assert result.cut_value == 4
    assert result.n_frustrated is not None
    assert payload["supercell"] == [2, 2]
    assert payload["solver"] == "frontier-dp"
    assert "labels:" in text
    assert "n_frustrated:" in text


def test_enumeration_workflow_exposes_typed_api_and_rows() -> None:
    """The enumeration workflow is callable without an ``argparse.Namespace``."""
    module = _import_required("archimono.workflows.enumerate")

    assert is_dataclass(module.EnumerationConfig)
    assert is_dataclass(module.EnumerationResult)
    signature = inspect.signature(module.enumerate_structures)
    assert list(signature.parameters) == ["config"]
    assert signature.parameters["config"].annotation is not argparse.Namespace

    config = module.EnumerationConfig(
        tiling="kagome",
        n_b=1,
        supercell="2x2",
        min_cut=0,
        backend="orderly",
        emit_json=False,
    )

    result = module.enumerate_structures(config)
    payload = module.to_payload(result)
    text_lines = module.format_text(result).splitlines()

    assert len(result.structures) > 0
    assert all(sum(structure.labels) == 1 for structure in result.structures)
    assert payload["count"] == len(result.structures)
    assert payload["supercell"] == [2, 2]
    assert text_lines[0] == f"count: {len(result.structures)}"
    assert all("labels:" in line and "cut_value:" in line for line in text_lines[1:])
