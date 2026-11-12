"""Fast static checks for the benchmark CLI adapter boundary."""

from __future__ import annotations

import ast
import pathlib

_PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[3]
_BENCHMARK_CLI = _PROJECT_ROOT / "src" / "archimono" / "cli" / "benchmark.py"

_BANNED_TOP_LEVEL_IMPORTS = (
    "matplotlib",
    "networkx",
    "numpy",
    "sqlite3",
    "archimono.assignment",
    "archimono.tilings.registry",
    "archimono.workflows.benchmark",
)


def _is_banned_import(module_name: str) -> bool:
    """Return whether *module_name* is banned at CLI module import time.

    Args:
        module_name: Imported module name from an AST import node.

    Returns:
        ``True`` when the import targets a banned module or submodule.
    """
    return any(
        module_name == banned or module_name.startswith(f"{banned}.")
        for banned in _BANNED_TOP_LEVEL_IMPORTS
    )


def test_benchmark_cli_has_no_heavy_top_level_imports() -> None:
    tree = ast.parse(_BENCHMARK_CLI.read_text(encoding="utf-8"))
    banned: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _is_banned_import(alias.name):
                    banned.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            if _is_banned_import(node.module):
                banned.append(node.module)

    assert banned == []


def test_benchmark_cli_public_defs_are_command_seam_only() -> None:
    tree = ast.parse(_BENCHMARK_CLI.read_text(encoding="utf-8"))
    public_defs = {
        node.name
        for node in tree.body
        if isinstance(node, ast.FunctionDef | ast.ClassDef)
        and not node.name.startswith("_")
    }

    assert public_defs == {"configure_parser", "run"}
