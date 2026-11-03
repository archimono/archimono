"""CLI adapter for solver-certification workflows."""

from __future__ import annotations

import argparse as _argparse
import pathlib as _pathlib
import sys as _sys

from archimono import _paths

__all__ = ["configure_parser", "run"]

_DEFAULT_TARGET_SIZES_BRUTEFORCE: list[int] = [12, 24]
_DEFAULT_TARGET_SIZES_FRONTIER_DP_VALUE: list[int] = [12, 24, 36, 48, 60]
_DEFAULT_TARGET_SIZES_ANNEALING: list[int] = [12, 24, 36, 48, 60]

_DEFAULT_MAX_N_BRUTEFORCE: int = 40
_DEFAULT_MAX_N_FRONTIER_DP_VALUE: int = 60
_DEFAULT_MAX_N_ANNEALING: int = 60

_PROJECT_ROOT = _paths.PROJECT_ROOT


def configure_parser(subparser: _argparse.ArgumentParser) -> None:
    """Configure the ``archimono certify`` subparser."""
    subparser.description = (
        "Unified MAX-CUT certification workflow. "
        "Dispatches to the selected solver backend."
    )
    subparser.add_argument(
        "--solver",
        choices=["bruteforce", "frontier-dp-value", "annealing"],
        required=True,
        help="Solver backend to use.",
    )
    subparser.add_argument(
        "--max-n",
        type=int,
        default=None,
        help=(
            "Maximum target size to consider when --target-sizes is omitted. "
            "Defaults to 40 for bruteforce and 60 for "
            "frontier-dp-value/annealing."
        ),
    )
    subparser.add_argument(
        "--target-sizes",
        type=int,
        nargs="*",
        default=None,
        help="Explicit target sizes. Overrides the per-solver defaults.",
    )
    subparser.add_argument(
        "--tilings",
        nargs="*",
        default=None,
        help=(
            "Tiling keys to solve. Accepts individual registry keys, 'all', or "
            "'frustrated' (the 7 non-bipartite tilings). "
            "Defaults to 'all'."
        ),
    )
    subparser.add_argument(
        "--json-out-dir",
        type=_pathlib.Path,
        default=None,
        help="Directory for per-case JSON output. Defaults vary by solver.",
    )
    subparser.add_argument(
        "--markdown-out",
        type=_pathlib.Path,
        default=None,
        help="Path for markdown summary output. Defaults vary by solver.",
    )
    subparser.add_argument(
        "--from-json-dir",
        type=_pathlib.Path,
        nargs="?",
        const=True,
        default=None,
        help=(
            "Skip solving and rebuild markdown from existing JSON files. "
            "Pass a directory path or omit the path argument to use the "
            "solver's default json-out-dir."
        ),
    )
    subparser.add_argument(
        "--recursive",
        action="store_true",
        help="When --from-json-dir is set, recurse into subdirectories.",
    )
    subparser.add_argument(
        "--one-supercell",
        action="store_true",
        help="[bruteforce only] Solve only the single squarest supercell per case.",
    )
    subparser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=60.0,
        help=(
            "[frontier-dp-value only] Skip supercells whose estimated peak "
            "memory exceeds this limit in GB. Default: 60."
        ),
    )
    subparser.add_argument(
        "--bruteforce-md",
        type=_pathlib.Path,
        default=_PROJECT_ROOT / "docs" / "draft" / "bruteforce-certification.md",
        help=(
            "[frontier-dp-value only] Path to bruteforce-certification.md "
            "for the note column. Ignored if the file does not exist."
        ),
    )


def _default_json_out_dir(solver: str) -> _pathlib.Path:
    """Return the default JSON output directory for *solver*."""
    return {
        "bruteforce": _PROJECT_ROOT / "tmp" / "bruteforce-certification",
        "frontier-dp-value": _PROJECT_ROOT / "tmp" / "certify-max-cut",
        "annealing": _PROJECT_ROOT / "tmp" / "annealing-stress",
    }[solver]


def _default_markdown_out(solver: str) -> _pathlib.Path:
    """Return the default markdown output path for *solver*."""
    # Defaults write under tmp/ (not into the committed docs tree) so a partial
    # or smoke run cannot silently overwrite a committed report; pass an explicit
    # --markdown-out docs/... to publish.
    return _PROJECT_ROOT / "tmp" / {
        "bruteforce": "bruteforce-certification.md",
        "frontier-dp-value": "certified-max-cut.md",
        "annealing": "stress-test-results.md",
    }[solver]


def _default_max_n(solver: str) -> int:
    """Return the default maximum target size for *solver*."""
    return {
        "bruteforce": _DEFAULT_MAX_N_BRUTEFORCE,
        "frontier-dp-value": _DEFAULT_MAX_N_FRONTIER_DP_VALUE,
        "annealing": _DEFAULT_MAX_N_ANNEALING,
    }[solver]


def _default_target_sizes(solver: str) -> list[int]:
    """Return the default target-size list for *solver*."""
    return {
        "bruteforce": _DEFAULT_TARGET_SIZES_BRUTEFORCE,
        "frontier-dp-value": _DEFAULT_TARGET_SIZES_FRONTIER_DP_VALUE,
        "annealing": _DEFAULT_TARGET_SIZES_ANNEALING,
    }[solver]


def run(args: _argparse.Namespace) -> int | None:
    """Run the certification workflow from parsed arguments."""
    from archimono.certification import resolve_target_sizes
    from archimono.workflows.certify import (
        CertificationWorkflowConfig,
        run_certification_workflow,
    )

    solver = args.solver
    json_out_dir = args.json_out_dir or _default_json_out_dir(solver)
    markdown_out = args.markdown_out or _default_markdown_out(solver)
    tilings = args.tilings or ["all"]
    max_n = args.max_n if args.max_n is not None else _default_max_n(solver)
    target_sizes = resolve_target_sizes(
        args.target_sizes,
        default=_default_target_sizes(solver),
        max_n=max_n,
    )

    from_json_dir = None
    if args.from_json_dir is True:
        from_json_dir = json_out_dir
    elif args.from_json_dir is not None:
        from_json_dir = _pathlib.Path(args.from_json_dir)

    config = CertificationWorkflowConfig(
        solver=solver,
        json_out_dir=json_out_dir,
        markdown_out=markdown_out,
        target_sizes=target_sizes,
        tilings=tilings,
        from_json_dir=from_json_dir,
        recursive=args.recursive,
        one_supercell=args.one_supercell,
        memory_limit_gb=args.memory_limit_gb,
        bruteforce_md=args.bruteforce_md,
    )
    run_certification_workflow(config, stream=_sys.stdout)
    return None
