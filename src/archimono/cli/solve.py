"""CLI adapter for solving one binary MAX-CUT assignment."""

from __future__ import annotations

import argparse as _argparse
from collections.abc import Sequence as _Sequence
from typing import cast as _cast

from archimono.cli._errors import print_cli_error as _print_cli_error
from archimono.cli._supercell_help import SUPERCELL_EPILOG as _SUPERCELL_EPILOG
from archimono.cli._supercell_help import (
    SUPERCELL_OPTION_HELP as _SUPERCELL_OPTION_HELP,
)

__all__ = ["configure_parser", "run"]

_SOLVERS = (
    "bruteforce",
    "frontier-dp",
    "annealing",
    "greedy",
    "hadlock",
)


def configure_parser(subparser: _argparse.ArgumentParser) -> None:
    """Configure the ``archimono solve`` subparser."""
    subparser.description = "Solve one tiling assignment."
    subparser.formatter_class = _argparse.RawDescriptionHelpFormatter
    subparser.epilog = _SUPERCELL_EPILOG
    subparser.add_argument("tiling", help="Tiling registry key.")
    subparser.add_argument(
        "--supercell",
        default="1x1",
        help=_SUPERCELL_OPTION_HELP,
    )
    subparser.add_argument(
        "--nb",
        type=int,
        default=None,
        help="Count of the second species; defaults to the solver default.",
    )
    subparser.add_argument(
        "--solver",
        choices=_SOLVERS,
        default="bruteforce",
        help="Solver backend (default: bruteforce).",
    )
    subparser.add_argument(
        "--species",
        nargs="*",
        default=("A", "B"),
        help="Two species symbols; label 1 counts toward --nb (default: A B).",
    )
    subparser.add_argument(
        "--pbc",
        dest="pbc",
        action="store_true",
        default=True,
        help="Build a periodic graph (default).",
    )
    subparser.add_argument(
        "--no-pbc",
        dest="pbc",
        action="store_false",
        help="Build a non-periodic graph.",
    )
    subparser.add_argument(
        "--json",
        action="store_true",
        help="Emit JSON with raw integer labels.",
    )
    subparser.add_argument(
        "--memory-limit-gb",
        type=float,
        default=32.0,
        help=(
            "[frontier-dp only] Reject the solve if the reconstruction is "
            "estimated to exceed this memory limit in GB. Default: 32."
        ),
    )
    subparser.add_argument(
        "--all-optima",
        action="store_true",
        default=False,
        help=(
            "[frontier-dp only] Enumerate every optimal assignment and "
            "include all_labels in metadata. Default: off (one witness)."
        ),
    )
    subparser.add_argument(
        "--max-optima",
        type=int,
        default=10_000,
        help=(
            "[frontier-dp only] Cap on materialized optima when --all-optima "
            "is passed. Default: 10000."
        ),
    )


def run(args: _argparse.Namespace) -> int | None:
    """Run the solver; return 1 on invalid input, else None."""
    from archimono.workflows import solve as solve_workflow

    config = solve_workflow.SolveConfig(
        tiling=_cast(str, args.tiling),
        supercell=_cast(str, args.supercell),
        n_b=_cast(int | None, args.nb),
        solver=_cast(str, args.solver),
        species=_cast(_Sequence[str], args.species),
        pbc=_cast(bool, args.pbc),
        emit_json=_cast(bool, args.json),
        memory_limit_gb=_cast(float, args.memory_limit_gb),
        enumerate_all=_cast(bool, args.all_optima),
        max_optima=_cast(int, args.max_optima),
    )

    try:
        result = solve_workflow.run_solve(config)
    except (KeyError, ValueError) as exc:
        _print_cli_error(exc)
        return 1

    if config.emit_json:
        print(solve_workflow.format_json(result))
    else:
        print(solve_workflow.format_text(result))
    return None
