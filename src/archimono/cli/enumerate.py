"""CLI adapter for enumerating inequivalent binary structures."""

from __future__ import annotations

import argparse as _argparse
from typing import cast as _cast

from archimono.cli._errors import print_cli_error as _print_cli_error
from archimono.cli._supercell_help import SUPERCELL_EPILOG as _SUPERCELL_EPILOG
from archimono.cli._supercell_help import (
    SUPERCELL_OPTION_HELP as _SUPERCELL_OPTION_HELP,
)

__all__ = ["configure_parser", "run"]

_BACKENDS = ("orderly", "icet")


def configure_parser(subparser: _argparse.ArgumentParser) -> None:
    """Configure the ``archimono enumerate`` subparser."""
    subparser.description = "Enumerate inequivalent binary structures."
    subparser.formatter_class = _argparse.RawDescriptionHelpFormatter
    subparser.epilog = _SUPERCELL_EPILOG
    subparser.add_argument("tiling", help="Tiling registry key.")
    subparser.add_argument(
        "--nb",
        type=int,
        required=True,
        help="Required count of the second species.",
    )
    subparser.add_argument(
        "--supercell",
        default="1x1",
        help=_SUPERCELL_OPTION_HELP,
    )
    subparser.add_argument(
        "--min-cut",
        type=int,
        default=0,
        help="Minimum cut value to include (default: 0).",
    )
    subparser.add_argument(
        "--backend",
        choices=_BACKENDS,
        default="orderly",
        help="Enumeration backend (default: orderly).",
    )
    subparser.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output.",
    )


def run(args: _argparse.Namespace) -> int | None:
    """Run enumeration; return 1 on invalid input, else None."""
    from archimono.workflows import enumerate as enumeration_workflow

    config = enumeration_workflow.EnumerationConfig(
        tiling=_cast(str, args.tiling),
        n_b=_cast(int, args.nb),
        supercell=_cast(str, args.supercell),
        min_cut=_cast(int, args.min_cut),
        backend=_cast(str, args.backend),
        emit_json=_cast(bool, args.json),
    )

    try:
        result = enumeration_workflow.enumerate_structures(config)
    except (KeyError, ValueError) as exc:
        _print_cli_error(exc)
        return 1

    if config.emit_json:
        print(enumeration_workflow.format_json(result))
    else:
        print(enumeration_workflow.format_text(result))
    return None
