"""Parser adapter for the ``archimono verify`` command."""

import argparse as _argparse
import importlib as _importlib
from pathlib import Path as _Path
from typing import Any as _Any

from archimono import _paths

__all__ = ["configure_parser", "run"]

_PROJECT_ROOT = _paths.PROJECT_ROOT
_VERIFY_MODES = ("geometry", "exact-solver", "visualize", "all")


def configure_parser(subparser: _argparse.ArgumentParser) -> None:
    """Configure the ``archimono verify`` subparser."""
    subparser.description = "Verification and visualisation for archimono."
    subparser.formatter_class = _argparse.RawDescriptionHelpFormatter
    subparser.epilog = (
        "Modes:\n"
        "  geometry      Geometry checks + PNG per tiling + overview grid\n"
        "  exact-solver  BruteforceSolver against reference cases + PNGs\n"
        "  visualize     Solve all 11 tilings and render assignment PNGs\n"
        "  all           Run all three modes in sequence (default)\n"
    )
    subparser.add_argument(
        "--mode",
        choices=_VERIFY_MODES,
        default="all",
        help="Verification mode to run (default: all).",
    )
    subparser.add_argument(
        "--out-dir",
        type=_Path,
        default=_PROJECT_ROOT / "tmp" / "verify",
        help="Output directory for PNG images (default: tmp/verify/).",
    )


def run(args: _argparse.Namespace) -> int:
    """Run the selected verification modes; return 0 if all pass, else 1."""
    verification: _Any = _importlib.import_module("archimono.verification")
    config = verification.VerificationConfig(mode=args.mode, out_dir=args.out_dir)
    result = verification.run_verification(config)
    return 0 if result.ok else 1
