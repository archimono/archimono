"""Unified command-line entry point for archimono workflows."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from typing import Protocol, cast, runtime_checkable

from archimono.cli import (
    benchmark,
    certify,
    list_tilings,
    solve,
    verify,
)
from archimono.cli import (
    enumerate as enumerate_command,
)


class _CommandModule(Protocol):
    """Protocol implemented by package CLI command modules."""

    def configure_parser(self, subparser: argparse.ArgumentParser) -> None:
        """Configure one subcommand parser."""

    def run(self, args: argparse.Namespace) -> int | None:
        """Run one subcommand; None means exit code 0."""


@runtime_checkable
class _ReconfigurableTextStream(Protocol):
    """Protocol for text streams that can change their encoding."""

    def reconfigure(self, *, encoding: str) -> None:
        """Set this stream's text encoding."""


_COMMANDS: tuple[tuple[str, str, _CommandModule], ...] = (
    (
        "verify",
        "Run verification and visualization tests.",
        verify,
    ),
    (
        "certify",
        "Run MAX-CUT solver certification workflows.",
        certify,
    ),
    (
        "benchmark",
        "Benchmark solver execution, vertex ordering, and JIT kernels.",
        benchmark,
    ),
    (
        "list-tilings",
        "List all registered Archimedean tilings and primitive vertex counts.",
        list_tilings,
    ),
    (
        "solve",
        "Solve MAX-CUT for a specific tiling assignment.",
        solve,
    ),
    (
        "enumerate",
        "Enumerate inequivalent binary structures on a tiling.",
        enumerate_command,
    ),
)


def _build_parser() -> argparse.ArgumentParser:
    """Build the top-level ``archimono`` parser."""
    parser = argparse.ArgumentParser(
        prog="archimono",
        description=(
            "Systematic structure enumeration of 2D monolayer networks on "
            "Archimedean tilings."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
        metavar="COMMAND",
    )

    for name, help_text, command_module in _COMMANDS:
        command_parser = subparsers.add_parser(name, help=help_text)
        command_module.configure_parser(command_parser)
        command_parser.set_defaults(_run=command_module.run)

    return parser


def _configure_stdio_encoding() -> None:
    """Configure real stdio streams for Unicode CLI output."""
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, _ReconfigurableTextStream):
            stream.reconfigure(encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Run ``archimono`` on *argv* (default ``sys.argv``); return the exit code."""
    _configure_stdio_encoding()
    parser = _build_parser()
    args = parser.parse_args(argv)
    run = cast(Callable[[argparse.Namespace], int | None], args._run)
    result = run(args)
    if isinstance(result, int):
        return result
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
