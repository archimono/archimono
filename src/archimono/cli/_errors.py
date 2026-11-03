"""Shared CLI error-output helper."""

from __future__ import annotations

import sys


def print_cli_error(exc: KeyError | ValueError) -> None:
    """Print *exc* to stderr; a KeyError prints its message without repr quotes."""
    message = exc.args[0] if isinstance(exc, KeyError) and exc.args else str(exc)
    print(message, file=sys.stderr)
