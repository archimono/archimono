"""CLI adapter that lists registered tilings."""

from __future__ import annotations

import argparse as _argparse

__all__ = ["configure_parser", "run"]


def configure_parser(subparser: _argparse.ArgumentParser) -> None:
    """Configure the ``archimono list-tilings`` subparser."""
    subparser.description = "List registered tiling keys and vertex counts."


def run(args: _argparse.Namespace) -> int | None:
    """Print every registered tiling key with its primitive vertex count."""
    from archimono.tilings import registry

    _ = args
    for key in registry.available():
        print(f"{key} {registry.get(key).n_vertices}")
    return None
