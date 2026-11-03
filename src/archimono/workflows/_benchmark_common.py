"""Shared output and tiling-selection helpers for the benchmark workflows."""

from __future__ import annotations

from typing import TextIO

from archimono.certification import (
    FRUSTRATED_TILING_KEYS as _FRUSTRATED_TILING_KEYS,
)
from archimono.certification import (
    TILINGS as _CATALOGUE,
)
from archimono.certification import (
    resolve_target_sizes,
)

# Short-display variant of the certification catalogue: "6³ (hexagonal)"
# becomes "6³". These short names are persisted in the runs DB.
_ALL_TILINGS: list[tuple[str, str]] = [
    (key, display.split(" (")[0]) for key, display in _CATALOGUE
]


def _print(
    stream: TextIO | None,
    *values: object,
    sep: str = " ",
    end: str = "\n",
    flush: bool = False,
) -> None:
    """Write *values* to *stream*, or to standard output if it is None."""
    print(*values, sep=sep, end=end, file=stream, flush=flush)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _resolve_target_sizes(
    target_sizes: list[int] | None,
    *,
    default: list[int],
    max_n: int,
) -> list[int]:
    """Return sorted, deduplicated positive target sizes.

    Bare ``--target-sizes`` yields ``[]``, which falls back to the defaults
    exactly like the omitted flag.

    Args:
        target_sizes: Explicit sizes from the workflow config, possibly empty.
        default: Fallback list when no explicit sizes were given.
        max_n: Upper bound applied to the defaults.
    """
    return resolve_target_sizes(target_sizes or None, default=default, max_n=max_n)


def _selected_tilings(
    tilings: list[str] | None,
    catalogue: list[tuple[str, str]],
) -> list[tuple[str, str]]:
    """Return the ``(key, display)`` entries of *catalogue* selected by *tilings*.

    Raises:
        ValueError: If any requested tiling key is not in ``catalogue``.
    """
    if tilings is None:
        return catalogue
    selected = {name.lower() for name in tilings}
    if "frustrated" in selected:
        selected = (selected - {"frustrated"}) | _FRUSTRATED_TILING_KEYS
    matches = [(k, d) for k, d in catalogue if k.lower() in selected]
    unknown = sorted(selected - {k.lower() for k, _ in catalogue})
    if unknown:
        raise ValueError(f"Unknown tilings: {unknown}")
    return matches
