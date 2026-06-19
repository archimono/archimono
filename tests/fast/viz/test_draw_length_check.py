"""Regression test: draw_tiling_assignment validates label length (audit viz guardrail).

Before the fix there was no length check between ``labels`` and the site count,
so a mismatched ``labels`` list produced silently wrong colours or an opaque
numpy error. The fix raises a clear ``ValueError`` up front.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend; must precede pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import pytest  # noqa: E402

from archimono.tilings import registry  # noqa: E402
from archimono.viz import draw_tiling_assignment  # noqa: E402


def test_mismatched_labels_length_raises() -> None:
    """A labels list shorter than the site count raises a clear ValueError."""
    tiling = registry.get("kagome")
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="labels length"):
            draw_tiling_assignment(ax, tiling, (2, 2), [0, 1, 0])  # too short
    finally:
        plt.close(fig)


def test_correct_labels_length_does_not_raise() -> None:
    """A correctly sized labels list renders without error."""
    tiling = registry.get("kagome")
    n_sites = 2 * 2 * tiling.n_vertices
    labels = [i % 2 for i in range(n_sites)]
    fig, ax = plt.subplots()
    try:
        draw_tiling_assignment(ax, tiling, (2, 2), labels)
    finally:
        plt.close(fig)
