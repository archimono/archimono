"""Shared ``--supercell`` help text for the ``solve`` and ``enumerate`` commands."""

from __future__ import annotations

# Smallest diagonal supercell whose periodic graph builds, per tiling.
# tests/fast/cli/test_supercell_help.py checks every entry against Tiling.graph.
MIN_DIAGONAL_SUPERCELL: tuple[tuple[str, str, tuple[int, int]], ...] = (
    ("3.12²", "truncated_hexagonal", (1, 1)),
    ("3⁴.6", "snub_hexagonal", (1, 1)),
    ("3.4.6.4", "rhombitrihexagonal", (1, 1)),
    ("4.6.12", "truncated_trihexagonal", (1, 1)),
    ("3³.4²", "elongated_triangular", (3, 1)),
    ("3².4.3.4", "snub_square", (2, 2)),
    ("3.6.3.6", "kagome", (2, 2)),
    ("4.8²", "truncated_square", (2, 2)),
    ("4⁴", "square", (2, 2)),
    ("6³", "hexagonal", (2, 2)),
    ("3⁶", "triangular", (3, 3)),
)

SUPERCELL_OPTION_HELP = (
    "Diagonal supercell as AxB or A,B (default: 1x1). Many tilings need 2x2 "
    'or larger; see "Supercells" below.'
)


def _render_epilog() -> str:
    lines = [
        "Supercells:",
        "  The graph wraps around the supercell boundary (periodic boundaries).",
        "  If the supercell is too small, wrapping joins a node to itself, joins",
        "  one node pair by two distinct edges, or breaks the two-colouring of a",
        "  bipartite tiling. The graph stores at most one edge per node pair, so",
        "  these sizes are rejected instead of being built with missing edges.",
        "  Use a larger supercell.",
        "",
        "  Smallest supported diagonal supercell:",
    ]
    sizes = sorted(
        {size for _, _, size in MIN_DIAGONAL_SUPERCELL},
        key=lambda s: (s[0] * s[1], s),
    )
    for size in sizes:
        label = f"{size[0]}x{size[1]}"
        for config, key, s in MIN_DIAGONAL_SUPERCELL:
            if s == size:
                lines.append(f"    {label:<5}{config} ({key})")
                label = ""
    lines += [
        "  Larger sizes are usually supported, but not always; the command",
        "  reports any size it rejects.",
    ]
    return "\n".join(lines) + "\n"


SUPERCELL_EPILOG = _render_epilog()
