"""Frontier-order benchmark workflow: compare vertex-ordering heuristics."""

from __future__ import annotations

import pathlib
from dataclasses import dataclass
from typing import TextIO

from archimono.assignment import (
    compute_frontier_sizes,
    compute_frontier_width,
    estimate_peak_memory_bytes,
)
from archimono.certification import (
    TILINGS as _CATALOGUE,
)
from archimono.certification import (
    format_supercell,
)
from archimono.tilings import base, registry
from archimono.workflows._benchmark_common import (
    _print,
    _resolve_target_sizes,
    _selected_tilings,
)


@dataclass(slots=True)
class FrontierOrderConfig:
    """Configuration for the frontier-order comparison workflow.

    Attributes:
        target_sizes: Explicit target sizes, or ``None`` to use the default
            sequence truncated by ``max_n``.
        max_n: Upper bound applied when ``target_sizes`` is ``None``.
        tilings: Optional tiling-key filter.  ``"frustrated"`` expands to the
            frustrated Archimedean tilings.
        md_out: Markdown output path.
    """

    target_sizes: list[int] | None
    max_n: int
    tilings: list[str] | None
    md_out: pathlib.Path


@dataclass(slots=True)
class FrontierOrderResult:
    """Result from the frontier-order workflow.

    Attributes:
        markdown_path: Markdown path written.
        sections: Per-tiling sections rendered into the report.
    """

    markdown_path: pathlib.Path
    sections: list[_FoSection]


_FO_DEFAULT_SIZES: list[int] = [12, 24, 36, 48, 60, 72, 84, 96]
_FO_HEURISTICS: tuple[str, ...] = ("natural", "ck", "greedy", "geometric")

# (n, supercell, per-heuristic widths, memory estimate in GB for the winner)
_FoEntry = tuple[int, str, dict[str, int], float]
_FoSection = tuple[str, str, int, int, list[_FoEntry]]


def _fo_winner(widths: dict[str, int]) -> str:
    """Return slash-joined names of heuristics tied for minimum frontier width.

    Args:
        widths: Mapping from heuristic name to its maximum frontier width.
    """
    min_w = min(widths.values())
    return "/".join(h for h in _FO_HEURISTICS if widths[h] == min_w)


def _render_frontier_order_markdown(sections: list[_FoSection]) -> str:
    """Render the frontier-order comparison markdown document."""
    lines: list[str] = [
        "# Frontier Order Comparison",
        "",
        "Compares vertex-ordering heuristics by **maximum frontier width k**"
        " across all Archimedean tilings and supercells.",
        "DP memory scales as O(2^k), so the heuristic with the smallest k"
        " is preferred to avoid out-of-memory errors.",
        "",
        "Heuristics: `natural` (ascending node ID), `ck` (Cuthill-McKee),"
        " `greedy` (min-degree elimination), `geometric` (best geometric sweep).",
        "",
        "## Per-tiling detail",
        "",
    ]

    for _, tiling_display, n_v, coord, entries in sections:
        if not entries:
            continue
        lines += [
            f"### {tiling_display}",
            "",
            f"n_v = {n_v}, coordination = {coord}",
            "",
            "| n | supercell | natural | ck | greedy | geometric | winner"
            " | memory (best k) |",
            "|---|-----------|---------|----|---------|-----------|----|---------|",
        ]
        for n, supercell, widths, mem_gb in entries:
            w = _fo_winner(widths)
            lines.append(
                f"| {n} | {supercell} |"
                f" {widths['natural']} | {widths['ck']} |"
                f" {widths['greedy']} | {widths['geometric']} | {w} | {mem_gb:.1f} GB |"
            )
        lines.append("")

    return "\n".join(lines) + "\n"


def run_frontier_order(
    config: FrontierOrderConfig,
    *,
    stream: TextIO | None = None,
) -> FrontierOrderResult:
    """Run the frontier-order comparison workflow."""
    target_sizes = _resolve_target_sizes(
        config.target_sizes, default=_FO_DEFAULT_SIZES, max_n=config.max_n
    )
    selected = _selected_tilings(config.tilings, _CATALOGUE)
    sections: list[_FoSection] = []

    for tiling_key, tiling_display in selected:
        tiling = registry.get(tiling_key)
        n_v = tiling.n_vertices
        coord = tiling.coordination
        _print(stream, f"\n[{tiling_display}] n_v={n_v}", flush=True)
        entries: list[_FoEntry] = []

        for target_n in target_sizes:
            matrices = list(base.valid_supercell_matrices(tiling, target_n))
            for matrix in matrices:
                graph = tiling.graph(matrix)
                n = graph.number_of_nodes()
                supercell = format_supercell(matrix)
                widths = {h: compute_frontier_width(graph, h) for h in _FO_HEURISTICS}
                best_heuristic = min(
                    _FO_HEURISTICS, key=lambda h: (widths[h], _FO_HEURISTICS.index(h))
                )
                mem_gb = (
                    estimate_peak_memory_bytes(
                        compute_frontier_sizes(graph, best_heuristic), n // 2
                    )
                    / 1e9
                )
                entries.append((n, supercell, widths, mem_gb))
                _print(
                    stream,
                    f"  n={n:>3} {supercell:<14}"
                    + "  ".join(f"{h}={widths[h]}" for h in _FO_HEURISTICS),
                    flush=True,
                )

        sections.append((tiling_key, tiling_display, n_v, coord, entries))

    md = _render_frontier_order_markdown(sections)
    config.md_out.parent.mkdir(parents=True, exist_ok=True)
    config.md_out.write_text(md, encoding="utf-8")
    _print(stream, f"\nMarkdown written to {config.md_out}", flush=True)
    return FrontierOrderResult(markdown_path=config.md_out, sections=sections)
