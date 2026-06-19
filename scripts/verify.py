"""Consolidated verification and visualisation entry-point for archimono.

Runs one or more verification modes via ``--mode``:

* ``geometry``     — geometry checks + PNG per tiling + overview grid
                     (migrated from ``verify_tilings.py``)
* ``exact-solver`` — BruteforceSolver against hardcoded reference cases + PNGs
                     (migrated from ``verify_exact_solver.py``)
* ``visualize``    — solve all 11 tilings and render PNGs
                     (migrated from ``visualize_kagome.py``)
* ``all``          — run all three in sequence (default)

Usage::

    python scripts/verify.py [--mode {geometry,exact-solver,visualize,all}]
                              [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias, TypedDict

import matplotlib

matplotlib.use("Agg")
# isort: split

import matplotlib.axes
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from matplotlib.patches import Polygon

from archimono.assignment import AnnealingSolver, BipartiteAssigner, BruteforceSolver
from archimono.tilings import registry
from archimono.tilings.base import Tiling
from archimono.viz import (
    draw_tiling_assignment,
    iter_pbc_edges,
    legend_handles,
    supercell_positions,
)

# Type aliases used in exact-solver and visualize modes.
if TYPE_CHECKING:
    _RenderedCase: TypeAlias = (
        tuple[str, Tiling, nx.Graph[int], tuple[int, ...], list[str]]
    )
else:
    _RenderedCase: TypeAlias = tuple[str, Tiling, nx.Graph, tuple[int, ...], list[str]]
_VisResult: TypeAlias = tuple[
    str,
    str,
    tuple[int, int],
    Tiling,
    tuple[int, ...],
    float,
    int,
]

# ---------------------------------------------------------------------------
# geometry mode — constants
# ---------------------------------------------------------------------------

# (n_prim_verts, coord, is_bipartite, bond_Å)
_EXPECTED: dict[str, tuple[int, int, bool, float]] = {
    "4⁴":        (2,  4, True,  1.80),
    "6³":        (2,  3, True,  1.42),
    "4.8²":      (4,  3, True,  1.50),
    "4.6.12":    (12, 3, True,  1.42),
    "3⁶":        (1,  6, False, 2.46),
    "3.6.3.6":   (3,  4, False, 1.42),
    "3³.4²":     (4,  5, False, 1.50),
    "3².4.3.4":  (4,  5, False, 1.50),
    "3⁴.6":      (6,  5, False, 1.42),
    "3.4.6.4":   (6,  4, False, 1.42),
    "3.12²":     (6,  3, False, 1.42),
}

# Visualisation supercells (chosen for ~20-60 nodes in the plot)
_VIS_SC: dict[str, tuple[int, int]] = {
    "4⁴":       (5, 5),
    "6³":       (5, 5),
    "4.8²":     (4, 4),
    "4.6.12":   (2, 2),
    "3⁶":       (6, 6),
    "3.6.3.6":  (4, 4),
    "3³.4²":    (4, 4),
    "3².4.3.4": (4, 4),
    "3⁴.6":     (3, 3),
    "3.4.6.4":  (3, 3),
    "3.12²":    (2, 2),
}

# Even supercell used only for the bipartite check
_BPCHECK_SC: dict[str, tuple[int, int]] = {k: (4, 4) for k in _EXPECTED}

_COLORS: dict[bool, str] = {True: "#4C72B0", False: "#DD8452"}

# ---------------------------------------------------------------------------
# exact-solver mode — constants
# ---------------------------------------------------------------------------


class _CaseSpec(TypedDict):
    """Typed specification for one BruteforceSolver reference case."""

    supercell: tuple[int, int]
    n_b: int
    expected_cut: float
    expected_frustrated: int


_CASES: dict[str, _CaseSpec] = {
    "hexagonal": {
        "supercell": (2, 2),
        "n_b": 4,
        "expected_cut": 12.0,
        "expected_frustrated": 0,
    },
    "triangular": {
        "supercell": (2, 2),
        "n_b": 2,
        "expected_cut": 4.0,
        "expected_frustrated": 2,
    },
    "kagome": {
        "supercell": (2, 2),
        "n_b": 6,
        "expected_cut": 16.0,
        "expected_frustrated": 8,
    },
    "truncated_hexagonal": {
        "supercell": (2, 2),
        "n_b": 12,
        "expected_cut": 28.0,
        "expected_frustrated": 8,
    },
}

# ---------------------------------------------------------------------------
# visualize mode — constants
# ---------------------------------------------------------------------------

# (canonical name, human-readable title, supercell)
_TILINGS: list[tuple[str, str, tuple[int, int]]] = [
    ("hexagonal",              "6³ — Hexagonal",                (2, 2)),
    ("square",                 "4⁴ — Square",                   (2, 2)),
    ("truncated_square",       "4.8² — Truncated Square",       (2, 2)),
    ("truncated_trihexagonal", "4.6.12 — Omnitruncated Trihex", (2, 2)),
    ("kagome",                 "3.6.3.6 — Kagome",              (2, 2)),
    ("triangular",             "3⁶ — Triangular",               (2, 2)),
    ("truncated_hexagonal",    "3.12² — Truncated Hexagonal",   (2, 2)),
    ("elongated_triangular",   "3³.4² — Elongated Triangular",  (2, 2)),
    ("rhombitrihexagonal",     "3.4.6.4 — Rhombitrihex.",       (2, 2)),
    ("snub_square",            "3².4.3.4 — Snub Square",        (2, 2)),
    ("snub_hexagonal",         "3⁴.6 — Snub Hexagonal",        (2, 2)),
]

_VIS_SOLVER = AnnealingSolver(n_restarts=50, seed=42)

# ---------------------------------------------------------------------------
# Helpers shared by geometry mode
# ---------------------------------------------------------------------------


def _real_edges(
    g: nx.Graph[int],
    bond: float,
    tol_frac: float = 0.12,
) -> list[tuple[int, int]]:
    """Return only edges whose Euclidean length is approximately *bond*.

    Wrap-around PBC edges are excluded because their Cartesian length
    differs from the physical bond length.

    Args:
        g: NetworkX graph with ``pos`` node attributes.
        bond: Expected bond length in Å.
        tol_frac: Fractional tolerance applied symmetrically around *bond*.

    Returns:
        List of ``(u, v)`` edge tuples whose length lies within the
        tolerance band.
    """
    hi = bond * (1.0 + tol_frac)
    lo = bond * (1.0 - tol_frac)
    out: list[tuple[int, int]] = []
    for u, v in g.edges():
        pu = np.array(g.nodes[u]["pos"], dtype=float)
        pv = np.array(g.nodes[v]["pos"], dtype=float)
        d = float(np.linalg.norm(pu - pv))
        if lo <= d <= hi:
            out.append((u, v))
    return out


def _check_bond_lengths(
    g: nx.Graph[int],
    bond: float,
) -> tuple[bool, float, float]:
    """Return ``(ok, min_length, max_length)`` for real bonds in *g*.

    Args:
        g: NetworkX graph with ``pos`` node attributes.
        bond: Expected bond length in Å.

    Returns:
        Tuple of a boolean pass/fail flag and the observed min and max
        bond lengths.
    """
    real = _real_edges(g, bond)
    if not real:
        return False, 0.0, 0.0
    lengths = [
        float(
            np.linalg.norm(
                np.array(g.nodes[u]["pos"], dtype=float)
                - np.array(g.nodes[v]["pos"], dtype=float)
            )
        )
        for u, v in real
    ]
    lo, hi = min(lengths), max(lengths)
    tol = 0.05 * bond
    ok = abs(lo - bond) < tol and abs(hi - bond) < tol
    return ok, lo, hi


def _render_pbc_tiling(
    ax: matplotlib.axes.Axes,
    tiling: Tiling,
    supercell: tuple[int, int],
    bond: float,
    *,
    color: str,
    atom_size: float = 40.0,
    bond_width: float = 1.4,
    cell_lw: float = 1.4,
    ghost_alpha: float = 0.42,
) -> None:
    """Draw a tiling on *ax* with PBC ghosts and a unit-cell outline.

    Geometry-only view (single species colour).  Uses
    :func:`archimono.viz.supercell_positions` and
    :func:`archimono.viz.iter_pbc_edges` for consistent geometry.

    Args:
        ax: Matplotlib axes to draw on.
        tiling: Tiling instance providing geometry.
        supercell: ``(na, nb)`` supercell dimensions.
        bond: Expected bond length in Å (used for view padding).
        color: Atom fill colour.
        atom_size: Scatter marker area in points².
        bond_width: Bond line width in points.
        cell_lw: Supercell outline line width.
        ghost_alpha: Opacity of ghost atoms in image cells.
    """
    na, nb = supercell
    a1: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        tiling.lattice_vectors[0], dtype=float
    )
    a2: np.ndarray[tuple[int], np.dtype[np.float64]] = np.array(
        tiling.lattice_vectors[1], dtype=float
    )
    super_a = na * a1
    super_b = nb * a2
    pos = supercell_positions(tiling, na, nb)

    for ga in (-1, 0, 1):
        for gb in (-1, 0, 1):
            if ga == 0 and gb == 0:
                continue
            shift = ga * super_a + gb * super_b
            ax.scatter(
                (pos + shift)[:, 0], (pos + shift)[:, 1],
                s=atom_size * 0.65, c=color,
                edgecolors="white", linewidths=0.3, alpha=ghost_alpha,
                zorder=2,
            )

    for u, v, img_a, img_b in iter_pbc_edges(tiling, supercell):
        pu = pos[u]
        pv = pos[v] + img_a * super_a + img_b * super_b
        ax.plot(
            [pu[0], pv[0]], [pu[1], pv[1]],
            color="#666666", linewidth=bond_width, alpha=0.85,
            solid_capstyle="round", zorder=3,
        )
        if (img_a, img_b) != (0, 0):
            pu2 = pos[u] - img_a * super_a - img_b * super_b
            ax.plot(
                [pu2[0], pos[v][0]], [pu2[1], pos[v][1]],
                color="#666666", linewidth=bond_width, alpha=0.85,
                solid_capstyle="round", zorder=3,
            )

    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=atom_size, c=color,
        edgecolors="white", linewidths=0.5,
        zorder=5,
    )

    corners = [np.zeros(2), super_a, super_a + super_b, super_b]
    ax.add_patch(
        Polygon(corners, fill=False, edgecolor="#222222",
                linewidth=cell_lw, linestyle="-")
    )

    pad = 2.5 * bond
    all_corners = np.array([np.zeros(2), super_a, super_a + super_b, super_b])
    xmin, ymin = (all_corners.min(axis=0) - pad).tolist()
    xmax, ymax = (all_corners.max(axis=0) + pad).tolist()
    dx, dy = xmax - xmin, ymax - ymin
    if dx > dy:
        extra = (dx - dy) / 2.0
        ymin -= extra
        ymax += extra
    elif dy > dx:
        extra = (dy - dx) / 2.0
        xmin -= extra
        xmax += extra
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal")
    ax.axis("off")


def _run_geometry_checks(
    key: str,
    tiling: Tiling,
    g_vis: nx.Graph[int],
    g_bp: nx.Graph[int],
) -> tuple[list[str], int | None]:
    """Run geometry verification checks for one tiling.

    Args:
        key: Vertex-configuration key in ``_EXPECTED``.
        tiling: Tiling instance.
        g_vis: Supercell graph used for visualisation.
        g_bp: Even supercell graph used for bipartite check.

    Returns:
        Tuple of ``(errors, coord)``: the list of error strings (empty means
        all checks passed) and the tiling's coordination number, or ``None``
        if computing it raised ``ValueError``. Returning the cached value lets
        callers display it without re-invoking ``tiling.coordination`` (whose
        re-raise would otherwise escape an unguarded call site).
    """
    exp_n_prim, exp_coord, exp_bip, exp_bond = _EXPECTED[key]
    errors: list[str] = []
    coord: int | None = None

    if tiling.n_vertices != exp_n_prim:
        errors.append(
            f"n_vertices: got {tiling.n_vertices}, expected {exp_n_prim}"
        )

    try:
        coord = tiling.coordination
        if coord != exp_coord:
            errors.append(f"coordination: got {coord}, expected {exp_coord}")
    except ValueError as exc:
        errors.append(f"coordination check failed: {exc}")

    if tiling.is_bipartite != exp_bip:
        errors.append(
            f"is_bipartite: got {tiling.is_bipartite}, expected {exp_bip}"
        )

    nx_bip = nx.is_bipartite(g_bp)
    if nx_bip != tiling.is_bipartite:
        errors.append(
            f"NetworkX bipartite={nx_bip} (even 4×4 supercell) "
            f"conflicts with declared is_bipartite={tiling.is_bipartite}"
        )

    bond_ok, lo, hi = _check_bond_lengths(g_vis, exp_bond)
    if not bond_ok:
        errors.append(
            f"bond lengths not uniform: min={lo:.4f} Å, max={hi:.4f} Å "
            f"(expected {exp_bond:.2f} Å ± 5%)"
        )

    exp_ec = exp_n_prim * exp_coord // 2
    actual_ec = len(tiling.edges)
    if actual_ec != exp_ec:
        errors.append(
            f"primitive edge count: got {actual_ec}, "
            f"expected {exp_n_prim}×{exp_coord}÷2 = {exp_ec}"
        )

    real = _real_edges(g_vis, exp_bond)
    if not real:
        errors.append("no real bonds found in the visualisation supercell")
    else:
        h = g_vis.edge_subgraph(real).copy()
        if not nx.is_connected(h):
            errors.append(
                "real-bond subgraph of the visualisation supercell is disconnected"
            )

    return errors, coord


# ---------------------------------------------------------------------------
# Helpers shared by exact-solver mode
# ---------------------------------------------------------------------------


def _safe_name(vertex_config: str) -> str:
    """Return a filesystem-safe stem for a tiling name.

    Args:
        vertex_config: Tiling registry key or name.

    Returns:
        String with non-alphanumeric runs replaced by underscores.
    """
    return re.sub(r"[^0-9A-Za-z]+", "_", vertex_config).strip("_")


def _run_exact_case(
    tiling_name: str,
    *,
    solver: BruteforceSolver,
    out_dir: Path,
) -> tuple[Tiling, nx.Graph[int], tuple[int, ...], list[str], Path]:
    """Solve, validate, and render one documented brute-force reference case.

    Args:
        tiling_name: Key in ``_CASES``.
        solver: BruteforceSolver instance to use.
        out_dir: Directory where the PNG is saved.

    Returns:
        Tuple of ``(tiling, graph, labels, errors, out_path)``.
    """
    case = _CASES[tiling_name]
    supercell: tuple[int, int] = case["supercell"]
    tiling = registry.get(tiling_name)
    graph = tiling.graph(supercell)
    result = solver.solve(graph, species=("B", "N"), n_b=case["n_b"])

    errors: list[str] = []
    cut_value = result.cut_value if result.cut_value is not None else 0.0
    n_frustrated = result.n_frustrated if result.n_frustrated is not None else 0
    if cut_value != case["expected_cut"]:
        errors.append(
            f"cut_value: got {cut_value}, expected {case['expected_cut']}"
        )
    if n_frustrated != case["expected_frustrated"]:
        errors.append(
            f"n_frustrated: got {n_frustrated}, "
            f"expected {case['expected_frustrated']}"
        )
    n_b_actual = sum(1 for label in result.labels if label == 1)
    if n_b_actual != case["n_b"]:
        errors.append(f"n_b: got {n_b_actual}, expected {case['n_b']}")

    n = graph.number_of_nodes()
    node_size = max(80, int(2400 / n))
    out_path = out_dir / f"exact_{_safe_name(tiling_name)}.png"

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_tiling_assignment(
        ax, tiling, supercell, result.labels,
        node_size=float(node_size), edge_width=2.0, ghost=True,
    )
    ax.set_title(
        f"{tiling_name}  |  B/N exact assignment"
        f"  |  supercell {supercell[0]}x{supercell[1]}"
        f"  |  n_b={case['n_b']}  |  MAX-CUT={int(cut_value)}"
        f"  |  frustrated={n_frustrated}",
        fontsize=10, pad=8,
    )
    ax.legend(
        handles=legend_handles(), loc="lower center",
        bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=9,
    )
    fig.tight_layout(pad=0.2)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)

    return tiling, graph, result.labels, errors, out_path


# ---------------------------------------------------------------------------
# Helpers shared by visualize mode
# ---------------------------------------------------------------------------


def _solve_tiling(
    name: str,
    supercell: tuple[int, int],
) -> tuple[Tiling, tuple[int, ...], float, int]:
    """Return ``(tiling, labels, cut_value, n_frustrated)`` for one tiling.

    Bipartite tilings are solved with :class:`BipartiteAssigner`; frustrated
    tilings use :data:`_VIS_SOLVER`.

    Args:
        name: Registry key for the tiling.
        supercell: ``(na, nb)`` supercell dimensions.

    Returns:
        Tuple of the tiling instance, integer labels, cut value, and the
        number of frustrated bonds.
    """
    tiling = registry.get(name)
    n = len(tiling.vertices) * supercell[0] * supercell[1]
    n_b = n // 2

    if tiling.is_bipartite:
        result = BipartiteAssigner.assign(tiling, n_b=n_b, supercell=supercell)
    else:
        graph = tiling.graph(supercell)
        result = _VIS_SOLVER.solve(graph, ["B", "N"])

    cut_value = result.cut_value if result.cut_value is not None else 0.0
    n_frustrated = result.n_frustrated if result.n_frustrated is not None else 0
    return tiling, result.labels, cut_value, n_frustrated


def _render_vis_single(
    *,
    name: str,
    title: str,
    supercell: tuple[int, int],
    out_dir: Path,
) -> tuple[Tiling, tuple[int, ...], float, int, Path]:
    """Render and save one tiling assignment figure.

    Args:
        name: Registry key for the tiling.
        title: Human-readable label shown as the figure title.
        supercell: ``(na, nb)`` supercell dimensions.
        out_dir: Directory where the PNG is saved.

    Returns:
        Tuple of ``(tiling, labels, cut_value, n_frustrated, out_path)``.
    """
    tiling, labels, cut_value, n_frustrated = _solve_tiling(name, supercell)
    n = len(labels)
    node_size = max(80, int(2400 / n))

    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    draw_tiling_assignment(
        ax, tiling, supercell, labels,
        node_size=float(node_size), edge_width=2.0, ghost=True,
    )
    ax.set_title(
        f"{title}  |  {supercell[0]}×{supercell[1]} supercell"
        f"  |  n={n}  |  MAX-CUT={int(cut_value)}"
        f"  |  frustrated={n_frustrated}",
        fontsize=9, pad=8,
    )
    ax.legend(
        handles=legend_handles(), loc="lower center",
        bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=9,
    )
    fig.tight_layout(pad=0.2)

    safe = name.replace(".", "_")
    out_path = out_dir / f"tiling_{safe}.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return tiling, labels, cut_value, n_frustrated, out_path


# ---------------------------------------------------------------------------
# Mode runners
# ---------------------------------------------------------------------------


def run_geometry(out_dir: Path) -> bool:
    """Verify geometry of all 11 Archimedean tilings and render PNGs.

    Saves one PNG per tiling and an overview grid to *out_dir*.

    Args:
        out_dir: Directory where output images are written.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    all_keys = sorted(_EXPECTED.keys())
    all_errors: dict[str, list[str]] = {}
    all_coords: dict[str, int | None] = {}
    overall_pass = True

    print(f"\n{'─' * 72}")
    print(f"  Verifying {len(all_keys)} Archimedean tilings")
    print(f"{'─' * 72}\n")

    for key in all_keys:
        try:
            tiling = registry.get(key, bond_length=_EXPECTED[key][3])
        except KeyError as exc:
            print(f"  [SKIP] {key}: not found in registry ({exc})")
            overall_pass = False
            continue

        bond = _EXPECTED[key][3]
        sc_vis = _VIS_SC[key]
        sc_bp = _BPCHECK_SC[key]
        g_vis = tiling.graph(sc_vis)
        g_bp = tiling.graph(sc_bp)

        errors, coord = _run_geometry_checks(key, tiling, g_vis, g_bp)
        all_errors[key] = errors
        all_coords[key] = coord

        safe = (
            key
            .replace("⁴", "4").replace("³", "3")
            .replace("²", "2").replace(".", "_")
        )
        out_path = out_dir / f"{safe}.png"
        fig, ax = plt.subplots(figsize=(6, 6))
        _render_pbc_tiling(
            ax, tiling, sc_vis, bond, color=_COLORS[tiling.is_bipartite]
        )
        fig.tight_layout(pad=0.2)
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        if errors:
            overall_pass = False
        icon = "✓" if not errors else "✗"
        status = "PASS" if not errors else "FAIL"
        print(
            f"  {icon} [{status}]  {key:<14}  "
            f"n_prim={tiling.n_vertices}  coord={coord if coord is not None else '?'}"
            f"  bip={tiling.is_bipartite}  |  {out_path.name}"
        )
        for err in errors:
            print(f"          ERROR: {err}")

    # Overview grid
    n_cols = 4
    n_rows = (len(all_keys) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))

    for idx, key in enumerate(all_keys):
        ax = axes.flat[idx]
        try:
            tiling = registry.get(key, bond_length=_EXPECTED[key][3])
            bond = _EXPECTED[key][3]
            _render_pbc_tiling(
                ax, tiling, _VIS_SC[key], bond,
                color=_COLORS[tiling.is_bipartite],
                atom_size=22, bond_width=1.0, cell_lw=1.1,
            )
            errs = all_errors.get(key, [])
            z = all_coords.get(key)
            ax.set_title(
                f"{key}  z={z if z is not None else '?'}\n"
                f"{'✓ PASS' if not errs else '✗ FAIL: ' + errs[0][:40]}",
                fontsize=8,
                color="black" if not errs else "red",
            )
        except Exception as exc:  # noqa: BLE001
            ax.text(0.5, 0.5, f"ERROR\n{exc}", ha="center", va="center",
                    transform=ax.transAxes, color="red", fontsize=8)
            ax.set_title(key, fontsize=9)
            ax.set_aspect("equal")
            ax.axis("off")

    for ax in axes.flat[len(all_keys):]:
        ax.set_visible(False)

    fig.legend(
        handles=[
            mpatches.Patch(color=_COLORS[True], label="Bipartite"),
            mpatches.Patch(color=_COLORS[False], label="Frustrated (non-bipartite)"),
        ],
        loc="lower right", fontsize=10, framealpha=0.9,
    )
    fig.suptitle("All 11 Archimedean Tilings — Geometry Verification",
                 fontsize=14, y=1.005)
    fig.tight_layout()
    overview_path = out_dir / "overview_geometry.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'─' * 72}")
    print(f"  Result: {'ALL PASS ✓' if overall_pass else 'SOME FAILURES ✗'}")
    print(f"  Images → {out_dir}/")
    print(f"  Overview → {overview_path.name}")
    print(f"{'─' * 72}\n")

    return overall_pass


def run_exact_solver(out_dir: Path) -> bool:
    """Verify BruteforceSolver against hardcoded reference cases.

    Saves one PNG per case and an overview grid to *out_dir*.

    Args:
        out_dir: Directory where output images are written.

    Returns:
        ``True`` if all checks pass, ``False`` otherwise.
    """
    solver = BruteforceSolver()
    rendered_cases: list[_RenderedCase] = []
    overall_pass = True

    print(f"\n{'-' * 72}")
    print(f"  Verifying {len(_CASES)} BruteforceSolver reference cases")
    print(f"{'-' * 72}\n")

    for tiling_name in _CASES:
        tiling, graph, labels, errors, out_path = _run_exact_case(
            tiling_name, solver=solver, out_dir=out_dir
        )
        rendered_cases.append((tiling_name, tiling, graph, labels, errors))

        if errors:
            overall_pass = False

        case = _CASES[tiling_name]
        status = "PASS" if not errors else "FAIL"
        icon = "OK" if not errors else "XX"
        print(
            f"  {icon} [{status}]  {tiling_name:<20}  "
            f"n={graph.number_of_nodes():<2}  "
            f"n_b={case['n_b']:<2}  "
            f"MAX-CUT={int(case['expected_cut']):<2}  "
            f"frustrated={case['expected_frustrated']:<2}  |  {out_path.name}"
        )
        for error in errors:
            print(f"          ERROR: {error}")

    # Overview grid
    n_cols = 2
    n_rows = (len(rendered_cases) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 6 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()

    for index, (tiling_name, tiling, graph, labels, errors) in enumerate(
        rendered_cases
    ):
        case = _CASES[tiling_name]
        ax = axes_array[index]
        n = graph.number_of_nodes()
        draw_tiling_assignment(
            ax, tiling, case["supercell"], labels,
            node_size=float(max(30, int(800 / n))),
            edge_width=1.2,
            ghost=False,
        )
        title_color = "black" if not errors else "red"
        status_str = "PASS" if not errors else f"FAIL: {errors[0]}"
        ax.set_title(
            f"{tiling_name}  {case['supercell'][0]}x{case['supercell'][1]}\n"
            f"MAX-CUT={int(case['expected_cut'])}, "
            f"frustrated={case['expected_frustrated']}\n{status_str}",
            fontsize=9,
            color=title_color,
        )

    for ax in axes_array[len(rendered_cases):]:
        ax.set_visible(False)

    fig.legend(
        handles=legend_handles(),
        loc="lower right",
        fontsize=10,
        frameon=False,
    )
    fig.suptitle("BruteforceSolver Verification", fontsize=14, y=1.005)
    fig.tight_layout()

    overview_path = out_dir / "overview_exact_solver.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n{'-' * 72}")
    print(f"  Result: {'ALL PASS' if overall_pass else 'SOME FAILURES'}")
    print(f"  Images -> {out_dir}/")
    print(f"  Overview -> {overview_path.name}")
    print(f"{'-' * 72}\n")

    return overall_pass


def run_visualize(out_dir: Path) -> None:
    """Solve all 11 tilings and render PNG assignments.

    Saves one PNG per tiling and an overview grid to *out_dir*.

    Args:
        out_dir: Directory where output images are written.
    """
    print(f"\n{'─' * 72}")
    print("  Visualising all 11 Archimedean tilings")
    print(f"{'─' * 72}\n")

    all_results: list[_VisResult] = []
    for name, title, supercell in _TILINGS:
        tiling, labels, cut_value, n_frustrated, out_path = _render_vis_single(
            name=name, title=title, supercell=supercell, out_dir=out_dir
        )
        n = len(labels)
        all_results.append(
            (name, title, supercell, tiling, labels, cut_value, n_frustrated)
        )
        print(
            f"  {name:<25s}  n={n:<3d}  cut={int(cut_value):<3d}"
            f"  frustrated={n_frustrated:<3d}  →  {out_path.name}"
        )

    # Overview grid
    n_cols = 4
    n_rows = (len(all_results) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5.5 * n_rows))
    axes_flat = np.atleast_1d(axes).ravel()

    for idx, (
        name, title, supercell, tiling, labels, cut_value, n_frustrated
    ) in enumerate(all_results):
        ax = axes_flat[idx]
        n = len(labels)
        draw_tiling_assignment(
            ax, tiling, supercell, labels,
            node_size=float(max(20, int(600 / n))),
            edge_width=1.2,
            ghost=False,
        )
        ax.set_title(
            f"{title}\nn={n}  cut={int(cut_value)}  frust={n_frustrated}",
            fontsize=8,
        )

    for ax in axes_flat[len(all_results):]:
        ax.set_visible(False)

    fig.legend(
        handles=legend_handles(),
        loc="lower right", fontsize=10, frameon=False,
    )
    fig.suptitle(
        "Archimedean Tilings — Optimal B/N Assignment", fontsize=14, y=1.005
    )
    fig.tight_layout()

    overview_path = out_dir / "overview_all_tilings.png"
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\n  Overview → {overview_path}")
    print(f"  Images   → {out_dir}/")
    print(f"{'─' * 72}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        Parsed :class:`argparse.Namespace` with ``mode`` and ``out_dir``.
    """
    parser = argparse.ArgumentParser(
        description="Verification and visualisation for archimono.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Modes:\n"
            "  geometry      Geometry checks + PNG per tiling + overview grid\n"
            "  exact-solver  BruteforceSolver against reference cases + PNGs\n"
            "  visualize     Solve all 11 tilings and render assignment PNGs\n"
            "  all           Run all three modes in sequence (default)\n"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["geometry", "exact-solver", "visualize", "all"],
        default="all",
        help="Verification mode to run (default: all).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("tmp") / "verify",
        help="Output directory for PNG images (default: tmp/verify/).",
    )
    return parser.parse_args()


def main() -> int:
    """Run the selected verification mode(s).

    Reads ``--mode`` and ``--out-dir`` from the command line, creates the
    output directory if needed, then delegates to the appropriate runner(s).

    Returns:
        Process exit code: ``0`` if selected verification modes pass, else ``1``.
    """
    args = _parse_args()
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mode: str = args.mode
    ok = True

    if mode in ("geometry", "all"):
        ok = run_geometry(out_dir) and ok

    if mode in ("exact-solver", "all"):
        ok = run_exact_solver(out_dir) and ok

    if mode in ("visualize", "all"):
        run_visualize(out_dir)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
