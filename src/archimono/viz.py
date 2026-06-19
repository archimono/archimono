"""Shared visualisation utilities for Archimedean tiling assignments.

Provides the canonical rendering template used by all tiling image output in
this project.  Requires the optional ``viz`` extra (``matplotlib>=3.7``).

Typical usage::

    import matplotlib.pyplot as plt
    from archimono.viz import draw_tiling_assignment, legend_handles

    fig, ax = plt.subplots(figsize=(6, 6))
    draw_tiling_assignment(ax, tiling, supercell=(2, 2), labels=result.labels)
    ax.legend(handles=legend_handles(), loc="lower center",
              bbox_to_anchor=(0.5, -0.04), ncol=4, frameon=False, fontsize=9)
    fig.savefig("out.png", dpi=160, bbox_inches="tight")
"""

from __future__ import annotations

import math

try:
    from matplotlib.axes import Axes
    from matplotlib.lines import Line2D
    from matplotlib.patches import Polygon
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "archimono.viz requires matplotlib. "
        'Install it with: pip install "archimono[viz]"'
    ) from _exc

import numpy as np
from numpy.typing import NDArray

from archimono.tilings.base import Tiling

# ── Colour palette ────────────────────────────────────────────────────────────
#: Blue — B-species atom colour.
COLOR_B: str = "#4C72B0"
#: Orange — N-species atom colour.
COLOR_N: str = "#DD8452"
#: Green — heterogeneous B–N bond (cut edge).
COLOR_CUT: str = "#2CA02C"
#: Red — frustrated homo bond (B–B or N–N).
COLOR_FRUST: str = "#D62728"
#: Light blue — ghost B atom in periodic image cells.
COLOR_GHOST_B: str = "#A8BFD9"
#: Light orange — ghost N atom in periodic image cells.
COLOR_GHOST_N: str = "#EFC2A0"
#: Near-black — supercell parallelogram outline.
COLOR_CELL: str = "#222222"

#: Map from 0/1 label to atom colour.
NODE_COLORS: dict[int, str] = {0: COLOR_B, 1: COLOR_N}
#: Map from 0/1 label to ghost atom colour.
GHOST_COLORS: dict[int, str] = {0: COLOR_GHOST_B, 1: COLOR_GHOST_N}


# ── Geometry helpers ──────────────────────────────────────────────────────────

def supercell_positions(
    tiling: Tiling,
    na: int,
    nb: int,
) -> NDArray[np.float64]:
    """Return Cartesian coordinates of every site in the (na, nb) supercell.

    Ordering matches ``Tiling.graph``: outer loop ia, inner loop ib, innermost
    loop over primitive-cell vertices.

    Args:
        tiling: The tiling whose geometry defines site positions.
        na: Number of primitive cells along the first lattice vector.
        nb: Number of primitive cells along the second lattice vector.

    Returns:
        Float array of shape ``(na * nb * n_v, 2)`` in Ångströms.
    """
    a1 = np.asarray(tiling.lattice_vectors[0], dtype=float)
    a2 = np.asarray(tiling.lattice_vectors[1], dtype=float)
    out: list[NDArray[np.float64]] = []
    for ia in range(na):
        for ib in range(nb):
            for v in tiling.vertices:
                out.append((v[0] + ia) * a1 + (v[1] + ib) * a2)
    return np.array(out, dtype=np.float64)


def iter_pbc_edges(
    tiling: Tiling,
    supercell: tuple[int, int],
) -> list[tuple[int, int, int, int]]:
    """Return de-duplicated ``(u, v, img_a, img_b)`` bond records.

    ``img_a`` and ``img_b`` are the integer supercell-image offsets for
    endpoint *v* relative to *u* in the infinite lattice.  Interior bonds have
    ``img_a = img_b = 0``; bonds that wrap the supercell boundary have
    non-zero offsets.  Each bond appears exactly once.

    Args:
        tiling: The tiling whose primitive-cell edge list is iterated.
        supercell: ``(na, nb)`` supercell dimensions.

    Returns:
        List of ``(u, v, img_a, img_b)`` tuples, one per unique bond.
    """
    na, nb = supercell
    n_v = len(tiling.vertices)
    seen: set[tuple[int, int, int, int]] = set()
    records: list[tuple[int, int, int, int]] = []

    for ia in range(na):
        for ib in range(nb):
            for vi, vj, da, db in tiling.edges:
                u = ((ia % na) * nb + (ib % nb)) * n_v + vi
                ja_inf, jb_inf = ia + da, ib + db
                ja, jb = ja_inf % na, jb_inf % nb
                img_a = (ja_inf - ja) // na
                img_b = (jb_inf - jb) // nb
                v = ((ja % na) * nb + (jb % nb)) * n_v + vj
                key = (u, v, img_a, img_b)
                rev = (v, u, -img_a, -img_b)
                if key in seen or rev in seen:
                    continue
                seen.add(key)
                records.append(key)
    return records


# ── Core rendering primitive ──────────────────────────────────────────────────

def draw_tiling_assignment(
    ax: Axes,
    tiling: Tiling,
    supercell: tuple[int, int],
    labels: tuple[int, ...] | list[int],
    *,
    node_size: float = 200.0,
    edge_width: float = 2.0,
    ghost: bool = True,
) -> None:
    """Draw a tiling assignment onto *ax* using the canonical project style.

    Renders ghost atoms in the eight surrounding image cells, bonds as
    PBC-correct split stubs (each wrapping bond is drawn as two half-stubs,
    one leaving from each side of the cell boundary), central-cell atoms
    coloured by species, and a clean parallelogram supercell outline.

    Labels are encoded as 0/1 integers matching ``AssignmentResult.labels``:
    ``0`` → B (blue), ``1`` → N (orange).

    Args:
        ax: Matplotlib axes to draw on.
        tiling: The tiling providing geometry (lattice vectors, vertices, edges).
        supercell: ``(na, nb)`` supercell dimensions.
        labels: Per-site species labels, one entry per site in supercell order.
            Values must be ``0`` (B) or ``1`` (N).
        node_size: Marker area in points² (passed to ``ax.scatter``).
        edge_width: Bond line width in points.
        ghost: If ``True``, draw faded atoms in the eight surrounding image
            cells to show periodic context.
    """
    na, nb = supercell
    a1 = np.asarray(tiling.lattice_vectors[0], dtype=float)
    a2 = np.asarray(tiling.lattice_vectors[1], dtype=float)
    super_a = na * a1
    super_b = nb * a2

    pos = supercell_positions(tiling, na, nb)
    lbl = list(labels)

    if len(lbl) != len(pos):
        raise ValueError(
            f"labels length {len(lbl)} does not match site count {len(pos)} "
            f"for supercell {supercell} of '{tiling.vertex_config}'."
        )

    # ── view bounds: square, covering the full cell + context ────────────────
    # Compute the bounding box of the four supercell corners, then extend to a
    # square view so that aspect='equal' fills the figure regardless of the
    # cell's aspect ratio.  The square half-extent also determines how many
    # ghost-cell layers to draw.
    margin = 0.55 * float(np.linalg.norm(a1))
    cell_corners = np.array([np.zeros(2), super_a, super_a + super_b, super_b])
    xmin_c, ymin_c = cell_corners.min(axis=0)
    xmax_c, ymax_c = cell_corners.max(axis=0)
    cx = (xmin_c + xmax_c) * 0.5
    cy = (ymin_c + ymax_c) * 0.5
    # Square half-extent: large enough to enclose the full cell on all sides.
    half = max(xmax_c - xmin_c, ymax_c - ymin_c) * 0.5 + margin

    # ── ghost atoms: fill the square view ────────────────────────────────────
    if ghost:
        # Number of periodic images needed in each supercell direction to
        # cover the square view.  +1 ensures partial images at the boundary.
        ghost_a = math.ceil(half / float(np.linalg.norm(super_a))) + 1
        ghost_b = math.ceil(half / float(np.linalg.norm(super_b))) + 1
        ghost_positions: list[NDArray[np.float64]] = []
        ghost_colors: list[str] = []
        for ga in range(-ghost_a, ghost_a + 1):
            for gb in range(-ghost_b, ghost_b + 1):
                if ga == 0 and gb == 0:
                    continue
                shift = ga * super_a + gb * super_b
                ghost_positions.append(pos + shift)
                ghost_colors.extend(GHOST_COLORS[lbl[i]] for i in range(len(lbl)))
        if ghost_positions:
            all_gp = np.concatenate(ghost_positions, axis=0)
            ax.scatter(
                all_gp[:, 0], all_gp[:, 1],
                s=node_size * 0.35,
                c=ghost_colors,
                edgecolors="white", linewidths=0.4, alpha=0.50, zorder=2,
            )

    # ── bonds: PBC split-stub rendering ──────────────────────────────────────
    for u, v, img_a, img_b in iter_pbc_edges(tiling, supercell):
        pu = pos[u]
        pv = pos[v] + img_a * super_a + img_b * super_b
        is_cut = lbl[u] != lbl[v]
        color = COLOR_CUT if is_cut else COLOR_FRUST
        ls: str | tuple[int, tuple[int, int]] = "solid" if is_cut else (0, (3, 2))
        ax.plot(
            [pu[0], pv[0]], [pu[1], pv[1]],
            color=color, linewidth=edge_width, linestyle=ls,
            zorder=3, solid_capstyle="round",
        )
        if (img_a, img_b) != (0, 0):
            # Mirror stub on the opposite boundary edge
            pu2 = pos[u] - img_a * super_a - img_b * super_b
            ax.plot(
                [pu2[0], pos[v][0]], [pu2[1], pos[v][1]],
                color=color, linewidth=edge_width, linestyle=ls,
                zorder=3, solid_capstyle="round",
            )

    # ── central-cell atoms ────────────────────────────────────────────────────
    ax.scatter(
        pos[:, 0], pos[:, 1],
        s=node_size,
        c=[NODE_COLORS[lbl[i]] for i in range(len(lbl))],
        edgecolors="black", linewidths=0.9, zorder=5,
    )

    # ── supercell parallelogram outline ───────────────────────────────────────
    corners = [np.zeros(2), super_a, super_a + super_b, super_b]
    ax.add_patch(
        Polygon(corners, fill=False, edgecolor=COLOR_CELL, linewidth=1.6)
    )

    ax.set_xlim(cx - half, cx + half)
    ax.set_ylim(cy - half, cy + half)
    ax.set_aspect("equal")
    ax.axis("off")


# ── Legend helper ─────────────────────────────────────────────────────────────

def legend_handles(
    species_b: str = "B",
    species_n: str = "N",
) -> list[Line2D]:
    """Return the four standard ``Line2D`` legend handles.

    Args:
        species_b: Label string for the B-like species.
        species_n: Label string for the N-like species.

    Returns:
        List of four handles: B atom, N atom, B–N cut bond, frustrated bond.
    """
    return [
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=COLOR_B, markeredgecolor="black",
            markersize=10, label=species_b,
        ),
        Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=COLOR_N, markeredgecolor="black",
            markersize=10, label=species_n,
        ),
        Line2D([0], [0], color=COLOR_CUT, lw=2, label=f"{species_b}–{species_n} (cut)"),
        Line2D(
            [0], [0], color=COLOR_FRUST, lw=2, ls=(0, (3, 2)),
            label=f"frustrated ({species_b}–{species_b} / {species_n}–{species_n})",
        ),
    ]
