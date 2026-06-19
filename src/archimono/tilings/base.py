"""Abstract base class for Archimedean tiling geometry.

Each concrete subclass encodes one of the 11 Archimedean tilings and provides
lattice vectors, fractional vertex coordinates, and a neighbour edge list for
the unit cell.  Supercell expansion is handled here so subclasses only
need to describe the unit-cell geometry.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import NDArray

# Type alias for the supercell parameter: either a (na, nb) tuple or a 2×2
# integer matrix.  The tuple form is the original diagonal supercell; the
# matrix form enables non-rectangular (sheared) supercells.
SupercellLike = Sequence[int] | NDArray[np.intp]


class Tiling(abc.ABC):
    """Geometry-only description of a 2D Archimedean tiling.

    Subclasses must implement :attr:`vertex_config`, :attr:`lattice_vectors`,
    :attr:`vertices`, :attr:`edges`, and :attr:`is_bipartite`.

    All coordinates are in fractional units relative to :attr:`lattice_vectors`.
    Edges are expressed as pairs of vertex indices within the unit cell,
    plus a lattice offset ``(da, db)`` that locates the second vertex when it
    lives in a neighbouring cell.

    Args:
        bond_length: Edge length in arbitrary units.  All lattice vectors and
            Cartesian node positions scale linearly with this value.  Default
            ``1.0`` (dimensionless unit bond).  Pass e.g. ``1.42`` for
            graphene/h-BN (Å) or any other physical or simulation unit.

    Raises:
        ValueError: If *bond_length* is not positive.
    """

    def __init__(self, bond_length: float = 1.0) -> None:
        if bond_length <= 0:
            raise ValueError(
                f"bond_length must be positive; got {bond_length}."
            )
        self._bond_length = float(bond_length)

    @property
    def bond_length(self) -> float:
        """Edge length scale; all lattice vectors are proportional to this."""
        return self._bond_length

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @property
    @abc.abstractmethod
    def vertex_config(self) -> str:
        """Vertex configuration string, e.g. ``'3.6.3.6'``."""

    @property
    @abc.abstractmethod
    def lattice_vectors(self) -> NDArray[np.float64]:
        """Shape ``(2, 2)`` array of primitive lattice vectors (Å)."""

    @property
    @abc.abstractmethod
    def vertices(self) -> NDArray[np.float64]:
        """Shape ``(n, 2)`` fractional coordinates of unit-cell vertices."""

    @property
    @abc.abstractmethod
    def edges(self) -> list[tuple[int, int, int, int]]:
        """Unit-cell edges as ``(i, j, da, db)`` tuples.

        ``i`` and ``j`` are vertex indices in the unit cell.
        ``(da, db)`` is the lattice offset such that the Cartesian position of
        the second vertex is ``vertices[j] + da*a + db*b`` where ``a`` and
        ``b`` are the lattice vectors.
        """

    @property
    @abc.abstractmethod
    def is_bipartite(self) -> bool:
        """``True`` if the tiling graph contains no odd-length cycles."""

    # ------------------------------------------------------------------
    # Derived helpers
    # ------------------------------------------------------------------

    @property
    def n_vertices(self) -> int:
        """Number of vertices in the unit cell."""
        return len(self.vertices)

    @property
    def coordination(self) -> int:
        """Coordination number (edges per vertex, assumed uniform)."""
        counts: dict[int, int] = {}
        for i, j, _, _ in self.edges:
            counts[i] = counts.get(i, 0) + 1
            counts[j] = counts.get(j, 0) + 1
        if not counts:
            return 0
        values = set(counts.values())
        if len(values) != 1:
            raise ValueError(
                f"Tiling '{self.vertex_config}' is not vertex-transitive: "
                f"coordination numbers {values}"
            )
        return next(iter(values))

    def graph(
        self,
        supercell: SupercellLike = (1, 1),
        *,
        pbc: bool = True,
    ) -> nx.Graph[int]:
        """Return the tiling graph for the given supercell.

        Args:
            supercell: Either an ``(na, nb)`` tuple of positive integers for
                diagonal supercells, or a 2×2 integer numpy array (supercell
                matrix) whose columns define the supercell lattice vectors in
                units of the primitive vectors.  The matrix form enables
                non-rectangular (sheared) supercells that can satisfy PBC
                constraints at atom counts unreachable by diagonal cells.
            pbc: If ``True`` (default), periodic boundary conditions are
                applied and cross-boundary edges wrap around via modular
                arithmetic.  If ``False``, edges that would cross the
                supercell boundary are omitted, producing a finite (non-
                toroidal) graph.  Non-PBC graphs from planar tilings are
                themselves planar.

        Returns:
            An undirected :class:`networkx.Graph` whose nodes are integer
            indices and carry the attribute ``pos`` (Cartesian position in Å).
            Edges carry no attributes.

        Raises:
            ValueError: If *supercell* does not have the right shape, contains
                non-positive diagonal entries (tuple form), has non-positive
                determinant (matrix form), or produces a degenerate graph
                (self-loops or bipartiteness loss under PBC).
        """
        # --- Matrix supercell (2D array) path ---
        arr = np.asarray(supercell)
        if arr.ndim == 2:
            if arr.shape != (2, 2):
                raise ValueError(
                    f"supercell matrix must be 2×2; got shape {arr.shape}."
                )
            # Reject non-integer-valued matrices: astype(np.intp) would
            # silently truncate floats like [[2.9, 0], [0, 3]] and build the
            # wrong graph.  Accept integer dtypes and float arrays whose
            # values are exactly integer-valued.
            if not np.issubdtype(arr.dtype, np.integer) and not np.all(
                arr == np.round(arr)
            ):
                raise ValueError(
                    f"supercell matrix must contain integer values; got "
                    f"{arr.tolist()}."
                )
            return self._graph_matrix(arr.astype(np.intp), pbc=pbc)

        # --- Diagonal supercell (na, nb) tuple path (original) ---
        if len(supercell) != 2:
            raise ValueError(
                f"supercell must have exactly two entries (na, nb); got "
                f"{tuple(supercell)}."
            )
        na, nb = int(supercell[0]), int(supercell[1])
        if na < 1 or nb < 1:
            raise ValueError(
                f"supercell entries must be positive integers; got "
                f"({supercell[0]}, {supercell[1]})."
            )
        n_prim = self.n_vertices
        a_vec, b_vec = self.lattice_vectors[0], self.lattice_vectors[1]

        g: nx.Graph[int] = nx.Graph()
        g.graph["tiling_vertex_config"] = self.vertex_config
        g.graph["tiling_is_bipartite"] = self.is_bipartite
        g.graph["supercell"] = (na, nb)
        g.graph["pbc"] = pbc

        # Add nodes with Cartesian positions.
        for ia in range(na):
            for ib in range(nb):
                for v in range(n_prim):
                    node_id = _node_index(ia, ib, v, na, nb, n_prim)
                    frac = self.vertices[v]
                    cart = (frac[0] + ia) * a_vec + (frac[1] + ib) * b_vec
                    g.add_node(node_id, pos=cart)

        # Add edges, applying periodic boundary conditions when pbc=True.
        for ia in range(na):
            for ib in range(nb):
                for i, j, da, db in self.edges:
                    ia2_raw = ia + da
                    ib2_raw = ib + db
                    if not pbc:
                        if not (0 <= ia2_raw < na and 0 <= ib2_raw < nb):
                            continue
                        ia2, ib2 = ia2_raw, ib2_raw
                    else:
                        ia2 = ia2_raw % na
                        ib2 = ib2_raw % nb
                    u = _node_index(ia, ib, i, na, nb, n_prim)
                    v = _node_index(ia2, ib2, j, na, nb, n_prim)
                    if u == v:
                        raise ValueError(
                            f"Supercell ({na}, {nb}) produces a self-loop via "
                            f"edge ({i}, {j}, {da}, {db}) in "
                            f"'{self.vertex_config}'.  Increase the supercell "
                            f"dimension(s) along the direction(s) where this "
                            f"edge folds back on itself.  See "
                            f"docs/reference/tilings.md, "
                            f"§PBC supercell constraints."
                        )
                    g.add_edge(u, v)

        if pbc and self.is_bipartite and not nx.is_bipartite(g):
            raise ValueError(
                f"Supercell ({na}, {nb}) produces a non-bipartite graph for "
                f"declared-bipartite tiling '{self.vertex_config}'.  This is a "
                f"PBC parity artefact — the supercell folds a cross-cell edge "
                f"onto two same-sublattice vertices.  See "
                f"docs/reference/tilings.md, "
                f"§PBC supercell constraints."
            )

        return g

    # ------------------------------------------------------------------
    # Matrix supercell implementation
    # ------------------------------------------------------------------

    def _graph_matrix(
        self,
        s_mat: NDArray[np.intp],
        *,
        pbc: bool = True,
    ) -> nx.Graph[int]:
        """Build the tiling graph for a general 2×2 integer supercell matrix.

        The supercell lattice vectors are ``S @ [a, b]`` where ``a`` and ``b``
        are the primitive lattice vectors.  Lattice points inside the
        fundamental domain are enumerated by finding all integer pairs
        ``(ia, ib)`` whose supercell-fractional coordinates
        ``S⁻¹ @ (ia, ib)`` lie in ``[0, 1)²``.

        Args:
            s_mat: A 2×2 integer array with non-zero determinant.
            pbc: If ``True``, periodic boundary conditions wrap cross-boundary
                edges.  If ``False``, those edges are omitted.

        Returns:
            An undirected :class:`networkx.Graph` following the same
            conventions as the diagonal path in :meth:`graph`.

        Raises:
            ValueError: On non-positive determinant, lattice-point enumeration
                mismatch, self-loops, or bipartiteness loss.
        """
        det = int(s_mat[0, 0]) * int(s_mat[1, 1]) - int(s_mat[0, 1]) * int(s_mat[1, 0])
        if det <= 0:
            raise ValueError(
                f"Supercell matrix {s_mat.tolist()} has non-positive determinant "
                f"({det}); matrix must have positive determinant."
            )
        n_cells = det

        n_prim = self.n_vertices
        a_vec, b_vec = self.lattice_vectors[0], self.lattice_vectors[1]

        # --- Exact integer inverse via adjugate ---
        # For a 2×2 matrix S, S⁻¹ = adj(S) / det(S) where
        # adj([[a,b],[c,d]]) = [[d,-b],[-c,a]].
        # Using adj(S) @ v (integer) and dividing by det avoids all
        # floating-point error in fractional-coordinate computations.
        a0, b0 = int(s_mat[0, 0]), int(s_mat[0, 1])
        c0, d0 = int(s_mat[1, 0]), int(s_mat[1, 1])
        # adj(S): rows are [d0, -b0] and [-c0, a0]
        adj = [[d0, -b0], [-c0, a0]]

        def _frac_exact(px: int, py: int) -> tuple[int, int, int]:
            """Return (num_x, num_y, den).

            S⁻¹@[px,py] = (num_x/den, num_y/den).
            """
            return adj[0][0] * px + adj[0][1] * py, adj[1][0] * px + adj[1][1] * py, det

        # --- Enumerate lattice points inside the supercell ---
        # Search bound: any lattice point (ia, ib) in the fundamental domain
        # satisfies |ia| ≤ Σ|S_ij| and |ib| ≤ Σ|S_ij| because S⁻¹·(ia,ib) ∈
        # [0,1)² implies (ia,ib) is a non-negative combination of S's columns
        # with coefficients in [0,1).
        max_coord = int(np.sum(np.abs(s_mat))) + 1

        cell_points: list[tuple[int, int]] = []
        for ia in range(-max_coord, max_coord):
            for ib in range(-max_coord, max_coord):
                nx_, ny_, nd = _frac_exact(ia, ib)
                # S⁻¹@(ia,ib) ∈ [0,1)²  iff  0 ≤ num < den  (for positive det)
                if 0 <= nx_ < nd and 0 <= ny_ < nd:
                    cell_points.append((ia, ib))

        if len(cell_points) != n_cells:
            raise ValueError(
                f"Expected {n_cells} lattice points in supercell "
                f"{s_mat.tolist()} but found {len(cell_points)}."
            )

        # Deterministic ordering: sort by (ia, ib) so node IDs are stable.
        cell_points.sort()
        cell_to_idx: dict[tuple[int, int], int] = {
            pt: idx for idx, pt in enumerate(cell_points)
        }

        g: nx.Graph[int] = nx.Graph()
        g.graph["tiling_vertex_config"] = self.vertex_config
        g.graph["tiling_is_bipartite"] = self.is_bipartite
        g.graph["supercell"] = s_mat.tolist()
        g.graph["pbc"] = pbc

        # --- Nodes ---
        for idx, (ia, ib) in enumerate(cell_points):
            for v in range(n_prim):
                node_id = idx * n_prim + v
                frac = self.vertices[v]
                cart = (frac[0] + ia) * a_vec + (frac[1] + ib) * b_vec
                g.add_node(node_id, pos=cart)

        # --- Edges ---
        for idx, (ia, ib) in enumerate(cell_points):
            for i, j, da, db in self.edges:
                px, py = ia + da, ib + db
                nx_, ny_, nd = _frac_exact(px, py)

                if not pbc:
                    # Skip edges that would require wrapping.
                    if not (0 <= nx_ < nd and 0 <= ny_ < nd):
                        continue
                    rx, ry = nx_, ny_
                else:
                    rx, ry = nx_ % nd, ny_ % nd

                ia2 = (a0 * rx + b0 * ry) // nd
                ib2 = (c0 * rx + d0 * ry) // nd

                u = idx * n_prim + i
                try:
                    v_idx = cell_to_idx[(ia2, ib2)]
                except KeyError:
                    raise ValueError(
                        f"Supercell matrix {s_mat.tolist()} exact wrap-back "
                        f"produced lattice point ({ia2}, {ib2}) which is not "
                        f"in the enumerated fundamental domain for "
                        f"'{self.vertex_config}'. This is a bug — check "
                        f"_frac_exact or the adjugate/modular reduction."
                    ) from None
                v_node = v_idx * n_prim + j

                if u == v_node:
                    raise ValueError(
                        f"Supercell matrix {s_mat.tolist()} produces a "
                        f"self-loop via edge ({i}, {j}, {da}, {db}) in "
                        f"'{self.vertex_config}'.  Choose a larger supercell "
                        f"matrix.  See docs/reference/tilings.md, "
                        f"§PBC supercell constraints."
                    )
                g.add_edge(u, v_node)

        if pbc and self.is_bipartite and not nx.is_bipartite(g):
            raise ValueError(
                f"Supercell matrix {s_mat.tolist()} produces a non-bipartite "
                f"graph for declared-bipartite tiling '{self.vertex_config}'.  "
                f"This is a PBC parity artefact — the supercell folds a "
                f"cross-cell edge onto two same-sublattice vertices.  See "
                f"docs/reference/tilings.md, "
                f"§PBC supercell constraints."
            )

        return g


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _node_index(ia: int, ib: int, v: int, na: int, nb: int, n_prim: int) -> int:
    """Flatten a (cell_a, cell_b, vertex) triple to a single node index."""
    return (ia * nb + ib) * n_prim + v


def _sublattice_parities(tiling: Tiling) -> tuple[int, int]:
    """Return the sublattice parity shift per primitive lattice step.

    For a bipartite tiling the effective sublattice of vertex ``v`` at cell
    ``(ia, ib)`` is ``(base_color[v] + p_a * ia + p_b * ib) % 2``.  This
    function derives ``(p_a, p_b)`` from the 2-colouring of the ``(2, 2)``
    supercell, which is always a valid supercell for every bipartite tiling.

    Non-bipartite tilings always return ``(0, 0)`` because the parity check
    is not applicable to them.

    Args:
        tiling: Any tiling in the registry.

    Returns:
        A pair ``(p_a, p_b)`` where each value is 0 or 1.
    """
    if not tiling.is_bipartite:
        return (0, 0)
    g = tiling.graph((2, 2))
    coloring: dict[int, int] = nx.bipartite.color(g)
    n_v = tiling.n_vertices
    c_ref = coloring[(0 * 2 + 0) * n_v]
    c_a   = coloring[(1 * 2 + 0) * n_v]
    c_b   = coloring[(0 * 2 + 1) * n_v]
    return (c_a - c_ref) % 2, (c_b - c_ref) % 2


def _matrix_supercell_is_valid(
    tiling: Tiling,
    s_mat: NDArray[np.intp],
    sublattice_parities: tuple[int, int] | None = None,
) -> bool:
    """Return True iff *s_mat* is a valid supercell matrix for *tiling*.

    Analytically equivalent to checking whether ``tiling.graph(s_mat)``
    raises :class:`ValueError`, but without building the graph.  The two
    failure modes are:

    - **Self-loop**: for a same-vertex edge ``(i, i, da, db)``, the
      displacement ``[da, db]`` is in the supercell lattice, i.e.,
      ``S⁻¹ @ [da, db]`` is an integer vector.
    - **Bipartite parity loss**: for a declared-bipartite tiling, at least
      one column of *s_mat* has an odd dot-product with the sublattice parity
      vector ``(p_a, p_b)``, causing a cross-cell edge to connect two
      same-sublattice vertices.

    Args:
        tiling: The tiling to validate against.
        s_mat: A 2×2 integer HNF matrix with positive determinant.
        sublattice_parities: Precomputed ``(p_a, p_b)`` from
            :func:`_sublattice_parities`.  When ``None``, the parities are
            computed on demand.  Pass a precomputed value when calling this
            function in a tight loop over many candidate matrices for the
            same tiling.

    Returns:
        ``True`` if *s_mat* passes both checks; ``False`` if either fails.
    """
    s_inv = np.linalg.inv(s_mat.astype(np.float64))

    # Self-loop check: S⁻¹ @ [da, db] must not be an integer vector for any
    # same-vertex edge (i == j).
    for i, j, da, db in tiling.edges:
        if i == j:
            frac = s_inv @ np.array([da, db], dtype=np.float64)
            if np.allclose(frac, np.round(frac), atol=1e-9):
                return False

    # Bipartite parity check: each column of S must have an even dot-product
    # with the sublattice parity vector (p_a, p_b).
    if tiling.is_bipartite:
        p_a, p_b = (
            sublattice_parities
            if sublattice_parities is not None
            else _sublattice_parities(tiling)
        )
        if p_a != 0 or p_b != 0:
            for col in (s_mat[:, 0], s_mat[:, 1]):
                if (p_a * int(col[0]) + p_b * int(col[1])) % 2 != 0:
                    return False

    return True


def min_valid_supercell_matrix(
    tiling: Tiling,
    target_n: int,
) -> NDArray[np.intp] | None:
    """Find a valid supercell matrix for *tiling* at *target_n* atoms.

    Searches all 2×2 Hermite Normal Form (HNF) matrices whose determinant
    equals ``target_n / n_prim``.  Diagonal matrices (no shear) are tried
    first, sorted by aspect ratio; shear matrices follow, sorted by
    condition number.  Validity is checked analytically via
    :func:`_matrix_supercell_is_valid` without building the graph.  Returns
    the first valid matrix, or ``None`` if no valid supercell exists at the
    requested size.

    This is useful for tilings like 4.8² (truncated square) whose PBC
    constraints prevent all diagonal supercells at certain atom counts;
    a shear matrix can sometimes satisfy the constraints instead.

    Args:
        tiling: The tiling to build a supercell for.
        target_n: Desired number of atoms in the supercell.

    Returns:
        A 2×2 integer numpy array, or ``None`` if *target_n* is not a
        multiple of ``tiling.n_vertices`` or no valid supercell exists.
    """
    candidates = valid_supercell_matrices(tiling, target_n)
    return None if not candidates else candidates[0]


def valid_supercell_matrices(
    tiling: Tiling,
    target_n: int,
) -> list[NDArray[np.intp]]:
    """Return all valid HNF supercell matrices for *tiling* at *target_n* atoms.

    Searches all 2×2 Hermite Normal Form (HNF) matrices whose determinant
    equals ``target_n / n_prim``. Diagonal matrices (no shear) are ordered
    first, sorted by aspect ratio; shear matrices follow, sorted by condition
    number. Every matrix for which :func:`_matrix_supercell_is_valid` returns
    ``True`` is returned.

    Args:
        tiling: The tiling to build supercells for.
        target_n: Desired number of atoms in the supercell.

    Returns:
        A list of valid 2×2 integer numpy arrays. The list is empty if
        *target_n* is not a multiple of ``tiling.n_vertices`` or no valid
        supercell exists.
    """
    n_prim = tiling.n_vertices
    if target_n % n_prim != 0:
        return []
    k = target_n // n_prim  # required |det(S)|

    candidates: list[NDArray[np.intp]] = []
    for a in range(1, k + 1):
        if k % a != 0:
            continue
        d = k // a
        for b in range(d):
            candidates.append(np.array([[a, b], [0, d]], dtype=np.intp))

    def _sort_key(s: NDArray[np.intp]) -> tuple[int, float]:
        is_shear = int(s[0, 1] != 0)
        cond = float(np.linalg.cond(s.astype(np.float64)))
        return (is_shear, cond)

    candidates.sort(key=_sort_key)

    # Precompute sublattice parities once; reused for every candidate matrix.
    parities = _sublattice_parities(tiling)

    valid: list[NDArray[np.intp]] = []
    for s in candidates:
        if _matrix_supercell_is_valid(tiling, s, sublattice_parities=parities):
            valid.append(s)

    return valid
