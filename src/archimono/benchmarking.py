"""SQLite-backed store for solve-run records, and sweep configuration presets.

Defines :class:`SolveRun`, :class:`SweepConfig`, :data:`SWEEP_CONFIGS`, and
provides a fast indexed query interface via :class:`SolveRunDB`.

Typical workflows::

    # Query a subset
    from archimono.benchmarking import SolveRunDB
    db = SolveRunDB("tmp/solve-runs.db")
    runs = db.for_tiling_solver_n("kagome", "GreedySolver", 48)

    # Write new runs
    conn = open_db("tmp/solve-runs.db")
    insert_run(conn, run)
    conn.commit()
"""

from __future__ import annotations

import dataclasses
import json
import sqlite3
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# SolveRun dataclass
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class SolveRun:
    """Outcome of one single-restart solver call on one tiling supercell.

    Attributes:
        tiling_key: Registry key identifying the tiling, e.g. ``"hexagonal"``.
        solver: Solver class name, e.g. ``"GreedySolver"`` or
            ``"AnnealingSolver"``.
        run_index: Zero-based index of this run within the sampling batch.
        seed: RNG seed used for this run (``base_seed + run_index``).
        n_vertices: Actual number of vertices in the constructed graph.
        supercell: Supercell string, e.g. ``"(3x4)"``.
        n_edges: Edge count of the constructed graph.
        cut_value: MAX-CUT value returned by the solver, or ``None`` on
            failure.
        cut_efficiency: ``cut_value / n_edges``, or ``None``.
        n_frustrated: Number of frustrated edges, or ``None`` on failure.
        labels: Per-node binary assignment tuple, or ``None`` on failure.
        runtime_ms: Wall-clock time in milliseconds for the solver call.
        status: ``"ok"`` on success, ``"error"`` on any exception.
        note: Error message or diagnostic note; empty string on success.
        metadata: Raw metadata dict from the solver result.
        sweep_label: Sweep-config label (e.g. ``"fast"``, ``"default"``,
            ``"thorough"``) or ``None`` for single-config runs.
    """

    tiling_key: str
    solver: str
    run_index: int
    seed: int
    n_vertices: int
    supercell: str
    n_edges: int
    cut_value: float | None
    cut_efficiency: float | None
    n_frustrated: int | None
    labels: tuple[int, ...] | None
    runtime_ms: float
    status: str
    note: str
    metadata: dict[str, Any]
    sweep_label: str | None = None


# ---------------------------------------------------------------------------
# SweepConfig dataclass and presets
# ---------------------------------------------------------------------------


@dataclasses.dataclass(slots=True)
class SweepConfig:
    """One named configuration for a sampling sweep.

    Greedy and annealing solvers have different sensitivity axes, so each
    config specifies independent run counts per solver. Greedy quality scales
    with restart count; annealing quality scales with steps per temperature.

    Attributes:
        label: Short name for this preset, e.g. ``"fast"``.
        greedy_n_runs: Number of independent greedy restarts to sample.
        annealing_n_runs: Number of independent annealing runs to sample.
        annealing_steps_per_temperature: Steps per temperature level used by
            the annealing solver in each run.
    """

    label: str
    greedy_n_runs: int
    annealing_n_runs: int
    annealing_steps_per_temperature: int


SWEEP_CONFIGS: list[SweepConfig] = [
    SweepConfig("fast", 50, 10, 20),
    SweepConfig("default", 200, 10, 50),
    SweepConfig("thorough", 500, 10, 100),
]
"""Named sampling presets in increasing thoroughness order.

Each entry controls how many runs are collected and (for annealing) how many
steps are taken per temperature level.
"""

# ---------------------------------------------------------------------------
# DDL
# ---------------------------------------------------------------------------

_CREATE_GRAPHS = """
CREATE TABLE IF NOT EXISTS graphs (
    graph_id    INTEGER NOT NULL,
    tiling_key  TEXT    NOT NULL,
    supercell   TEXT    NOT NULL,
    n_vertices  INTEGER NOT NULL,
    n_edges     INTEGER NOT NULL,
    PRIMARY KEY (graph_id),
    UNIQUE (tiling_key, supercell, n_vertices, n_edges)
)
"""

_CREATE_INDEX_GRAPHS = """
CREATE INDEX IF NOT EXISTS idx_graphs_tiling
    ON graphs (tiling_key)
"""

_CREATE_SOLVER_CONFIGS = """
CREATE TABLE IF NOT EXISTS solver_configs (
    solver_config_id      INTEGER NOT NULL,
    solver                TEXT    NOT NULL,
    n_restarts            INTEGER NOT NULL,
    temperature           REAL,
    min_temperature       REAL,
    cooling_rate          REAL,
    steps_per_temperature INTEGER,
    PRIMARY KEY (solver_config_id),
    UNIQUE (solver, n_restarts, temperature, min_temperature,
            cooling_rate, steps_per_temperature)
)
"""

_CREATE_RUNS = """
CREATE TABLE IF NOT EXISTS runs (
    run_id           INTEGER NOT NULL,
    graph_id         INTEGER NOT NULL REFERENCES graphs (graph_id),
    solver_config_id INTEGER NOT NULL REFERENCES solver_configs (solver_config_id),
    run_index        INTEGER NOT NULL,
    seed             INTEGER NOT NULL,
    sweep_label      TEXT    NOT NULL DEFAULT '',
    cut_value        REAL,
    cut_efficiency   REAL,
    n_frustrated     INTEGER,
    labels           TEXT,
    runtime_ms       REAL    NOT NULL,
    status           TEXT    NOT NULL,
    note             TEXT    NOT NULL,
    metadata         TEXT    NOT NULL,
    PRIMARY KEY (run_id),
    UNIQUE (graph_id, solver_config_id, seed, sweep_label)
)
"""

_CREATE_INDEX_RUNS_GRAPH_SOLVER = """
CREATE INDEX IF NOT EXISTS idx_runs_graph_solver
    ON runs (graph_id, solver_config_id)
"""

_CREATE_INDEX_RUNS_SOLVER_CONFIG = """
CREATE INDEX IF NOT EXISTS idx_runs_solver_config
    ON runs (solver_config_id)
"""

# ---------------------------------------------------------------------------
# DML helpers
# ---------------------------------------------------------------------------

_UPSERT_GRAPH = """
INSERT INTO graphs (tiling_key, supercell, n_vertices, n_edges)
VALUES (?, ?, ?, ?)
ON CONFLICT (tiling_key, supercell, n_vertices, n_edges) DO NOTHING
"""

_SELECT_GRAPH_ID = """
SELECT graph_id FROM graphs
WHERE tiling_key = ?
  AND supercell = ?
  AND n_vertices = ?
  AND n_edges = ?
"""

# NULL-safe lookup: SQLite UNIQUE treats NULLs as distinct, so we use a
# SELECT-first approach instead of INSERT OR IGNORE + SELECT.
_SELECT_SOLVER_CONFIG_ID = """
SELECT solver_config_id FROM solver_configs
WHERE solver = ?
  AND n_restarts = ?
  AND temperature IS ?
  AND min_temperature IS ?
  AND cooling_rate IS ?
  AND steps_per_temperature IS ?
"""

_INSERT_SOLVER_CONFIG = """
INSERT INTO solver_configs
    (solver, n_restarts, temperature, min_temperature,
     cooling_rate, steps_per_temperature)
VALUES (?, ?, ?, ?, ?, ?)
"""

_INSERT_RUN = """
INSERT OR {conflict} INTO runs (
    graph_id, solver_config_id, run_index, seed,
    sweep_label, cut_value, cut_efficiency, n_frustrated, labels,
    runtime_ms, status, note, metadata
) VALUES (
    ?, ?, ?, ?,
    ?, ?, ?, ?, ?,
    ?, ?, ?, ?
)
"""

# Canonical SELECT for all query methods.  Column aliases match SolveRun field
# names so _row_to_run() requires no changes when the schema changes.
_SELECT_ALL = """
SELECT
    g.tiling_key,
    g.supercell,
    g.n_vertices,
    g.n_edges,
    sc.solver,
    sc.n_restarts,
    sc.temperature,
    sc.min_temperature,
    sc.cooling_rate,
    sc.steps_per_temperature,
    r.run_index,
    r.seed,
    r.sweep_label,
    r.cut_value,
    r.cut_efficiency,
    r.n_frustrated,
    r.labels,
    r.runtime_ms,
    r.status,
    r.note,
    r.metadata
FROM runs r
JOIN graphs        g  USING (graph_id)
JOIN solver_configs sc USING (solver_config_id)
"""

_ORDER = " ORDER BY g.tiling_key, g.n_vertices, sc.solver, r.run_index"

# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def open_db(path: str | Path) -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists.

    Args:
        path: File path for the SQLite database.  Pass ``":memory:"`` for an
            in-memory database.

    Returns:
        Open :class:`sqlite3.Connection` with row factory set to
        :attr:`sqlite3.Row`.
    """
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.execute(_CREATE_GRAPHS)
    conn.execute(_CREATE_INDEX_GRAPHS)
    conn.execute(_CREATE_SOLVER_CONFIGS)
    conn.execute(_CREATE_RUNS)
    conn.execute(_CREATE_INDEX_RUNS_GRAPH_SOLVER)
    conn.execute(_CREATE_INDEX_RUNS_SOLVER_CONFIG)
    conn.commit()
    return conn


def _get_or_create_graph_id(conn: sqlite3.Connection, run: SolveRun) -> int:
    """Upsert a graph row and return its ``graph_id``.

    Args:
        conn: Open database connection.
        run: Run whose graph metadata should be registered.

    Returns:
        Integer ``graph_id`` for this run's graph.
    """
    conn.execute(
        _UPSERT_GRAPH,
        (run.tiling_key, run.supercell, run.n_vertices, run.n_edges),
    )
    row = conn.execute(
        _SELECT_GRAPH_ID,
        (run.tiling_key, run.supercell, run.n_vertices, run.n_edges),
    ).fetchone()
    return int(row[0])


def _extract_solver_config(
    run: SolveRun,
) -> tuple[str, int, Any, Any, Any, Any]:
    """Extract solver config fields from a :class:`SolveRun`.

    Args:
        run: Run record to extract solver configuration from.

    Returns:
        Tuple of ``(solver, n_restarts, temperature, min_temperature,
        cooling_rate, steps_per_temperature)`` suitable for DB insertion.
    """
    n_restarts: int = int(run.metadata.get("n_restarts", 1))
    temperature: float | None = run.metadata.get("temperature")
    min_temperature: float | None = run.metadata.get("min_temperature")
    cooling_rate: float | None = run.metadata.get("cooling_rate")
    steps: int | None = run.metadata.get("n_steps_per_temperature")
    return (run.solver, n_restarts, temperature, min_temperature, cooling_rate, steps)


def _get_or_create_solver_config_id(
    conn: sqlite3.Connection, run: SolveRun
) -> int:
    """Return the solver_config_id for this run, inserting a row if needed.

    Uses a SELECT-first strategy because SQLite treats NULL values as distinct
    in UNIQUE constraints, making INSERT OR IGNORE unreliable when any config
    column is NULL.

    Args:
        conn: Open database connection.
        run: Run whose solver configuration should be registered.

    Returns:
        Integer ``solver_config_id`` for this run's solver configuration.
    """
    solver, n_restarts, temperature, min_temperature, cooling_rate, steps = (
        _extract_solver_config(run)
    )
    lookup_params = (
        solver, n_restarts, temperature, min_temperature, cooling_rate, steps
    )
    existing = conn.execute(_SELECT_SOLVER_CONFIG_ID, lookup_params).fetchone()
    if existing is not None:
        return int(existing[0])
    cursor = conn.execute(
        _INSERT_SOLVER_CONFIG,
        (solver, n_restarts, temperature, min_temperature, cooling_rate, steps),
    )
    assert cursor.lastrowid is not None
    return int(cursor.lastrowid)


def insert_run(
    conn: sqlite3.Connection,
    run: SolveRun,
    *,
    replace: bool = False,
    graph_id_cache: dict[tuple[str, str, int, int], int] | None = None,
    solver_config_id_cache: dict[tuple[Any, ...], int] | None = None,
) -> bool:
    """Insert one :class:`SolveRun` into the database.

    Args:
        conn: Open database connection.
        run: Run record to insert.
        replace: If ``True``, use ``INSERT OR REPLACE`` to overwrite any
            existing row with the same primary key.  If ``False`` (default),
            silently ignore duplicates via ``INSERT OR IGNORE``.
        graph_id_cache: Optional caller-owned dict for caching graph IDs
            across repeated calls with the same graph.
        solver_config_id_cache: Optional caller-owned dict for caching
            solver-config IDs across repeated calls with the same config.

    Returns:
        ``True`` if SQLite inserted or replaced a row; ``False`` if an
        existing row caused an ``INSERT OR IGNORE`` no-op.
    """
    conflict = "REPLACE" if replace else "IGNORE"

    graph_key = (run.tiling_key, run.supercell, run.n_vertices, run.n_edges)
    if graph_id_cache is not None and graph_key in graph_id_cache:
        graph_id = graph_id_cache[graph_key]
    else:
        graph_id = _get_or_create_graph_id(conn, run)
        if graph_id_cache is not None:
            graph_id_cache[graph_key] = graph_id

    config_key = _extract_solver_config(run)
    if solver_config_id_cache is not None and config_key in solver_config_id_cache:
        solver_config_id = solver_config_id_cache[config_key]
    else:
        solver_config_id = _get_or_create_solver_config_id(conn, run)
        if solver_config_id_cache is not None:
            solver_config_id_cache[config_key] = solver_config_id

    cursor = conn.execute(
        _INSERT_RUN.format(conflict=conflict),
        (
            graph_id,
            solver_config_id,
            run.run_index,
            run.seed,
            run.sweep_label or "",
            run.cut_value,
            run.cut_efficiency,
            run.n_frustrated,
            json.dumps(list(run.labels)) if run.labels is not None else None,
            run.runtime_ms,
            run.status,
            run.note,
            json.dumps(run.metadata),
        ),
    )
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Row deserialization
# ---------------------------------------------------------------------------


def _row_to_run(row: sqlite3.Row) -> SolveRun:
    """Deserialize a database row back to a :class:`SolveRun`.

    Args:
        row: A row returned by a query using :data:`_SELECT_ALL`.

    Returns:
        Populated :class:`SolveRun` instance.
    """
    raw_labels = row["labels"]
    labels: tuple[int, ...] | None = (
        tuple(int(v) for v in json.loads(raw_labels))
        if raw_labels is not None
        else None
    )
    metadata: dict[str, Any] = json.loads(row["metadata"])
    return SolveRun(
        tiling_key=row["tiling_key"],
        solver=row["solver"],
        run_index=row["run_index"],
        seed=row["seed"],
        sweep_label=row["sweep_label"] or None,
        n_vertices=row["n_vertices"],
        supercell=row["supercell"],
        n_edges=row["n_edges"],
        cut_value=row["cut_value"],
        cut_efficiency=row["cut_efficiency"],
        n_frustrated=row["n_frustrated"],
        labels=labels,
        runtime_ms=row["runtime_ms"],
        status=row["status"],
        note=row["note"],
        metadata=metadata,
    )


# ---------------------------------------------------------------------------
# SolveRunDB
# ---------------------------------------------------------------------------


class SolveRunDB:
    """SQLite-backed collection of solve-run records.

    Args:
        db_path: Path to the SQLite database created by :func:`open_db`.

    Example::

        db = SolveRunDB("tmp/solve-runs.db")
        runs = db.for_tiling_solver_n("kagome", "GreedySolver", 48)
    """

    def __init__(self, db_path: str | Path) -> None:
        self._conn = open_db(db_path)

    def _query(self, sql: str, params: tuple[object, ...] = ()) -> list[SolveRun]:
        """Execute *sql* with *params* and deserialize all result rows.

        Args:
            sql: SQL query string; must be based on :data:`_SELECT_ALL`.
            params: Positional bind parameters.

        Returns:
            List of :class:`SolveRun` instances.
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [_row_to_run(r) for r in rows]

    # ------------------------------------------------------------------
    # Catalog properties
    # ------------------------------------------------------------------

    @property
    def tilings(self) -> list[str]:
        """Sorted unique tiling keys present in the database.

        Returns:
            Sorted list of ``tiling_key`` strings.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT tiling_key FROM graphs ORDER BY tiling_key"
        ).fetchall()
        return [r[0] for r in rows]

    @property
    def solvers(self) -> list[str]:
        """Sorted unique solver names present in the database.

        Returns:
            Sorted list of solver name strings.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT solver FROM solver_configs ORDER BY solver"
        ).fetchall()
        return [r[0] for r in rows]

    @property
    def n_vertices_values(self) -> list[int]:
        """Sorted unique n_vertices values present in the database.

        Returns:
            Sorted list of ``n_vertices`` integers.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT n_vertices FROM graphs ORDER BY n_vertices"
        ).fetchall()
        return [r[0] for r in rows]

    @property
    def supercells(self) -> list[str]:
        """Sorted unique supercell strings present in the database.

        Returns:
            Sorted list of supercell strings.
        """
        rows = self._conn.execute(
            "SELECT DISTINCT supercell FROM graphs ORDER BY supercell"
        ).fetchall()
        return [r[0] for r in rows]

    # ------------------------------------------------------------------
    # Query methods
    # ------------------------------------------------------------------

    def all(self) -> list[SolveRun]:
        """Return all runs in the database.

        Returns:
            All :class:`SolveRun` records, sorted by tiling key, n_vertices,
            solver, and run index.
        """
        return self._query(_SELECT_ALL + _ORDER)

    def for_tiling(self, tiling_key: str) -> list[SolveRun]:
        """Return all runs for a specific tiling.

        Args:
            tiling_key: Registry key, e.g. ``"hexagonal"``.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL + " WHERE g.tiling_key=?" + _ORDER,
            (tiling_key,),
        )

    def for_solver(self, solver: str) -> list[SolveRun]:
        """Return all runs for a specific solver.

        Args:
            solver: Solver class name, e.g. ``"GreedySolver"``.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL + " WHERE sc.solver=?" + _ORDER,
            (solver,),
        )

    def for_n_vertices(self, n_vertices: int) -> list[SolveRun]:
        """Return all runs for a specific graph size.

        Args:
            n_vertices: Vertex count of the graph.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL + " WHERE g.n_vertices=?" + _ORDER,
            (n_vertices,),
        )

    def for_tiling_solver(self, tiling_key: str, solver: str) -> list[SolveRun]:
        """Return all runs for a specific (tiling, solver) pair.

        Args:
            tiling_key: Registry key, e.g. ``"hexagonal"``.
            solver: Solver class name, e.g. ``"AnnealingSolver"``.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL + " WHERE g.tiling_key=? AND sc.solver=?" + _ORDER,
            (tiling_key, solver),
        )

    def for_tiling_solver_n(
        self, tiling_key: str, solver: str, n_vertices: int
    ) -> list[SolveRun]:
        """Return all runs for a specific (tiling, solver, n_vertices) triple.

        Args:
            tiling_key: Registry key, e.g. ``"hexagonal"``.
            solver: Solver class name, e.g. ``"GreedySolver"``.
            n_vertices: Vertex count of the graph.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL
            + " WHERE g.tiling_key=? AND sc.solver=? AND g.n_vertices=?"
            + _ORDER,
            (tiling_key, solver, n_vertices),
        )

    def for_supercell(self, supercell: str) -> list[SolveRun]:
        """Return all runs for a specific supercell string.

        Args:
            supercell: Supercell string, e.g. ``"(2x3)"``.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL + " WHERE g.supercell=?" + _ORDER,
            (supercell,),
        )

    def for_tiling_solver_n_supercell(
        self,
        tiling_key: str,
        solver: str,
        n_vertices: int,
        supercell: str,
    ) -> list[SolveRun]:
        """Return all runs for a specific (tiling, solver, n_vertices, supercell) quad.

        Args:
            tiling_key: Registry key, e.g. ``"hexagonal"``.
            solver: Solver class name, e.g. ``"GreedySolver"``.
            n_vertices: Vertex count of the graph.
            supercell: Supercell string, e.g. ``"(2x3)"``.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL
            + " WHERE g.tiling_key=? AND sc.solver=?"
            + "   AND g.n_vertices=? AND g.supercell=?"
            + _ORDER,
            (tiling_key, solver, n_vertices, supercell),
        )

    def for_tiling_n(self, tiling_key: str, n_vertices: int) -> list[SolveRun]:
        """Return all runs for a specific (tiling, n_vertices) pair across solvers.

        Args:
            tiling_key: Registry key, e.g. ``"hexagonal"``.
            n_vertices: Vertex count of the graph.

        Returns:
            Filtered list of :class:`SolveRun` records.
        """
        return self._query(
            _SELECT_ALL + " WHERE g.tiling_key=? AND g.n_vertices=?" + _ORDER,
            (tiling_key, n_vertices),
        )

    def __len__(self) -> int:
        """Return the total number of run records in the database."""
        return int(self._conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0])

    def __repr__(self) -> str:
        """Return a concise string representation showing run counts and catalog."""
        n = len(self)
        return (
            f"SolveRunDB({n} runs, "
            f"tilings={self.tilings}, "
            f"solvers={self.solvers})"
        )

    def close(self) -> None:
        """Close the underlying database connection."""
        self._conn.close()
