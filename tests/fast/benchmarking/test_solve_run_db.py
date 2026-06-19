"""Fast tests for SolveRun, open_db, insert_run, and SolveRunDB."""

from __future__ import annotations

import sqlite3
from typing import Any

import pytest

from archimono.benchmarking import SolveRun, SolveRunDB, insert_run, open_db


def _make_run(
    *,
    tiling_key: str = "hexagonal",
    solver: str = "GreedySolver",
    run_index: int = 0,
    seed: int = 42,
    n_vertices: int = 12,
    supercell: str = "(2x3)",
    n_edges: int = 18,
    cut_value: float | None = 10.0,
    cut_efficiency: float | None = 10.0 / 18,
    n_frustrated: int | None = 8,
    labels: tuple[int, ...] | None = (0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1),
    runtime_ms: float = 1.5,
    status: str = "ok",
    note: str = "",
    metadata: dict[str, Any] | None = None,
    sweep_label: str | None = None,
) -> SolveRun:
    """Construct a SolveRun with sensible defaults for testing."""
    if metadata is None:
        metadata = {"n_restarts": 1}
    return SolveRun(
        tiling_key=tiling_key,
        solver=solver,
        run_index=run_index,
        seed=seed,
        n_vertices=n_vertices,
        supercell=supercell,
        n_edges=n_edges,
        cut_value=cut_value,
        cut_efficiency=cut_efficiency,
        n_frustrated=n_frustrated,
        labels=labels,
        runtime_ms=runtime_ms,
        status=status,
        note=note,
        metadata=metadata,
        sweep_label=sweep_label,
    )


class TestOpenDb:
    """open_db creates the expected tables in a fresh SQLite database."""

    def test_tables_exist(self) -> None:
        """open_db on :memory: creates graphs, solver_configs, and runs tables."""
        conn = open_db(":memory:")
        tables = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"graphs", "solver_configs", "runs"} <= tables
        conn.close()

    def test_returns_connection(self) -> None:
        """open_db returns a sqlite3.Connection."""
        conn = open_db(":memory:")
        assert isinstance(conn, sqlite3.Connection)
        conn.close()

    def test_idempotent(self) -> None:
        """Calling open_db twice on the same path does not raise."""
        import tempfile
        from pathlib import Path

        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            path = Path(f.name)
        try:
            c1 = open_db(path)
            c1.close()
            c2 = open_db(path)
            c2.close()
        finally:
            path.unlink(missing_ok=True)


class TestInsertRun:
    """insert_run persists SolveRun records and supports roundtrip retrieval."""

    def test_insert_roundtrip(self) -> None:
        """Inserting a run and querying it back yields the same field values."""
        conn = open_db(":memory:")
        run = _make_run()
        result = insert_run(conn, run)
        conn.commit()

        assert result is True

        db = SolveRunDB.__new__(SolveRunDB)
        db._conn = conn

        runs = db.for_tiling("hexagonal")
        assert len(runs) == 1
        r = runs[0]
        assert r.tiling_key == run.tiling_key
        assert r.solver == run.solver
        assert r.run_index == run.run_index
        assert r.seed == run.seed
        assert r.n_vertices == run.n_vertices
        assert r.supercell == run.supercell
        assert r.n_edges == run.n_edges
        assert r.cut_value == pytest.approx(run.cut_value)
        assert r.n_frustrated == run.n_frustrated
        assert r.labels == run.labels
        assert r.runtime_ms == pytest.approx(run.runtime_ms)
        assert r.status == run.status
        assert r.note == run.note
        assert r.metadata == run.metadata
        assert r.sweep_label == run.sweep_label

    def test_insert_returns_true(self) -> None:
        """insert_run returns True when a new row is inserted."""
        conn = open_db(":memory:")
        run = _make_run(seed=1)
        assert insert_run(conn, run) is True

    def test_duplicate_returns_false(self) -> None:
        """Inserting the same run twice returns False on the second call."""
        conn = open_db(":memory:")
        run = _make_run(seed=99)
        first = insert_run(conn, run)
        conn.commit()
        second = insert_run(conn, run)
        conn.commit()
        assert first is True
        assert second is False

    def test_replace_overwrites(self) -> None:
        """insert_run with replace=True returns True on duplicate."""
        conn = open_db(":memory:")
        run = _make_run(seed=7)
        insert_run(conn, run)
        conn.commit()
        result = insert_run(conn, run, replace=True)
        conn.commit()
        assert result is True

    def test_null_labels_roundtrip(self) -> None:
        """A run with labels=None roundtrips correctly."""
        conn = open_db(":memory:")
        run = _make_run(
            labels=None, cut_value=None, cut_efficiency=None, n_frustrated=None
        )
        insert_run(conn, run)
        conn.commit()

        db = SolveRunDB.__new__(SolveRunDB)
        db._conn = conn
        runs = db.for_tiling("hexagonal")
        assert runs[0].labels is None
        assert runs[0].cut_value is None

    def test_sweep_label_roundtrip(self) -> None:
        """sweep_label is persisted and retrieved correctly."""
        conn = open_db(":memory:")
        run = _make_run(sweep_label="fast", seed=101)
        insert_run(conn, run)
        conn.commit()

        db = SolveRunDB.__new__(SolveRunDB)
        db._conn = conn
        runs = db.for_tiling("hexagonal")
        assert runs[0].sweep_label == "fast"


class TestSolverConfigDedup:
    """_get_or_create_solver_config_id deduplicates configs, NULLs included.

    SQLite's UNIQUE constraint treats NULLs as distinct, so the lookup relies on
    ``IS ?`` (NULL-safe equality) to match rows whose annealing columns are NULL.
    These tests pin that behaviour for the greedy (all-NULL annealing params)
    and annealing (all-non-NULL) cases.
    """

    @staticmethod
    def _solver_config_count(conn: sqlite3.Connection) -> int:
        """Return the number of rows in the solver_configs table.

        Args:
            conn: Open database connection.

        Returns:
            Row count of the ``solver_configs`` table.
        """
        return int(conn.execute("SELECT COUNT(*) FROM solver_configs").fetchone()[0])

    def test_null_params_dedup_to_single_config(self) -> None:
        """Two greedy runs (NULL annealing params) share one solver_configs row.

        This is the core NULL-safe-lookup case: the ``IS ?`` query must match the
        existing all-NULL row instead of inserting a duplicate.
        """
        conn = open_db(":memory:")
        insert_run(conn, _make_run(seed=1, metadata={"n_restarts": 1}))
        insert_run(conn, _make_run(seed=2, metadata={"n_restarts": 1}))
        conn.commit()
        assert self._solver_config_count(conn) == 1
        conn.close()

    def test_null_vs_nonnull_params_distinct_configs(self) -> None:
        """A NULL-param run and a fully-specified run get distinct config rows."""
        conn = open_db(":memory:")
        insert_run(conn, _make_run(seed=1, metadata={"n_restarts": 1}))
        insert_run(
            conn,
            _make_run(
                solver="AnnealingSolver",
                seed=2,
                metadata={
                    "n_restarts": 1,
                    "temperature": 2.0,
                    "min_temperature": 0.1,
                    "cooling_rate": 0.99,
                    "n_steps_per_temperature": 50,
                },
            ),
        )
        conn.commit()
        assert self._solver_config_count(conn) == 2
        conn.close()

    def test_partial_null_params_distinct_configs(self) -> None:
        """Configs differing only in a single NULL vs non-NULL column are distinct.

        Exercises ``IS ?`` per-column: one run leaves ``cooling_rate`` NULL while
        an otherwise identical run sets it, so the lookup must not collapse them.
        """
        conn = open_db(":memory:")
        base_meta = {
            "n_restarts": 1,
            "temperature": 2.0,
            "min_temperature": 0.1,
            "n_steps_per_temperature": 50,
        }
        insert_run(
            conn, _make_run(solver="AnnealingSolver", seed=1, metadata=base_meta)
        )
        insert_run(
            conn,
            _make_run(
                solver="AnnealingSolver",
                seed=2,
                metadata={**base_meta, "cooling_rate": 0.99},
            ),
        )
        conn.commit()
        assert self._solver_config_count(conn) == 2
        conn.close()

    def test_identical_annealing_params_dedup(self) -> None:
        """Two annealing runs with identical params share one config row."""
        conn = open_db(":memory:")
        meta = {
            "n_restarts": 1,
            "temperature": 2.0,
            "min_temperature": 0.1,
            "cooling_rate": 0.99,
            "n_steps_per_temperature": 50,
        }
        insert_run(conn, _make_run(solver="AnnealingSolver", seed=1, metadata=meta))
        insert_run(
            conn, _make_run(solver="AnnealingSolver", seed=2, metadata=dict(meta))
        )
        conn.commit()
        assert self._solver_config_count(conn) == 1
        conn.close()

    def test_null_params_roundtrip_to_none(self) -> None:
        """Annealing params absent from metadata roundtrip as None on the SolveRun."""
        conn = open_db(":memory:")
        insert_run(conn, _make_run(seed=1, metadata={"n_restarts": 1}))
        conn.commit()

        db = SolveRunDB.__new__(SolveRunDB)
        db._conn = conn
        run = db.for_tiling("hexagonal")[0]
        assert "temperature" not in run.metadata
        conn.close()


class TestSolveRunDB:
    """SolveRunDB query methods filter and return correct subsets."""

    def _populated_db(self) -> SolveRunDB:
        """Return an in-memory SolveRunDB with a small mixed dataset."""
        conn = open_db(":memory:")
        runs = [
            _make_run(
                tiling_key="hexagonal", solver="GreedySolver", seed=1, run_index=0
            ),
            _make_run(
                tiling_key="hexagonal", solver="GreedySolver", seed=2, run_index=1
            ),
            _make_run(
                tiling_key="kagome", solver="GreedySolver", seed=3, run_index=0
            ),
            _make_run(
                tiling_key="kagome",
                solver="AnnealingSolver",
                seed=4,
                run_index=0,
                metadata={
                    "n_restarts": 1,
                    "temperature": 2.0,
                    "min_temperature": 0.1,
                    "cooling_rate": 0.99,
                    "n_steps_per_temperature": 50,
                },
            ),
            _make_run(
                tiling_key="hexagonal",
                solver="GreedySolver",
                seed=5,
                run_index=0,
                n_vertices=24,
                supercell="(4x3)",
                n_edges=36,
            ),
        ]
        for r in runs:
            insert_run(conn, r)
        conn.commit()
        db = SolveRunDB.__new__(SolveRunDB)
        db._conn = conn
        return db

    def test_all_returns_all(self) -> None:
        """all() returns every inserted run."""
        db = self._populated_db()
        assert len(db.all()) == 5

    def test_for_tiling_filters(self) -> None:
        """for_tiling returns only runs for the specified tiling."""
        db = self._populated_db()
        runs = db.for_tiling("kagome")
        assert len(runs) == 2
        assert all(r.tiling_key == "kagome" for r in runs)

    def test_for_solver_filters(self) -> None:
        """for_solver returns only runs for the specified solver."""
        db = self._populated_db()
        runs = db.for_solver("AnnealingSolver")
        assert len(runs) == 1
        assert runs[0].solver == "AnnealingSolver"

    def test_for_n_vertices_filters(self) -> None:
        """for_n_vertices returns only runs whose graph has that vertex count."""
        db = self._populated_db()
        runs = db.for_n_vertices(24)
        assert len(runs) == 1
        assert runs[0].n_vertices == 24

    def test_for_tiling_solver_filters(self) -> None:
        """for_tiling_solver filters by both tiling and solver."""
        db = self._populated_db()
        runs = db.for_tiling_solver("hexagonal", "GreedySolver")
        assert len(runs) == 3
        assert all(r.tiling_key == "hexagonal" for r in runs)

    def test_for_tiling_solver_n_filters(self) -> None:
        """for_tiling_solver_n filters by tiling, solver, and vertex count."""
        db = self._populated_db()
        runs = db.for_tiling_solver_n("hexagonal", "GreedySolver", 12)
        assert len(runs) == 2

    def test_tilings_property(self) -> None:
        """tilings property returns sorted unique tiling keys."""
        db = self._populated_db()
        assert db.tilings == ["hexagonal", "kagome"]

    def test_solvers_property(self) -> None:
        """solvers property returns sorted unique solver names."""
        db = self._populated_db()
        assert db.solvers == ["AnnealingSolver", "GreedySolver"]

    def test_len(self) -> None:
        """__len__ returns the total run count."""
        db = self._populated_db()
        assert len(db) == 5

    def test_for_supercell_filters(self) -> None:
        """for_supercell returns only runs with the specified supercell."""
        db = self._populated_db()
        runs = db.for_supercell("(4x3)")
        assert len(runs) == 1
        assert runs[0].supercell == "(4x3)"

    def test_for_tiling_n_filters(self) -> None:
        """for_tiling_n returns all solvers for a (tiling, n_vertices) pair."""
        db = self._populated_db()
        runs = db.for_tiling_n("kagome", 12)
        assert len(runs) == 2
        assert all(r.tiling_key == "kagome" for r in runs)

    def test_close(self) -> None:
        """close() does not raise."""
        conn = open_db(":memory:")
        db = SolveRunDB.__new__(SolveRunDB)
        db._conn = conn
        db.close()
