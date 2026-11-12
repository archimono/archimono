"""Fast tests for CertificationRecord JSON roundtrip helpers."""

from __future__ import annotations

import pathlib

import pytest

from archimono.certification import (
    CertificationRecord,
    load_case_json,
    load_case_json_dir,
    load_case_json_tree,
    load_case_labels,
    write_case_json,
)


def _make_record(
    *,
    tiling_key: str = "hexagonal",
    tiling_display: str = "6³ (hexagonal)",
    solver: str = "HadlockSolver",
    target_n: int = 12,
    realized_n: int | None = 12,
    supercell: str | None = "(2x3)",
    n_edges: int | None = 18,
    certified_cut_value: float | None = 9.0,
    certified_cut_efficiency: float | None = 0.5,
    runtime_ms: float = 123.4,
    status: str = "certified",
    assignment_index: int = 0,
    n_optimal_assignments: int = 1,
    vertex_config: str = "AB",
    note: str = "",
    metadata: dict | None = None,  # type: ignore[type-arg]
) -> CertificationRecord:
    return CertificationRecord(
        tiling_key=tiling_key,
        tiling_display=tiling_display,
        solver=solver,
        assignment_index=assignment_index,
        n_optimal_assignments=n_optimal_assignments,
        vertex_config=vertex_config,
        target_n=target_n,
        realized_n=realized_n,
        supercell=supercell,
        n_edges=n_edges,
        certified_cut_value=certified_cut_value,
        certified_cut_efficiency=certified_cut_efficiency,
        runtime_ms=runtime_ms,
        status=status,
        note=note,
        metadata=metadata or {},
    )


class TestWriteLoadCaseJson:
    """write_case_json + load_case_json roundtrip."""

    def test_roundtrip_basic(self, tmp_path: pathlib.Path) -> None:
        record = _make_record()
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        loaded = load_case_json(next(tmp_path.glob("*.json")))
        assert loaded.tiling_key == record.tiling_key
        assert loaded.tiling_display == record.tiling_display
        assert loaded.solver == record.solver
        assert loaded.target_n == record.target_n
        assert loaded.realized_n == record.realized_n
        assert loaded.supercell == record.supercell
        assert loaded.certified_cut_value == record.certified_cut_value
        assert loaded.certified_cut_efficiency == record.certified_cut_efficiency
        assert loaded.runtime_ms == pytest.approx(record.runtime_ms)
        assert loaded.status == record.status

    def test_file_is_created(self, tmp_path: pathlib.Path) -> None:
        record = _make_record()
        path = write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        assert path.exists()
        assert path.suffix == ".json"

    def test_out_dir_created_if_missing(self, tmp_path: pathlib.Path) -> None:
        new_dir = tmp_path / "nested" / "dir"
        assert not new_dir.exists()
        record = _make_record()
        write_case_json(out_dir=new_dir, record=record, generated_by="test")
        assert new_dir.exists()

    def test_roundtrip_null_fields(self, tmp_path: pathlib.Path) -> None:
        """Skip record with None fields survives roundtrip."""
        record = _make_record(
            realized_n=None,
            supercell=None,
            n_edges=None,
            certified_cut_value=None,
            certified_cut_efficiency=None,
            status="skip",
        )
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        loaded = load_case_json(next(tmp_path.glob("*.json")))
        assert loaded.realized_n is None
        assert loaded.supercell is None
        assert loaded.certified_cut_value is None
        assert loaded.status == "skip"

    def test_shear_supercell_filename(self, tmp_path: pathlib.Path) -> None:
        record = _make_record(supercell="[2 1; 0 3]")
        path = write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        # Filename must not contain characters invalid on Windows
        assert "[" not in path.name
        assert "]" not in path.name


class TestSidecarRoundtrip:
    """write_case_json writes .npz sidecar; load_case_labels reads it back."""

    def test_sidecar_roundtrip(self, tmp_path: pathlib.Path) -> None:
        labels: list[tuple[int, ...]] = [(0, 1, 0), (1, 0, 1), (0, 0, 1)]
        record = _make_record(
            metadata={"all_labels": [list(lbl) for lbl in labels]},
        )
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        loaded = load_case_json(next(tmp_path.glob("*.json")))
        recovered = load_case_labels(loaded, tmp_path)
        assert recovered == labels

    def test_sidecar_roundtrip_ndarray(self, tmp_path: pathlib.Path) -> None:
        """A uint8 ndarray all_labels writes a sidecar and roundtrips identically."""
        import numpy as np

        rows = np.array([[0, 1, 0], [1, 0, 1], [0, 0, 1]], dtype=np.uint8)
        record = _make_record(metadata={"all_labels": rows})
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        loaded = load_case_json(next(tmp_path.glob("*.json")))
        recovered = load_case_labels(loaded, tmp_path)
        assert recovered == [tuple(int(v) for v in row) for row in rows]

    def test_sidecar_file_exists(self, tmp_path: pathlib.Path) -> None:
        labels = [(0, 1), (1, 0)]
        record = _make_record(
            metadata={"all_labels": [list(lbl) for lbl in labels]},
        )
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        npz_files = list(tmp_path.glob("*.npz"))
        assert len(npz_files) == 1

    def test_all_labels_not_in_json(self, tmp_path: pathlib.Path) -> None:
        """JSON does not contain all_labels when sidecar is written."""
        import json

        labels = [(0, 1), (1, 0)]
        record = _make_record(
            metadata={"all_labels": [list(lbl) for lbl in labels]},
        )
        path = write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert "all_labels" not in payload["record"]["metadata"]

    def test_no_sidecar_without_labels(self, tmp_path: pathlib.Path) -> None:
        record = _make_record()
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        assert list(tmp_path.glob("*.npz")) == []

    def test_empty_all_labels_no_sidecar(self, tmp_path: pathlib.Path) -> None:
        """Empty all_labels=[] does not crash and writes no sidecar."""
        record = _make_record(metadata={"all_labels": []})
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        assert list(tmp_path.glob("*.npz")) == []

    def test_load_case_labels_empty_without_labels(
        self, tmp_path: pathlib.Path
    ) -> None:
        record = _make_record()
        write_case_json(out_dir=tmp_path, record=record, generated_by="test")
        loaded = load_case_json(next(tmp_path.glob("*.json")))
        assert load_case_labels(loaded, tmp_path) == []


class TestLoadCaseJsonDir:
    """load_case_json_dir loads all JSON records from a flat directory."""

    def test_loads_two_records(self, tmp_path: pathlib.Path) -> None:
        rec_a = _make_record(tiling_key="hexagonal", target_n=12)
        rec_b = _make_record(
            tiling_key="square",
            tiling_display="4⁴ (square)",
            target_n=16,
            supercell="(4x4)",
        )
        write_case_json(out_dir=tmp_path, record=rec_a, generated_by="test")
        write_case_json(out_dir=tmp_path, record=rec_b, generated_by="test")
        records = load_case_json_dir(tmp_path)
        assert len(records) == 2

    def test_field_values_preserved(self, tmp_path: pathlib.Path) -> None:
        rec_a = _make_record(tiling_key="hexagonal", target_n=12)
        rec_b = _make_record(
            tiling_key="square",
            tiling_display="4⁴ (square)",
            target_n=16,
            supercell="(4x4)",
        )
        write_case_json(out_dir=tmp_path, record=rec_a, generated_by="test")
        write_case_json(out_dir=tmp_path, record=rec_b, generated_by="test")
        records = load_case_json_dir(tmp_path)
        keys = {r.tiling_key for r in records}
        assert keys == {"hexagonal", "square"}
        target_ns = {r.target_n for r in records}
        assert target_ns == {12, 16}

    def test_empty_dir_returns_empty_list(self, tmp_path: pathlib.Path) -> None:
        assert load_case_json_dir(tmp_path) == []


class TestLoadCaseJsonTree:
    """load_case_json_tree loads all JSON records recursively."""

    def test_loads_records_from_subdirectories(self, tmp_path: pathlib.Path) -> None:
        sub_a = tmp_path / "hexagonal"
        sub_b = tmp_path / "square"
        rec_a = _make_record(tiling_key="hexagonal", target_n=12)
        rec_b = _make_record(
            tiling_key="square",
            tiling_display="4⁴ (square)",
            target_n=16,
            supercell="(4x4)",
        )
        write_case_json(out_dir=sub_a, record=rec_a, generated_by="test")
        write_case_json(out_dir=sub_b, record=rec_b, generated_by="test")
        records = load_case_json_tree(tmp_path)
        assert len(records) == 2

    def test_field_values_from_subdirs(self, tmp_path: pathlib.Path) -> None:
        sub_a = tmp_path / "hexagonal"
        sub_b = tmp_path / "square"
        rec_a = _make_record(tiling_key="hexagonal", target_n=12)
        rec_b = _make_record(
            tiling_key="square",
            tiling_display="4⁴ (square)",
            target_n=16,
            supercell="(4x4)",
        )
        write_case_json(out_dir=sub_a, record=rec_a, generated_by="test")
        write_case_json(out_dir=sub_b, record=rec_b, generated_by="test")
        records = load_case_json_tree(tmp_path)
        keys = {r.tiling_key for r in records}
        assert keys == {"hexagonal", "square"}
