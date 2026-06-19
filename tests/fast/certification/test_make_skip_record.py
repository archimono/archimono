"""Fast tests for make_skip_record factory."""

from __future__ import annotations

from archimono.certification import CertificationRecord, make_skip_record


class TestMakeSkipRecord:
    """make_skip_record produces valid skip placeholder records."""

    def test_status_is_skip(self) -> None:
        """Returned record has status='skip'."""
        record = make_skip_record(
            tiling_key="hexagonal",
            tiling_display="6³ (hexagonal)",
            solver="HadlockSolver",
            target_n=12,
            vertex_config="AB",
        )
        assert record.status == "skip"

    def test_runtime_ms_is_zero(self) -> None:
        """Returned record has runtime_ms=0.0."""
        record = make_skip_record(
            tiling_key="square",
            tiling_display="4⁴ (square)",
            solver="HadlockSolver",
            target_n=8,
            vertex_config="AB",
        )
        assert record.runtime_ms == 0.0

    def test_nullable_fields_are_none(self) -> None:
        """realized_n, supercell, n_edges, cut_value, cut_efficiency are None."""
        record = make_skip_record(
            tiling_key="triangular",
            tiling_display="3⁶ (triangular)",
            solver="BruteforceSolver",
            target_n=6,
            vertex_config="AB",
        )
        assert record.realized_n is None
        assert record.supercell is None
        assert record.n_edges is None
        assert record.certified_cut_value is None
        assert record.certified_cut_efficiency is None

    def test_fields_set_correctly(self) -> None:
        """tiling_key, tiling_display, solver, target_n, vertex_config are set."""
        record = make_skip_record(
            tiling_key="kagome",
            tiling_display="3.6.3.6 (kagome)",
            solver="HadlockSolver",
            target_n=18,
            vertex_config="AAB",
        )
        assert record.tiling_key == "kagome"
        assert record.tiling_display == "3.6.3.6 (kagome)"
        assert record.solver == "HadlockSolver"
        assert record.target_n == 18
        assert record.vertex_config == "AAB"

    def test_default_note(self) -> None:
        """Default note is 'no valid supercell'."""
        record = make_skip_record(
            tiling_key="hexagonal",
            tiling_display="6³ (hexagonal)",
            solver="HadlockSolver",
            target_n=12,
            vertex_config="AB",
        )
        assert record.note == "no valid supercell"

    def test_custom_note(self) -> None:
        """Custom note is stored."""
        record = make_skip_record(
            tiling_key="hexagonal",
            tiling_display="6³ (hexagonal)",
            solver="HadlockSolver",
            target_n=12,
            vertex_config="AB",
            note="custom reason",
        )
        assert record.note == "custom reason"

    def test_returns_certification_record(self) -> None:
        """Returns a CertificationRecord instance."""
        record = make_skip_record(
            tiling_key="hexagonal",
            tiling_display="6³ (hexagonal)",
            solver="HadlockSolver",
            target_n=12,
            vertex_config="AB",
        )
        assert isinstance(record, CertificationRecord)

    def test_assignment_index_zero(self) -> None:
        """assignment_index defaults to 0."""
        record = make_skip_record(
            tiling_key="hexagonal",
            tiling_display="6³ (hexagonal)",
            solver="HadlockSolver",
            target_n=12,
            vertex_config="AB",
        )
        assert record.assignment_index == 0

    def test_n_optimal_assignments_zero(self) -> None:
        """n_optimal_assignments defaults to 0."""
        record = make_skip_record(
            tiling_key="hexagonal",
            tiling_display="6³ (hexagonal)",
            solver="HadlockSolver",
            target_n=12,
            vertex_config="AB",
        )
        assert record.n_optimal_assignments == 0
