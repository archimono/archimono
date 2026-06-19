"""Fast tests for SweepConfig and SWEEP_CONFIGS."""

from __future__ import annotations

from archimono.benchmarking import SWEEP_CONFIGS, SweepConfig


class TestSweepConfigs:
    """SWEEP_CONFIGS contains exactly three well-formed presets."""

    def test_count(self) -> None:
        """SWEEP_CONFIGS has exactly 3 entries."""
        assert len(SWEEP_CONFIGS) == 3

    def test_labels_unique(self) -> None:
        """All entries have unique labels."""
        labels = [c.label for c in SWEEP_CONFIGS]
        assert len(labels) == len(set(labels))

    def test_labels_are_fast_default_thorough(self) -> None:
        """Labels are 'fast', 'default', and 'thorough'."""
        labels = {c.label for c in SWEEP_CONFIGS}
        assert labels == {"fast", "default", "thorough"}

    def test_all_numeric_fields_positive(self) -> None:
        """All integer fields are strictly positive for every entry."""
        for cfg in SWEEP_CONFIGS:
            assert cfg.greedy_n_runs > 0, f"{cfg.label}: greedy_n_runs not positive"
            assert cfg.annealing_n_runs > 0, (
                f"{cfg.label}: annealing_n_runs not positive"
            )
            assert cfg.annealing_steps_per_temperature > 0, (
                f"{cfg.label}: annealing_steps_per_temperature not positive"
            )

    def test_all_numeric_fields_are_ints(self) -> None:
        """All numeric fields are integers, not floats."""
        for cfg in SWEEP_CONFIGS:
            assert isinstance(cfg.greedy_n_runs, int)
            assert isinstance(cfg.annealing_n_runs, int)
            assert isinstance(cfg.annealing_steps_per_temperature, int)

    def test_sweep_config_is_dataclass(self) -> None:
        """SweepConfig is a dataclass with the expected fields."""
        import dataclasses

        fields = {f.name for f in dataclasses.fields(SweepConfig)}
        assert fields == {
            "label",
            "greedy_n_runs",
            "annealing_n_runs",
            "annealing_steps_per_temperature",
        }

    def test_fast_preset_values(self) -> None:
        """'fast' preset has the exact expected values."""
        fast = next(c for c in SWEEP_CONFIGS if c.label == "fast")
        assert fast.greedy_n_runs == 50
        assert fast.annealing_n_runs == 10
        assert fast.annealing_steps_per_temperature == 20

    def test_thorough_preset_values(self) -> None:
        """'thorough' preset has the exact expected values."""
        thorough = next(c for c in SWEEP_CONFIGS if c.label == "thorough")
        assert thorough.greedy_n_runs == 500
        assert thorough.annealing_n_runs == 10
        assert thorough.annealing_steps_per_temperature == 100
