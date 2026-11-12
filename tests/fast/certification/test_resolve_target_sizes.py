"""Fast tests for resolve_target_sizes helper."""

from __future__ import annotations

from archimono.certification import resolve_target_sizes


class TestResolveTargetSizes:
    """resolve_target_sizes returns correct sorted positive sizes."""

    def test_explicit_sizes_pass_through(self) -> None:
        """Explicit list is returned sorted."""
        result = resolve_target_sizes([6, 4, 12], default=[2, 4, 6, 8], max_n=20)
        assert result == [4, 6, 12]

    def test_none_uses_defaults_filtered_by_max_n(self) -> None:
        result = resolve_target_sizes(None, default=[4, 8, 12, 16], max_n=12)
        assert result == [4, 8, 12]

    def test_none_full_defaults_when_max_n_large(self) -> None:
        defaults = [4, 8, 12]
        result = resolve_target_sizes(None, default=defaults, max_n=100)
        assert result == [4, 8, 12]

    def test_non_positive_filtered_from_explicit(self) -> None:
        result = resolve_target_sizes([0, -1, 4, 8], default=[], max_n=20)
        assert result == [4, 8]

    def test_explicit_larger_than_max_n_included(self) -> None:
        """max_n does not filter explicit sizes."""
        result = resolve_target_sizes([20, 40], default=[4, 8], max_n=10)
        assert result == [20, 40]

    def test_empty_explicit_returns_empty(self) -> None:
        """Empty explicit list returns empty (no fallback to defaults)."""
        result = resolve_target_sizes([], default=[4, 8], max_n=20)
        assert result == []

    def test_none_defaults_empty_returns_empty(self) -> None:
        result = resolve_target_sizes(None, default=[], max_n=20)
        assert result == []

    def test_explicit_deduplication(self) -> None:
        result = resolve_target_sizes([4, 4, 8], default=[], max_n=20)
        assert result == [4, 8]

    def test_none_filters_and_deduplicates_defaults(self) -> None:
        result = resolve_target_sizes(
            None,
            default=[0, -4, 4, 4, 8, 12],
            max_n=8,
        )
        assert result == [4, 8]
