"""Fast tests for resolve_tilings helper."""

from __future__ import annotations

import pytest

from archimono.certification import TILINGS, resolve_tilings


class TestResolveTilings:
    """resolve_tilings expands keys to (registry_key, display) pairs."""

    def test_all_returns_eleven(self) -> None:
        """['all'] returns all 11 tilings."""
        result = resolve_tilings(["all"])
        assert len(result) == 11

    def test_all_matches_tilings_constant(self) -> None:
        """['all'] returns exactly TILINGS."""
        result = resolve_tilings(["all"])
        assert result == TILINGS

    def test_frustrated_returns_seven(self) -> None:
        """['frustrated'] returns the 7 frustrated tilings."""
        result = resolve_tilings(["frustrated"])
        assert len(result) == 7

    def test_frustrated_subset_of_all(self) -> None:
        """All frustrated tilings are in TILINGS."""
        frustrated = resolve_tilings(["frustrated"])
        all_keys = {k for k, _ in TILINGS}
        for key, _ in frustrated:
            assert key in all_keys

    def test_specific_key(self) -> None:
        """Single known key returns a one-element list."""
        result = resolve_tilings(["hexagonal"])
        assert len(result) == 1
        assert result[0][0] == "hexagonal"

    def test_specific_key_is_case_insensitive(self) -> None:
        """Single known key accepts mixed case."""
        result = resolve_tilings(["Hexagonal"])
        assert len(result) == 1
        assert result[0][0] == "hexagonal"

    def test_multiple_specific_keys(self) -> None:
        """Multiple known keys return matching subset."""
        result = resolve_tilings(["hexagonal", "square"])
        assert len(result) == 2
        keys = [k for k, _ in result]
        assert "hexagonal" in keys
        assert "square" in keys

    def test_invalid_key_raises_value_error(self) -> None:
        """Unknown key raises ValueError."""
        with pytest.raises(ValueError, match="unknown"):
            resolve_tilings(["nonexistent_tiling"])

    def test_none_returns_all(self) -> None:
        """None returns all tilings (same as ['all'])."""
        result = resolve_tilings(None)
        assert result == TILINGS

    def test_custom_catalogue(self) -> None:
        """Custom catalogue is used instead of TILINGS."""
        custom = [("a", "A"), ("b", "B")]
        result = resolve_tilings(["all"], catalogue=custom)
        assert result == custom

    def test_display_strings_preserved(self) -> None:
        """Display strings from TILINGS are preserved in output."""
        result = resolve_tilings(["hexagonal"])
        assert result[0][1] == "6³ (hexagonal)"

    def test_special_tokens_are_case_insensitive(self) -> None:
        """Special selector tokens accept mixed case."""
        result = resolve_tilings(["Frustrated"])
        assert len(result) == 7

    def test_case_variants_are_deduplicated(self) -> None:
        """Case variants of the same key only produce one result."""
        result = resolve_tilings(["Hexagonal", "hexagonal"])
        assert result == [("hexagonal", "6³ (hexagonal)")]
