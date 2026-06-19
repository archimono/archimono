"""Fast tests for format_supercell helper."""

from __future__ import annotations

import numpy as np

from archimono.certification import format_supercell


class TestFormatSupercell:
    """format_supercell formats 2x2 HNF matrices correctly."""

    def test_diagonal_2x3(self) -> None:
        """Diagonal matrix (2,0,0,3) -> '(2x3)'."""
        m = np.array([[2, 0], [0, 3]], dtype=np.intp)
        assert format_supercell(m) == "(2x3)"

    def test_shear_2_1_0_3(self) -> None:
        """Shear matrix (2,1,0,3) -> '[2 1; 0 3]'."""
        m = np.array([[2, 1], [0, 3]], dtype=np.intp)
        assert format_supercell(m) == "[2 1; 0 3]"

    def test_identity(self) -> None:
        """Identity matrix -> '(1x1)'."""
        m = np.array([[1, 0], [0, 1]], dtype=np.intp)
        assert format_supercell(m) == "(1x1)"

    def test_diagonal_3x3(self) -> None:
        """Diagonal (3,3) -> '(3x3)'."""
        m = np.array([[3, 0], [0, 3]], dtype=np.intp)
        assert format_supercell(m) == "(3x3)"

    def test_shear_with_zero_off_diagonal(self) -> None:
        """Strictly upper-triangular shear [4 3; 0 2] -> '[4 3; 0 2]'."""
        m = np.array([[4, 3], [0, 2]], dtype=np.intp)
        assert format_supercell(m) == "[4 3; 0 2]"
