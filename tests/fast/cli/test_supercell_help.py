"""The ``--supercell`` help table matches what ``Tiling.graph`` accepts."""

from __future__ import annotations

import pytest

from archimono.cli._supercell_help import MIN_DIAGONAL_SUPERCELL
from archimono.tilings import registry


def _builds(key: str, size: tuple[int, int]) -> bool:
    try:
        registry.get(key).graph(size)
    except ValueError:
        return False
    return True


def test_table_covers_every_tiling() -> None:
    all_configs = {registry.get(key).vertex_config for key in registry.available()}
    table_configs = [config for config, _, _ in MIN_DIAGONAL_SUPERCELL]
    assert set(table_configs) == all_configs
    assert len(table_configs) == len(all_configs)


@pytest.mark.parametrize(("config", "key", "size"), MIN_DIAGONAL_SUPERCELL)
def test_listed_size_is_the_smallest_that_builds(
    config: str, key: str, size: tuple[int, int]
) -> None:
    assert registry.get(key).vertex_config == config
    assert _builds(key, size)
    smaller = [
        (a, b)
        for a in range(1, 5)
        for b in range(1, 5)
        if (a * b, (a, b)) < (size[0] * size[1], size)
    ]
    assert [s for s in smaller if _builds(key, s)] == []
