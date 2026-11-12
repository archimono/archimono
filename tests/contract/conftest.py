"""Auto-apply 'contract' marker to all tests in this tier."""

from __future__ import annotations

from pathlib import Path

import pytest

_THIS_DIR = Path(__file__).parent


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Add the 'contract' marker to tests in tests/contract/."""
    for item in items:
        if _THIS_DIR in item.path.parents:
            item.add_marker(pytest.mark.contract)
