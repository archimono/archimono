"""Every ``docs/`` path named in ``src/`` or ``scripts/`` exists in this checkout."""

from __future__ import annotations

import pathlib
import re

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_DOC_PATH = re.compile(r"docs/[A-Za-z0-9_./-]*[A-Za-z0-9_]")
# Output directory that report commands create on demand.
_OUTPUT_DIRS = ("docs/draft",)


def test_docs_paths_named_in_source_exist() -> None:
    missing = []
    for top in ("src", "scripts"):
        for path in sorted((_ROOT / top).rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for match in _DOC_PATH.finditer(text):
                ref = match.group(0)
                if ref.startswith(_OUTPUT_DIRS):
                    continue
                if not (_ROOT / ref).exists():
                    line = text.count("\n", 0, match.start()) + 1
                    missing.append(f"{path.relative_to(_ROOT)}:{line}: {ref}")
    assert missing == []
