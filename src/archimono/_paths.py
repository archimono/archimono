"""Repository-root resolution shared by CLI and workflow default paths.

Default output locations (``docs/draft/...``, ``tmp/...``) assume an editable
checkout; from an installed wheel this resolves inside ``site-packages`` and
callers should pass explicit paths instead.
"""

from __future__ import annotations

import pathlib

#: Repository root of an editable checkout: two levels above this file
#: (``src/archimono/_paths.py`` -> repo root).
PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parents[2]
