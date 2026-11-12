"""Sphinx configuration for the archimono API reference.

The site is generated entirely from source docstrings via ``autodoc``; it
contains no hand-written scientific narrative. Citations rendered in the
output come directly from the module/function docstrings.
"""

from __future__ import annotations

import sys
from importlib import metadata
from pathlib import Path

# Make the package importable without an editable install (src layout).
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

# -- Project information -----------------------------------------------------
project = "archimono"
author = "archimono contributors"
copyright = "2026, archimono contributors"

try:
    release = metadata.version("archimono")
except metadata.PackageNotFoundError:
    release = "0.1.0"
version = release

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
}

# The project uses Google-style docstrings (see CLAUDE.md).
napoleon_google_docstring = True
napoleon_numpy_docstring = False

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "networkx": ("https://networkx.org/documentation/stable/", None),
}

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- HTML output -------------------------------------------------------------
html_theme = "furo"
html_title = f"archimono {release}"
html_static_path = ["_static"]
