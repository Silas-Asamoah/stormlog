"""Sphinx configuration for GPU Memory Profiler docs."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))


project = "GPU Memory Profiler"
author = "GPU Memory Profiler Team"
copyright = "2026, GPU Memory Profiler Team"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}

napoleon_google_docstring = True
napoleon_numpy_docstring = True

autodoc_mock_imports = [
    "rich",
    "torch",
    "tensorflow",
    "numpy",
    "pandas",
    "scipy",
    "psutil",
    "matplotlib",
    "seaborn",
    "plotly",
    "dash",
    "dash_bootstrap_components",
    "textual",
    "pyfiglet",
    "flask",
]

myst_enable_extensions = ["colon_fence"]
suppress_warnings = [
    "myst.xref_missing",
    "misc.highlighting_failure",
]

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []
html_title = "GPU Memory Profiler Docs"
