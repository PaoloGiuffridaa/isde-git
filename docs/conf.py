"""Sphinx configuration for the project documentation."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

project = "ISDE Git Exercise"
copyright = "2026, ISDE Git Exercise contributors"
author = "ISDE Git Exercise contributors"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
]

autodoc_typehints = "description"
napoleon_numpy_docstring = True
exclude_patterns = ["build", "_build", "Thumbs.db", ".DS_Store"]

html_theme = "alabaster"
