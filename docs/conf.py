# docs/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# Add the project root to the Python path. This allows Sphinx to find your
# 'neupi' module and automatically generate documentation from its docstrings.
# '..' goes up from 'docs/' to the project root.
sys.path.insert(0, os.path.abspath(".."))


# -- Project information -----------------------------------------------------
project = "NeuPI"
copyright = "2025, Shivvrat Arya"
author = "Shivvrat Arya"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Core library to generate documentation from docstrings.
    "sphinx.ext.autosummary",  # Create summary tables.
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings.
    "sphinx.ext.viewcode",  # Add links to highlighted source code.
    "myst_parser",  # To parse Markdown files like your README.
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# This allows you to use markdown files in your documentation
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
