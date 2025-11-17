# docs/source/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# '..' goes to docs/, another '..' goes to the project root.
sys.path.insert(0, os.path.abspath("../.."))

release = "1.1.0"


# -- Project information -----------------------------------------------------
project = "NeuPI"
copyright = "2025, Shivvrat Arya"
author = "Shivvrat Arya"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",  # Automatically generate docs from docstrings
    "sphinx.ext.napoleon",  # Support for Google and NumPy style docstrings
    "sphinx.ext.viewcode",  # Add links to highlighted source code
    "sphinx.ext.mathjax",  # Render math via JavaScript
    "myst_parser",  # Parse Markdown files
    "nbsphinx",  # Render Jupyter notebooks
    "sphinx.ext.autosummary",  # Generate auto-summaries for modules
]

# MOCK IMPORTS: This is the crucial addition.
# It tells Sphinx to ignore these modules, which might fail to import in the
# Read the Docs build environment, allowing the documentation to build correctly.
autodoc_mock_imports = [
    "torch",
    "numpy",
    "neupi.training.pm_ssl.io.uai_reader_cython",
    "neupi.discretize.cython_kn.kn_binary_vectors",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- nbsphinx configuration --------------------------------------------------

# Execute notebooks during the Sphinx build process.
nbsphinx_execute = "always"

# Do not stop the build on errors in notebooks.
nbsphinx_allow_errors = True


source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
