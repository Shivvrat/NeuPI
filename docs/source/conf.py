# docs/source/conf.py

import os
import sys

# -- Path setup --------------------------------------------------------------
# '..' goes to docs/, another '..' goes to the project root.
sys.path.insert(0, os.path.abspath("../.."))


# -- Project information -----------------------------------------------------
project = "NeuPI"
copyright = "2025, Shivvrat Arya"
author = "Shivvrat Arya"
release = "0.1.0"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "myst_parser",
]

# MOCK IMPORTS: This is the crucial addition.
# It tells Sphinx to ignore these modules, which might fail to import in the
# Read the Docs build environment, allowing the documentation to build correctly.
autodoc_mock_imports = ["torch", "numpy", "neupi.training.pm_ssl.io.uai_reader_cython"]


templates_path = ["_templates"]
exclude_patterns = []

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]
