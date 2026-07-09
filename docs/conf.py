# Configuration file for the Sphinx documentation builder.
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

# Add the project root so autodoc can import the packages
sys.path.insert(0, os.path.abspath(".."))

# ---------------------------------------------------------------------------
# Project information
# ---------------------------------------------------------------------------
project = "HealthRisk AI"
copyright = "2024, HealthRisk AI Team"
author = "HealthRisk AI Team"
release = "0.1.0"
version = "0.1"

# ---------------------------------------------------------------------------
# General configuration
# ---------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",        # Auto-generate docs from docstrings
    "sphinx.ext.autosummary",    # Summary tables for modules/classes
    "sphinx.ext.napoleon",       # NumPy / Google docstring styles
    "sphinx.ext.viewcode",       # Add [source] links to generated docs
    "sphinx.ext.intersphinx",    # Cross-references to external projects
    "sphinx.ext.todo",           # TODO directives
    "sphinx.ext.coverage",       # Coverage checker
    "sphinx.ext.mathjax",        # Math rendering
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# ---------------------------------------------------------------------------
# Autodoc settings
# ---------------------------------------------------------------------------
autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": "__weakref__",
}

autosummary_generate = True
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True

# ---------------------------------------------------------------------------
# Intersphinx mapping
# ---------------------------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# ---------------------------------------------------------------------------
# HTML output — Furo theme
# ---------------------------------------------------------------------------
html_theme = "furo"
html_static_path = ["_static"]

html_theme_options = {
    "sidebar_hide_name": False,
    "navigation_with_keys": True,
    "light_css_variables": {
        "color-brand-primary": "#1A73E8",
        "color-brand-content": "#1A73E8",
    },
    "dark_css_variables": {
        "color-brand-primary": "#4EA8DE",
        "color-brand-content": "#4EA8DE",
    },
}

html_title = "HealthRisk AI Documentation"
html_short_title = "HealthRisk AI"
