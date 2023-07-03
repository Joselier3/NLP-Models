"""Sphinx configuration."""
project = "NLP Models"
author = "Jose Elier Fajardo"
copyright = "2023, Jose Elier Fajardo"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
