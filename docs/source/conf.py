import os
import sys

sys.path.insert(0, os.path.abspath('../../'))


# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
project = 'issych'
copyright = '2025, Issaku Kawashima'
author = 'Issaku Kawashima'
release = '0.0.6.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'numpydoc',
    ]

autosummary_generate = True
autodoc_class_signature = 'separated'
numpydoc_show_class_members = True
templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# -- Intersphinx ------------------------------------------------

intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/reference/', None),
    'matplotlib': ('https://matplotlib.org/', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'statsmodels': ('https://www.statsmodels.org/stable/', None),
    'seaborn': ('https://seaborn.pydata.org/', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
    'pingouin': ('https://pingouin-stats.org/', None)
}
