# -- Path setup --------------------------------------------------------------

import re
import snapatac2

# -- Project information -----------------------------------------------------

project = 'SnapATAC2'
copyright = '2022, Kai Zhang'
author = 'Kai Zhang'

# The short X.Y version (including .devXXXX, rcX, b1 suffixes if present)
version = re.sub(r'(\d+\.\d+)\.\d+(.*)', r'\1\2', snapatac2.__version__)
version = re.sub(r'(\.dev\d+).*?$', r'\1', version)
# The full version, including alpha/beta/rc tags.
release = snapatac2.__version__
print("%s %s" % (version, release))

# -- General configuration ---------------------------------------------------

suppress_warnings = ['ref.citation']
default_role = 'code'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "nbsphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_plotly_directive",
]

source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

myst_enable_extensions = [
    "amsmath",
    #"colon_fence",
    #"deflist",
    "dollarmath",
    #"fieldlist",
    #"html_admonition",
    #"html_image",
    #"linkify",
    #"replacements",
    #"smartquotes",
    #"strikethrough",
    #"substitution",
    #"tasklist",
]

# Generate the API documentation when building
autosummary_generate = True
autodoc_member_order = 'bysource'
# autodoc_default_flags = ['members']
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_rtype = True  # having a separate entry generally helps readability
napoleon_use_param = True
napoleon_custom_sections = [('Params', 'Parameters')]
todo_include_todos = False

intersphinx_mapping = {
    "cycler": ("https://matplotlib.org/cycler/", None),
    "h5py": ("http://docs.h5py.org/en/stable/", None),
    "ipython": ("https://ipython.readthedocs.io/en/stable/", None),
    "matplotlib": ("https://matplotlib.org/", None),
    "networkx": (
        "https://networkx.github.io/documentation/networkx-1.10/",
        None,
    ),
    "numpy": ("https://docs.scipy.org/doc/numpy/", None),
    "pandas": ("https://pandas.pydata.org/pandas-docs/stable/", None),
    "pytest": ("https://docs.pytest.org/en/latest/", None),
    "python": ("https://docs.python.org/3", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "seaborn": ("https://seaborn.pydata.org/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

smv_branch_whitelist = r'main'  # Include all branches

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


# -- Options for HTML output -------------------------------------------------

html_theme = 'pydata_sphinx_theme'
html_show_sphinx = False
html_static_path = ['_static']
html_css_files = [
    'css/custom.css',
]

if ".dev" in version:
    switcher_version = "dev"
else:
    switcher_version = f"{version}"

html_theme_options = {
    "github_url": "https://github.com/kaizhang/SnapATAC2",
    "navbar_end": ["version-switcher", "theme-switcher", "navbar-icon-links"],

    "switcher": {
      "version_match": switcher_version,
      "json_url": "https://kzhang.org/SnapATAC2/_static/versions.json", 
    },
}