# Configuration file for the Sphinx documentation builder.

from __future__ import annotations

import os
import sys
from pathlib import Path
import types

# -------------------------------------------------------------------------
# PATH: make the project importable
# -------------------------------------------------------------------------
# docs/source/conf.py -> docs/source -> docs -> <repo root>
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))  # so "import micropyzzotmet" works if it's a package
sys.path.insert(0, str(ROOT))  # keep simple + explicit

# If your modules are NOT inside a package folder but live at repo root,
# this still makes them importable (e.g. "import downscaling_variables").
# If they live in micropyzzotmet/, ROOT on sys.path is correct.

sys.stderr.write("### CONF.PY LOADED ###\n")
sys.stderr.write(f"### ROOT = {ROOT} ###\n")

# -------------------------------------------------------------------------
# PROJECT INFO
# -------------------------------------------------------------------------
project = "MicroPyzzotMet"
author = "Riccardo Barella"
copyright = "2025, Riccardo Barella"
release = "0.1.0"

# -------------------------------------------------------------------------
# GENERAL CONFIG
# -------------------------------------------------------------------------
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc.typehints",
]

autosummary_generate = True
autodoc_member_order = "bysource"
napoleon_google_docstring = True
napoleon_numpy_docstring = True

templates_path = ["_templates"]
exclude_patterns: list[str] = []

# -------------------------------------------------------------------------
# IMPORTANT: mock heavy scientific stack *as packages* so submodule imports work
# -------------------------------------------------------------------------
def _ensure_pkg(name: str) -> types.ModuleType:
    """
    Create a fake importable *package* module (has __path__) so that
    'import pkg.subpkg' does not crash.
    """
    if name in sys.modules:
        return sys.modules[name]  # type: ignore[return-value]

    mod = types.ModuleType(name)
    mod.__file__ = f"<mocked {name}>"
    mod.__path__ = []  # makes it behave like a package
    sys.modules[name] = mod
    return mod

def _mock_tree(fullname: str) -> None:
    """
    Ensure all parents exist as packages, and fullname exists as a module.
    Example: fullname="numpy.linalg"
      - ensures "numpy" is a package
      - ensures "numpy.linalg" module exists
    """
    parts = fullname.split(".")
    for i in range(1, len(parts) + 1):
        name = ".".join(parts[:i])
        _ensure_pkg(name)

# These are the imports that commonly break autodoc builds (RTD/minimal env),
# OR can crash if binary wheels mismatch (your local "_ARRAY_API" error).
MOCK_MODULES = [
    # numpy + submodules used by your code
    "numpy",
    "numpy.core",
    "numpy.core.multiarray",
    "numpy.linalg",

    # scipy + submodules used by your code
    "scipy",
    "scipy.stats",
    "scipy.ndimage",
    "scipy.interpolate",

    # geospatial stack
    "rasterio",
    "rasterio.warp",
    "rasterio.transform",
    "rasterio.crs",
    "rioxarray",
    "pyproj",
    "affine",

    # I/O / science stack
    "xarray",
    "pandas",
    "netCDF4",
    "pvlib",

    # plotting (you import pyplot)
    "matplotlib",
    "matplotlib.pyplot",

    # utilities
    "joblib",
    "tqdm",
]

# Create package-like mocks (safe for submodule imports)
for m in MOCK_MODULES:
    _mock_tree(m)

# Also tell autodoc to mock them (double safety)
autodoc_mock_imports = MOCK_MODULES

# -------------------------------------------------------------------------
# HTML OUTPUT
# -------------------------------------------------------------------------
html_theme = "sphinx_rtd_theme"

