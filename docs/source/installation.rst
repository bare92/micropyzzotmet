Installation
============

This page describes the installation workflow for the current
CLI-based MicroPyzzotMet release.

Requirements
------------

- Python 3.10+
- ``pip``
- Git
- GDAL command-line tools available on your system, especially ``gdaldem``

Clone the repository
--------------------

Clone the project and enter the repository root::

    git clone https://github.com/bare92/micropyzzotmet.git
    cd micropyzzotmet

Recommended installation (venv)
-------------------------------

Create and activate a local virtual environment, then install the package::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .

Verify that the CLI entrypoint is available::

    micropyzzotmet --help

The dependency list in ``pyproject.toml`` is pinned to exact versions for
reproducibility.

Installed Python dependencies
-----------------------------

The project currently depends on:

- ``numpy``
- ``pandas``
- ``xarray``
- ``rasterio``
- ``rioxarray``
- ``pyproj``
- ``scipy``
- ``pvlib``
- ``netCDF4``
- ``joblib``
- ``tqdm``
- ``affine``
- ``matplotlib``
- ``zarr``
- ``fsspec``
- ``s3fs``
- ``dask``

EarthDataHub authentication
---------------------------

DEM and ERA5-Land downloads require EarthDataHub credentials.

You can provide credentials in one of these ways:

1. Set ``earthdatahub_pat`` in your JSON config file.
2. Configure a ``~/.netrc`` entry for ``earthdatahub.com``.

Example ``~/.netrc``::

    machine earthdatahub.com
    login YOUR_EDH_USERNAME
    password YOUR_EDH_PASSWORD

Set secure permissions::

    chmod 600 ~/.netrc

Alternative setup script (conda)
--------------------------------

If you prefer conda, the repository also includes
``setup_micropyzzotmet_env.sh``::

    chmod +x setup_micropyzzotmet_env.sh
    ./setup_micropyzzotmet_env.sh

Then follow the script output to activate that environment before running
``micropyzzotmet``.




