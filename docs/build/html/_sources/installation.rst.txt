Installation
============

This page explains how to install MicroPyzzotMet and create a fully
functional environment using the provided setup script. This is the
recommended method because it installs all required scientific libraries
and ensures compatibility with GDAL, rasterio, xarray, and other core
dependencies.

Requirements
------------

- Python 3.10+
- A working Anaconda or Miniconda installation
- Git

Clone the repository
--------------------

Clone the project from GitHub and enter the folder::

    git clone https://github.com/bare92/micropyzzotmet.git
    cd micropyzzotmet

Environment setup (recommended)
-------------------------------

MicroPyzzotMet provides a setup script that creates a complete conda
environment named ``microenv`` with all required dependencies.

From the project root, run::

    chmod +x setup_micropyzzotmet_env.sh
    ./setup_micropyzzotmet_env.sh

The script performs the following steps:

1. Removes any existing ``microenv`` environment.
2. Configures the conda-forge and defaults channels.
3. Creates a fresh conda environment with Python 3.10.
4. Installs all required scientific libraries (see list below).
5. Optionally installs the Spyder IDE.

Environment details
-------------------

The setup script installs the following Python libraries:

**Core scientific stack**

- ``numpy`` – numerical computation  
- ``pandas`` – tabular data handling  
- ``xarray`` – labeled multi-dimensional data (NetCDF, Zarr)  
- ``scipy`` – scientific utilities  

**Geospatial stack**

- ``gdal`` – raster I/O, projections, file drivers  
- ``rasterio`` – modern raster reading/writing  
- ``rioxarray`` – CRS-aware xarray extension  
- ``pyproj`` – projections and geospatial transforms  
- ``affine`` – grid coordinate transforms  

**Cloud, chunking, and file formats**

- ``zarr`` – chunked storage  
- ``fsspec`` – filesystem abstraction  
- ``s3fs`` – access to S3 and object storage  
- ``netCDF4`` – NetCDF support  
- ``h5netcdf`` – HDF5/NetCDF backend  

**Parallel and distributed computing**

- ``dask``  
- ``distributed``  
- ``joblib``  

**Model-specific utilities**

- ``pvlib`` – solar radiation and atmospheric modeling  
- ``matplotlib`` – plotting  
- ``tqdm`` – progress bars  

**Optional**

- ``spyder`` – IDE installed only if requested in the script  

Activating the environment
--------------------------

If the setup script instructs you to activate the environment, use::

    conda activate microenv

You must activate this environment each time before running MicroPyzzotMet.




