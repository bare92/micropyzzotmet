#!/bin/bash

# Environment name
ENV_NAME="microenv"
PYTHON_VERSION="3.10"

# Remove old environment if it exists
echo "Removing any existing environment: $ENV_NAME"
conda remove -y -n $ENV_NAME --all

# Configure channels
echo "Configuring conda channels..."
conda config --add channels conda-forge
conda config --add channels defaults
conda config --set channel_priority strict

# Create the new environment
echo "Creating environment $ENV_NAME with Python $PYTHON_VERSION..."
conda create -y -n $ENV_NAME python=$PYTHON_VERSION

# Activate the environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate $ENV_NAME

# Install required packages
echo "Installing required packages..."
conda install -y \
  numpy \
  pandas \
  xarray \
  scipy \
  rasterio \
  rioxarray \
  affine \
  pyproj \
  matplotlib \
  tqdm \
  joblib \
  pvlib \
  gdal \
  zarr \
  fsspec \
  s3fs \
  dask \
  distributed \
  netCDF4 \
  h5netcdf

# Optional: install spyder if you want an IDE in this environment
conda install -y spyder

echo "Environment '$ENV_NAME' is ready with Python $PYTHON_VERSION."

