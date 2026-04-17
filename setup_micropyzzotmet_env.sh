#!/bin/bash

# Environment name
ENV_NAME="microenv"
PYTHON_VERSION="3.10"



# Create the new environment with channels
echo "Creating environment $ENV_NAME with Python $PYTHON_VERSION..."

micromamba create -n $ENV_NAME \
  -c conda-forge \
  python=$PYTHON_VERSION \
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
  h5netcdf \
  spyder \
  pynacl

# Activate environment
echo "Activating environment..."
eval "$(micromamba shell hook -s bash)"
micromamba activate $ENV_NAME

echo "Environment '$ENV_NAME' is ready with Python $PYTHON_VERSION."
