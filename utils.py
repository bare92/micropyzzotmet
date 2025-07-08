#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:20:57 2025

@author: rbarella
"""

import json
import rasterio
import xarray as xr
import os
import numpy as np
import pandas as pd
from scipy.ndimage import convolve


def parse_yes_no_flag(value, var_name=""):
    """
    Converts 'y'/'n' string flags to boolean.
    
    Parameters:
        value (str): The input string, expected to be 'y' or 'n'.
        var_name (str): Optional variable name for clearer error messages.
        
    Returns:
        bool: True if 'y', False if 'n'.
        
    Raises:
        ValueError: If value is not 'y' or 'n'.
    """
    if value == "y":
        return True
    elif value == "n":
        return False
    else:
        raise ValueError(f"Invalid value for '{var_name}': {value}. Expected 'y' or 'n'.")


def create_full_micromet_folder_structure(base_path="."):
    folders = [
        "inputs/climate",
        "inputs/dem",
        "outputs"
    ]

    for folder in folders:
        path = os.path.join(base_path, folder)
        os.makedirs(path, exist_ok=True)

    print("Micromet folder structure created successfully.")


def load_config(config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_dem(dem_path):
    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
        dem_transform = src.transform
    return dem_data, dem_meta, dem_transform

def load_era_data(era_path, variables, start_date=None, end_date=None):
    era_ds = xr.open_dataset(era_path)

    # Optionally select variables and time range
    era_ds = era_ds[variables]
    if start_date and end_date and "time" in era_ds.dims:
        era_ds = era_ds.sel(time=slice(start_date, end_date))

    return era_ds


def compute_slope_aspect(dem_path, working_directory):
    """
    Compute slope and aspect from a DEM using gdaldem and save results to <working_directory>/input/dem.

    Parameters:
        dem_path (str): Path to the input DEM file.
        working_directory (str): Path to the working directory.
    
    Returns:
        slope_path (str), aspect_path (str): Paths to the generated output files.
    """
    output_dir = os.path.join(working_directory, 'inputs', 'dem')
    os.makedirs(output_dir, exist_ok=True)

    # Define output file paths
    slope_path = os.path.join(output_dir, 'slope.tif')
    aspect_path = os.path.join(output_dir, 'aspect.tif')

    # Build and run gdaldem commands
    slope_cmd = f'gdaldem slope "{dem_path}" "{slope_path}" -of GTiff'
    aspect_cmd = f'gdaldem aspect "{dem_path}" "{aspect_path}" -of GTiff'

    slope_status = os.system(slope_cmd)
    aspect_status = os.system(aspect_cmd)

    if slope_status == 0 and aspect_status == 0:
        print(f"Slope and aspect successfully saved in {output_dir}")
    else:
        print("Error running gdaldem commands")

    return slope_path, aspect_path



def compute_topographic_curvature(dem_path, working_directory, L=1000, dem_nodata=None):
    """
    Compute and save topographic curvature from a DEM using finite differences.

    Parameters:
        dem_path (str): Path to input DEM file
        working_directory (str): Output folder
        L (float): Length scale for curvature smoothing (m)
        dem_nodata (float or int): No-data value in DEM

    Returns:
        curvature_path (str): Path to saved curvature GeoTIFF
    """
    output_dir = os.path.join(working_directory, 'inputs', 'dem')
    os.makedirs(output_dir, exist_ok=True)
    curvature_path = os.path.join(output_dir, 'curvature.tif')

    if os.path.exists(curvature_path):
        print(f"Curvature already exists at {curvature_path}. Skipping.")
        return curvature_path

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        transform = src.transform
        meta = src.meta.copy()

    if dem_nodata is not None:
        dem[dem == dem_nodata] = np.nan

    # Compute grid spacing
    dx = transform.a
    dy = -transform.e
    cell_size = 0.5 * (dx + dy)

    # Define convolution kernels for curvature approximation
    kernel_diag = np.array([[1, 0, 1],
                            [0, -4, 0],
                            [1, 0, 1]], dtype=np.float32) / (np.sqrt(2) * 4 * cell_size)
    
    kernel_cross = np.array([[0, 1, 0],
                              [1, -4, 1],
                              [0, 1, 0]], dtype=np.float32) / (4 * cell_size)

    mask = np.isnan(dem)
    dem_filled = np.where(mask, np.nanmean(dem), dem)  # simple in-fill to avoid convolution artifacts

    c_diag = convolve(dem_filled, kernel_diag, mode='mirror')
    c_cross = convolve(dem_filled, kernel_cross, mode='mirror')
    curvature = c_diag + c_cross

    # Mask invalid areas
    curvature[mask] = np.nan

    # Normalize curvature (optional)
    curve_max = max(0.001, np.nanmax(np.abs(curvature)))
    curvature /= (2.0 * curve_max)

    # Save result
    meta.update(dtype='float32', count=1, nodata=np.nan)
    with rasterio.open(curvature_path, 'w', **meta) as dst:
        dst.write(curvature.astype(np.float32), 1)

    print(f"Curvature saved to {curvature_path}")
    return curvature_path


def write_downscaled_to_netcdf(
    variables_dict,
    time_list,
    dem_shape,
    dem_transform,
    dem_crs,
    out_nc
):
    """
    Save multiple downscaled variables to NetCDF with spatial referencing.

    Parameters:
        variables_dict: dict of {var_name: (data_list, units, description)}
        time_list: list of datetime objects
        dem_shape: shape of the DEM used as reference
        dem_transform: Affine transform of the DEM
        dem_crs: CRS of the DEM
        out_nc: full path to the output NetCDF file
    """

    height, width = dem_shape
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    dataset_vars = {}

    for var_name, (data_list, units, description) in variables_dict.items():
        data_stack = np.concatenate(data_list, axis=0)

        da = xr.DataArray(
            data_stack,
            dims=["time", "y", "x"],
            coords={"time": time_list, "y": y_coords, "x": x_coords},
            attrs={"units": units, "description": description}
        )

        dataset_vars[var_name] = da

    ds_out = xr.Dataset(dataset_vars)
    ds_out = ds_out.rio.write_transform(dem_transform)
    ds_out = ds_out.rio.write_crs(dem_crs)

    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    ds_out.to_netcdf(out_nc)

    print(f"\nSaved NetCDF: {out_nc}")





