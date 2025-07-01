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
