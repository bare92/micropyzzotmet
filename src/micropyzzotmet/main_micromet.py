#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main driver script for the MicroPezzottoMet downscaling workflow.

This module orchestrates the full processing chain, including:
  - project folder initialization,
  - DEM preparation and terrain-derivative computation,
  - ERA5-Land data download (optional),
  - variable-wise downscaling using MicroMet-inspired methods,
  - optional generation of S3M-compatible forcing files.

The script is intended to be executed from the command line with a
JSON configuration file defining the processing domain, time range,
and variables to downscale.

Author
------
rbarella
"""

import json
import rasterio
import xarray as xr
import os
import datetime
import sys
import glob
from .get_era5_land import get_era5
from .utils import *
from .downscaling_variables import *
import time
from joblib import Parallel, delayed

# Temperature
def run_temperature(curr_climate_file, dem_path, working_directory,
                    variables_to_downscale, custom_lapse_rates,
                    dem_nodata, time_chunk):
    """
    Run air-temperature downscaling for a single climate file.

    This is a lightweight wrapper around :func:`downscale_Temperature`
    that defines the appropriate output folder and forwards configuration
    options.

    Parameters
    ----------
    curr_climate_file : str
        Path to the input ERA5/ERA5-Land NetCDF file.
    dem_path : str
        Path to the DEM GeoTIFF.
    working_directory : str
        Root project directory.
    variables_to_downscale : dict
        Dictionary specifying which variables are enabled (from config).
    custom_lapse_rates : dict
        Optional user-defined lapse rates from configuration.
    dem_nodata : float or int
        No-data value used in the DEM.
    time_chunk : int
        Number of timesteps processed per write block.
    """

    output_folder = os.path.join(working_directory, 'outputs', 'Temperature')
    downscale_Temperature(
        dem_path, curr_climate_file, output_folder,
        custom_lapse_rate=custom_lapse_rates.get("Temperature"),
        dem_nodata=dem_nodata,
        time_chunk=time_chunk,
        source_temperature_var="t2m",
        output_temperature_var="t2m",
        output_file_prefix="temperature_downscaled"
    )


def run_temperature_min(curr_climate_file, dem_path, working_directory,
                        variables_to_downscale, custom_lapse_rates,
                        dem_nodata, time_chunk):
    """Run daily minimum air-temperature downscaling for a single climate file."""
    output_folder = os.path.join(working_directory, 'outputs', 'Temperature_min')
    downscale_Temperature(
        dem_path, curr_climate_file, output_folder,
        custom_lapse_rate=custom_lapse_rates.get("Temperature"),
        dem_nodata=dem_nodata,
        time_chunk=time_chunk,
        source_temperature_var="t_min",
        output_temperature_var="t_min",
        output_file_prefix="temperature_min_downscaled"
    )


def run_temperature_max(curr_climate_file, dem_path, working_directory,
                        variables_to_downscale, custom_lapse_rates,
                        dem_nodata, time_chunk):
    """Run daily maximum air-temperature downscaling for a single climate file."""
    output_folder = os.path.join(working_directory, 'outputs', 'Temperature_max')
    downscale_Temperature(
        dem_path, curr_climate_file, output_folder,
        custom_lapse_rate=custom_lapse_rates.get("Temperature"),
        dem_nodata=dem_nodata,
        time_chunk=time_chunk,
        source_temperature_var="t_max",
        output_temperature_var="t_max",
        output_file_prefix="temperature_max_downscaled"
    )

    
# Shortwave Radiation custom that uses era5 input ssrd
def run_shortwave(curr_climate_file, dem_path, working_directory,
                  variables_to_downscale, custom_lapse_rates,
                  dem_nodata, time_chunk):
    """
    Run shortwave radiation downscaling for a single climate file.

    This wrapper calls the custom pvlib-based shortwave downscaling
    routine, which applies topographic corrections using slope and
    aspect.

    Parameters
    ----------
    curr_climate_file : str
        Path to the input ERA5/ERA5-Land NetCDF file.
    dem_path : str
        Path to the DEM GeoTIFF.
    working_directory : str
        Root project directory.
    variables_to_downscale : dict
        Dictionary specifying which variables are enabled.
    custom_lapse_rates : dict
        Unused for shortwave (kept for interface consistency).
    dem_nodata : float or int
        No-data value used in the DEM.
    time_chunk : int
        Number of timesteps processed per write block.
    """

    output_folder = os.path.join(working_directory, 'outputs', 'SW')
    downscale_SW_custom(dem_path, curr_climate_file, output_folder, dem_nodata=dem_nodata, time_chunk=time_chunk)
    
# Relative Humidity
def run_relative_humidity(curr_climate_file, dem_path, working_directory,
                          variables_to_downscale, custom_lapse_rates,
                          dem_nodata, time_chunk):
    """
    Run relative humidity downscaling for a single climate file.

    Relative humidity is computed from downscaled air temperature
    and dew point temperature using the Magnus formulation.

    Parameters
    ----------
    curr_climate_file : str
        Path to the input ERA5/ERA5-Land NetCDF file.
    dem_path : str
        Path to the DEM GeoTIFF.
    working_directory : str
        Root project directory.
    variables_to_downscale : dict
        Dictionary specifying which variables are enabled.
    custom_lapse_rates : dict
        Optional monthly lapse rates for temperature.
    dem_nodata : float or int
        No-data value used in the DEM.
    time_chunk : int
        Number of timesteps processed per write block.
    """

    output_folder = os.path.join(working_directory, 'outputs', 'RH')
    downscale_RH(
        dem_path, curr_climate_file, output_folder,
        custom_lapse_rate=custom_lapse_rates.get("temperature", {}).get("monthly"),
        dem_nodata=dem_nodata,
        time_chunk=time_chunk
    )

# Precipitation
def run_precipitation(curr_climate_file, dem_path, working_directory,
                      variables_to_downscale, custom_lapse_rates,
                      dem_nodata, time_chunk):
    """
    Run precipitation downscaling for a single climate file.

    This wrapper applies the precipitation downscaling routine,
    optionally using user-defined monthly gamma parameters.

    Parameters
    ----------
    curr_climate_file : str
        Path to the input ERA5/ERA5-Land NetCDF file.
    dem_path : str
        Path to the DEM GeoTIFF.
    working_directory : str
        Root project directory.
    variables_to_downscale : dict
        Dictionary specifying which variables are enabled.
    custom_lapse_rates : dict
        Optional monthly precipitation scaling parameters.
    dem_nodata : float or int
        No-data value used in the DEM.
    time_chunk : int
        Number of timesteps processed per write block.
    """

    output_folder = os.path.join(working_directory, 'outputs', 'P')
    downscale_Precipitation(
        dem_path, curr_climate_file, output_folder,
        custom_gamma=custom_lapse_rates.get("precipitation", {}).get("monthly"),
        dem_nodata=dem_nodata,
        time_chunk=time_chunk
    )

# Wind
def run_wind(curr_climate_file, dem_path, working_directory,
             variables_to_downscale, dem_nodata, time_chunk):
    """
    Run wind speed and direction downscaling for a single climate file.

    Wind fields are adjusted using terrain slope and curvature
    to account for topographic acceleration and sheltering effects.

    Parameters
    ----------
    curr_climate_file : str
        Path to the input ERA5/ERA5-Land NetCDF file.
    dem_path : str
        Path to the DEM GeoTIFF.
    working_directory : str
        Root project directory.
    variables_to_downscale : dict
        Dictionary specifying which variables are enabled.
    dem_nodata : float or int
        No-data value used in the DEM.
    time_chunk : int
        Number of timesteps processed per write block.
    """

    output_folder = os.path.join(working_directory, 'outputs', 'Wind')
    downscale_Wind(
        dem_path, curr_climate_file, output_folder,
        slope_weight=0.5,
        dem_nodata=dem_nodata,
        time_chunk=time_chunk
    )

# Longwave Radiation
def run_longwave(curr_climate_file, dem_path, working_directory,
                 variables_to_downscale, custom_lapse_rates,
                 dem_nodata, time_chunk):
    """
    Run longwave radiation downscaling for a single climate file.

    Longwave radiation is adjusted for elevation effects using
    temperature lapse rates and a reference pressure level.

    Parameters
    ----------
    curr_climate_file : str
        Path to the input ERA5/ERA5-Land NetCDF file.
    dem_path : str
        Path to the DEM GeoTIFF.
    working_directory : str
        Root project directory.
    variables_to_downscale : dict
        Dictionary specifying which variables are enabled.
    custom_lapse_rates : dict
        Optional monthly lapse rates for temperature.
    dem_nodata : float or int
        No-data value used in the DEM.
    time_chunk : int
        Number of timesteps processed per write block.
    """

    output_folder = os.path.join(working_directory, 'outputs', 'LW')
    downscale_LW(
        dem_path, curr_climate_file, output_folder,
        z_700=3000,
        custom_lapse_rate=custom_lapse_rates.get("temperature", {}).get("monthly"),
        dem_nodata=dem_nodata,
        time_chunk=time_chunk
    )


def run_micropezzomet(config_path):
    """
    Execute the full MicroPezzottoMet downscaling workflow.

    This function controls the complete processing pipeline:
      1. Load and validate the JSON configuration file.
      2. Create the project folder structure.
      3. Prepare the DEM (download if needed).
      4. Download ERA5-Land data if not already provided.
      5. Compute terrain derivatives (slope, aspect, curvature).
      6. Downscale selected meteorological variables in parallel.
      7. Optionally generate S3M-compatible forcing files.

    All processing options (time range, variables, lapse rates,
    parallelization settings) are read from the configuration file.

    Parameters
    ----------
    config_path : str
        Path to the JSON configuration file.

    Returns
    -------
    None
        Executes the workflow and writes results to disk.

    Raises
    ------
    ValueError
        If configuration options are inconsistent or invalid.
    RuntimeError
        If required input data cannot be located or generated.
    """

    
    dem_nodata = None
    
    config = load_config(config_path)
    dem_nodata = config.get("dem_nodata", None)

    working_directory = config["working_directory"]
    
    create_full_micromet_folder_structure(base_path=working_directory)
    
    start_date = config["start_date"]
    end_date = config["end_date"]
    
    dem_path = config["dem_file"]
    if dem_path == None:
        dem_path = download_and_save_dem_from_config(config)
        
    
    era_path = config["era_file"]
    pat_token = config.get("earthdatahub_pat")
    netrc_machine = config.get("earthdatahub_machine", "earthdatahub.com")
    aggregate_daily = config["aggregate_daily"]
    jobs_downscaling = config["jobs_parallel_downscale"]
    jobs_download = config["jobs_parallel_download"]
    dem_nodata = config.get("dem_nodata", None)
    
    time_chunk = config.get("time_chunk", 24)

    
    if era_path is None:
        print("Downloading ERA5-Land data...")
        aggregate_daily = parse_yes_no_flag(aggregate_daily, "n")

        get_era5(
            start_date=start_date,
            end_date=end_date,
            refrence_area_path=dem_path,
            output_dir=os.path.join(working_directory, 'inputs/climate'),
            PAT=pat_token,
            jobs_download=jobs_download,
            aggregate_daily=aggregate_daily,
            machine=netrc_machine
        )

    compute_slope_aspect(dem_path, working_directory)
    
    compute_topographic_curvature(dem_path, working_directory)

    climate_files = sorted(glob.glob(os.path.join(working_directory, 'inputs/climate', '*.nc')))
    variables_to_downscale = config["variables_to_downscale"]
    custom_lapse_rates = config.get("custom_lapse_rates", {})
    
    # Air Temperature
    if parse_yes_no_flag(variables_to_downscale.get("t_air", "n"), "t_air"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_temperature)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )

    # Air Temperature (daily minimum)
    if parse_yes_no_flag(variables_to_downscale.get("t_air_min", "n"), "t_air_min"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_temperature_min)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )

    # Air Temperature (daily maximum)
    if parse_yes_no_flag(variables_to_downscale.get("t_air_max", "n"), "t_air_max"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_temperature_max)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )

    # Shortwave Radiation
    if parse_yes_no_flag(variables_to_downscale.get("sw_radiation", "n"), "sw_radiation"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_shortwave)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )

    # Relative Humidity
    if parse_yes_no_flag(variables_to_downscale.get("relative_humidity", "n"), "relative_humidity"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_relative_humidity)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )

    # Precipitation
    if parse_yes_no_flag(variables_to_downscale.get("precipitation", "n"), "precipitation"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_precipitation)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )

    # Wind
    if parse_yes_no_flag(variables_to_downscale.get("wind", "n"), "wind"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_wind)(f, dem_path, working_directory, variables_to_downscale, dem_nodata, time_chunk) for f in climate_files
        )

    # Longwave Radiation
    if parse_yes_no_flag(variables_to_downscale.get("lw_radiation", "n"), "lw_radiation"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_longwave)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata, time_chunk) for f in climate_files
        )
        
    # Optionally generate S3M-compatible inputs
    generate_s3m = parse_yes_no_flag(config.get("generate_s3m_input", "n"), "generate_s3m_input")
    if generate_s3m:
        print("\nCreating daily S3M input files in outputs/s3m ...")
        convert_micromet_to_s3m_inputs(
            micromet_output_dir=os.path.join(working_directory, "outputs"),
            output_dir=os.path.join(working_directory, "outputs", "s3m"),
            dem_path=dem_path,
            nodata_value=dem_nodata if dem_nodata is not None else -9999,
            n_jobs= -1 #  -1 for all available cores
        )


if __name__ == "__main__":
    """
    Command-line entry point.

    Usage
    -----
    python run_micromet.py path_to_config.json
    """

    if len(sys.argv) != 2:
        print("Usage: python run_micromet.py path_to_config.json")
    else:
        config_path = sys.argv[1]
        start_time = time.time()

        run_micropezzomet(config_path)

        end_time = time.time()
        elapsed = end_time - start_time
        elapsed_min = int(elapsed // 60)
        elapsed_sec = int(elapsed % 60)

        config = load_config(config_path)
    
        print("\nMicroPezzottoMet run completed.")
        print(f"Time range: {config['start_date']} to {config['end_date']}")
        print(f"Execution time: {elapsed_min} minutes and {elapsed_sec} seconds")

