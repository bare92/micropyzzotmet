#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:08:03 2025

@author: rbarella
"""
import json
import rasterio
import xarray as xr
import os
import datetime
import sys
import glob
from get_era5_land import get_era5
from utils import *
from downscaling_variables import *
import time
from joblib import Parallel, delayed

# Temperature
def run_temperature(curr_climate_file, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata):
    output_folder = os.path.join(working_directory, 'outputs', 'Temperature')
    downscale_Temperature(
        dem_path, curr_climate_file, output_folder,
        custom_lapse_rates.get("temperature", {}).get("monthly"),
        calibrate_lapse_rate=calibrate_lapse_rate,
        dem_nodata=dem_nodata
    )

# Shortwave Radiation
def run_shortwave(curr_climate_file, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata):
    output_folder = os.path.join(working_directory, 'outputs', 'SW')
    downscale_SW_original(
        dem_path, curr_climate_file, output_folder,
        z_700=3000, S0=1370.0,
        custom_lapse_rate=custom_lapse_rates.get("temperature", {}).get("monthly"),
        calibrate_lapse_rate=calibrate_lapse_rate,
        dem_nodata=dem_nodata
    )

# Relative Humidity
def run_relative_humidity(curr_climate_file, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata):
    output_folder = os.path.join(working_directory, 'outputs', 'RH')
    downscale_RH(
        dem_path, curr_climate_file, output_folder,
        custom_lapse_rate=custom_lapse_rates.get("temperature", {}).get("monthly"),
        calibrate_lapse_rate=calibrate_lapse_rate,
        dem_nodata=dem_nodata
    )

# Precipitation
def run_precipitation(curr_climate_file, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata):
    output_folder = os.path.join(working_directory, 'outputs', 'P')
    downscale_Precipitation(
        dem_path, curr_climate_file, output_folder,
        custom_gamma=custom_lapse_rates.get("precipitation", {}).get("monthly"),
        dem_nodata=dem_nodata
    )

# Wind
def run_wind(curr_climate_file, dem_path, working_directory, variables_to_downscale, dem_nodata):
    output_folder = os.path.join(working_directory, 'outputs', 'Wind')
    downscale_Wind(
        dem_path, curr_climate_file, output_folder,
        slope_weight=0.5,
        dem_nodata=dem_nodata
    )

# Longwave Radiation
def run_longwave(curr_climate_file, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata):
    output_folder = os.path.join(working_directory, 'outputs', 'LW')
    downscale_LW(
        dem_path, curr_climate_file, output_folder,
        z_700=3000,
        custom_lapse_rate=custom_lapse_rates.get("temperature", {}).get("monthly"),
        calibrate_lapse_rate=calibrate_lapse_rate,
        dem_nodata=dem_nodata
    )


def run_micropezzomet(config_path):
    dem_nodata = None
    
    config = load_config(config_path)
    dem_nodata = config.get("dem_nodata", None)

    working_directory = config["working_directory"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    dem_path = config["dem_file"]
    era_path = config["era_file"]
    pat_token = config["earthdatahub_pat"]
    aggregate_daily = config["aggregate_daily"]
    jobs_downscaling = config["jobs_parallel_downscale"]
    jobs_download = config["jobs_parallel_download"]
    dem_nodata = config.get("dem_nodata", None)
    calibrate_lapse_rate = parse_yes_no_flag(config["auto_calibrate_lapse_rate"], "n")
    

    create_full_micromet_folder_structure(base_path=working_directory)

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
            aggregate_daily=aggregate_daily
        )

    compute_slope_aspect(dem_path, working_directory)
    
    compute_topographic_curvature(dem_path, working_directory)

    climate_files = sorted(glob.glob(os.path.join(working_directory, 'inputs/climate', '*.nc')))
    variables_to_downscale = config["variables_to_downscale"]
    custom_lapse_rates = config.get("custom_lapse_rates", {})
    
    # Air Temperature
    if parse_yes_no_flag(variables_to_downscale.get("t_air", "n"), "t_air"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_temperature)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata) for f in climate_files
        )

    # Shortwave Radiation
    if parse_yes_no_flag(variables_to_downscale.get("sw_radiation", "n"), "sw_radiation"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_shortwave)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata) for f in climate_files
        )

    # Relative Humidity
    if parse_yes_no_flag(variables_to_downscale.get("relative_humidity", "n"), "relative_humidity"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_relative_humidity)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata) for f in climate_files
        )

    # Precipitation
    if parse_yes_no_flag(variables_to_downscale.get("precipitation", "n"), "precipitation"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_precipitation)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, dem_nodata) for f in climate_files
        )

    # Wind
    if parse_yes_no_flag(variables_to_downscale.get("wind", "n"), "wind"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_wind)(f, dem_path, working_directory, variables_to_downscale, dem_nodata) for f in climate_files
        )

    # Longwave Radiation
    if parse_yes_no_flag(variables_to_downscale.get("lw_radiation", "n"), "lw_radiation"):
        Parallel(n_jobs=jobs_downscaling)(
            delayed(run_longwave)(f, dem_path, working_directory, variables_to_downscale, custom_lapse_rates, calibrate_lapse_rate, dem_nodata) for f in climate_files
        )


if __name__ == "__main__":
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

