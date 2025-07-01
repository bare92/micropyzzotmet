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
import json
import datetime
import sys
import glob
from get_era5_land import get_era5
from utils import *
from downscaling_variables import *
import time
from joblib import Parallel, delayed

def run_month(curr_climate_file, dem_path, working_directory):
    
        month = os.path.basename(curr_climate_file).split('_')[2]
        year = os.path.basename(curr_climate_file).split('_')[1]
        
        output_folder_T = os.path.join(working_directory,'outputs', 'Temperature', '_'.join([year, month]))

        downscale_Temperature(dem_path, curr_climate_file, output_folder_T)

def run_micropezzomet(config_path):
    
        
    config = load_config(config_path)
    
    # Step 1: Setup
    working_directory = config["working_directory"]
    start_date = config["start_date"]
    end_date = config["end_date"]
    dem_path = config["dem_file"]
    era_path = config["era_file"]
    pat_token = config["earthdatahub_pat"] 
    time_step = config["time_step"]
    jobs_downscaling = config["jobs_parallel_downscale"]
    jobs_download = config["jobs_parallel_download"]
    
    create_full_micromet_folder_structure(base_path=working_directory)
    
    if era_path == None:
        
        
        print("Downloading ERA5-Land data...")
        
        if time_step == "24H":
            aggregate_daily = True
        else:
            aggregate_daily = False
        
        get_era5(
            start_date=start_date,
            end_date=end_date,
            refrence_area_path=dem_path,
            output_dir=os.path.join(working_directory, 'inputs/climate'),
            PAT=pat_token,
            jobs_download = jobs_download,
            aggregate_daily=aggregate_daily,            
            
        )
    
    climate_files = sorted(glob.glob(os.path.join(working_directory, 'inputs/climate', '*.nc')))
   
    Parallel(n_jobs=-1)(
        delayed(run_month)(f, dem_path, working_directory) for f in climate_files
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
        
        # Load config again just for summary
        config = load_config(config_path)
        print("\nMicroPezzottoMet run completed.")
        print(f"Time range: {config['start_date']} to {config['end_date']}")
        print(f"Execution time: {elapsed_min} minutes and {elapsed_sec} seconds")

