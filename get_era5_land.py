#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:59:40 2025

@author: rbarella
"""

# get_era5_zarr.py
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 18 15:59:40 2025

@author: rbarella
"""

# get_era5_zarr.py

import xarray as xr
import pandas as pd
import numpy as np
import os
import rasterio
from rasterio.warp import transform_bounds
from joblib import Parallel, delayed

def is_valid_netcdf(file_path):
    try:
        xr.open_dataset(file_path).close()
        return True
    except:
        return False

def lon_to_360(lon):
    return lon if lon >= 0 else lon + 360

def get_tiff_extent_latlon(tiff_path, buffer=0.4):
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        src_crs = src.crs
        if src_crs.to_epsg() != 4326:
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                src_crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top)
        else:
            lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top
    return lat_min - buffer, lat_max + buffer, lon_min - buffer, lon_max + buffer

def aggregate_to_daily(ds):
    mean_vars = ['t2m', 'd2m', 'u10', 'v10', 'sp', 'strd', 'ssrd']
    sum_vars = ['tp']

    agg_dict = {}
    for var in ds.data_vars:
        if var in mean_vars:
            agg_dict[var] = ds[var].resample(valid_time="1D").mean()
        elif var in sum_vars:
            agg_dict[var] = ds[var].resample(valid_time="1D").sum()
        else:
            agg_dict[var] = ds[var].resample(valid_time="1D").mean()
    return xr.Dataset(agg_dict)

def process_month(ds, start, output_dir, surface_vars, aggregate_daily=False):
    
    end = (start + pd.offsets.MonthEnd(0)) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    tag = "_daily" if aggregate_daily else ""
    surface_file = os.path.join(output_dir, f"era_{start.strftime('%Y_%m')}{tag}.nc")
    if is_valid_netcdf(surface_file):
        print(f"Skipping {start.strftime('%Y-%m')} - already exists.")
        return

    ds_month = ds.sel(valid_time=slice(start, end))
    ds_surface = ds_month[surface_vars]

    # Convert longitude to [-180, 180]
    
    longitudes = ds_surface['longitude'].values
    longitudes = np.where(longitudes > 180, longitudes - 360, longitudes)
    ds_surface = ds_surface.assign_coords(longitude=("longitude", longitudes))
    ds_surface = ds_surface.sortby('longitude')

    if aggregate_daily:
        ds_surface = aggregate_to_daily(ds_surface)
        
    # Assign coordinate attributes explicitly
    ds_surface.latitude.attrs.update(units='degrees_north', standard_name='latitude', axis='Y')
    ds_surface.longitude.attrs.update(units='degrees_east', standard_name='longitude', axis='X')

    # Assign CRS explicitly with rioxarray
    ds_surface = ds_surface.rio.write_crs("EPSG:4326", inplace=True)


    ds_surface.to_netcdf(surface_file)



def get_era5(start_date, end_date, refrence_area_path, output_dir, PAT, jobs_download = 1, aggregate_daily = False):
    os.makedirs(output_dir, exist_ok=True)
    lat_min, lat_max, lon_min, lon_max = get_tiff_extent_latlon(refrence_area_path)
    lat_slice = slice(lat_max, lat_min)
    lon_slice = slice(lon_to_360(lon_min), lon_to_360(lon_max))
    subset_dict = {"latitude": lat_slice, "longitude": lon_slice}

    variables = ['d2m', 't2m', 'sp', 'ssrd', 'strd', 'tp', 'u10', 'v10']
    url = f"https://edh:{PAT}@data.earthdatahub.destine.eu/era5/reanalysis-era5-land-no-antartica-v0.zarr"
    ds = xr.open_dataset(url, chunks={}, engine="zarr").astype("float32")
    ds = ds.sel(**subset_dict)
    ds = ds.sel(valid_time=slice(start_date, end_date))

    time_ranges = pd.date_range(start=start_date, end=end_date, freq='MS')
    
        
    Parallel(n_jobs=-1)(
        delayed(process_month)(ds, start, output_dir, variables, aggregate_daily=aggregate_daily) for start in time_ranges
    )
    

    print("ERA5 download complete.")
