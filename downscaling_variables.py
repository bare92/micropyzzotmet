#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 14:18:55 2025

@author: rbarella
"""

import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS
from tqdm import tqdm
import pandas as pd
from rasterio.warp import reproject, Resampling


def downscale_Temperature(dem_path, curr_climate_file, output_folder_T, custom_lapse_rate = False):
    
    geopotential_path = './auxiliary_data/geopotential3.nc'
    # Lapse rates per hemisphere
    lapse_rate_nohem = np.array([4.4,5.9,7.1,7.8,8.1,8.2,8.1,8.1,7.7,6.8,5.5,4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1,8.1,7.7,6.8,5.5,4.7,4.4,5.9,7.1,7.8,8.1,8.2]) / 1000.0
    
    os.makedirs(output_folder_T, exist_ok=True)
    
    
    # Read DEM
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    # Open NetCDF
    ds = xr.open_dataset(curr_climate_file)
    assert "t2m" in ds, "t2m variable not found in NetCDF"

    # Grid info
    lon = ds.longitude.values
    lat = ds.latitude.values
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    temp = ds["t2m"]
    
    # Create meshgrid of ERA5 lon/lat
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    # Determine hemisphere (based on DEM center)
    center_lat = (lat[0] + lat[-1]) / 2
    lapse_rate_all = lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem
   
    # Load geopotential
    geop = xr.open_dataset(geopotential_path)
    assert "z" in geop, "Missing 'z' in geopotential file"

    # Compute z0: geopotential/9.81 for each ERA pixel
    z0 = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z0[i, j] = Z.values.item() / 9.81

            except:
                z0[i, j] = np.nan  # fallback if not found

    # ERA grid transform
    dx = np.abs(lon[1] - lon[0])
    dy = np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)

    for i, timestep in enumerate(tqdm(time, desc="Downscaling temperature")):
        
        temp_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
        
        date = pd.to_datetime(str(timestep))
        month_index = date.month - 1
        lapse_rate = lapse_rate_all[month_index]

        # Downscaling
        
        t_0 = temp_raw + lapse_rate * (0 - z0)  # realistic version assuming single reference height

        # Reproject to DEM grid
        
        # Prepare output array
        t0_resampled = np.empty_like(dem, dtype=np.float32)
        
        # ERA grid is in EPSG:4326
        era_crs = CRS.from_epsg(4326)
        
        
       # Reproject temperature to DEM grid
        reproject(
            source=t_0,
            destination=t0_resampled,
            src_transform=era_transform,
            src_crs=era_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )
        
        temperature_downscaled = t0_resampled - lapse_rate * (dem - 0)
        dem_meta.update({
                            "dtype": "float32",
                            "count": 1
                        })
        # Save GeoTIFF
        out_name = f"temperature_downscaled_{date.strftime('%Y%m%dT%H%M')}.tif"
        out_path = os.path.join(output_folder_T, out_name)
        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(temperature_downscaled.astype(np.float32), 1)


    print(f"\nDownscaling complete. Files saved in: {output_folder_T}")






