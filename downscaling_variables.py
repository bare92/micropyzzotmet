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
import glob
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.crs import CRS
from tqdm import tqdm
import pandas as pd
from rasterio.warp import reproject, Resampling
import matplotlib.pyplot as plt
from affine import Affine
from utils import write_downscaled_to_netcdf
from scipy.stats import linregress
    
def downscale_Temperature(dem_path, curr_climate_file, output_folder_T, custom_lapse_rate=None, calibrate_lapse_rate=False, dem_nodata=None):
   

    geopotential_path = './auxiliary_data/geopotential3.nc'

    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0

    os.makedirs(output_folder_T, exist_ok=True)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    ds = xr.open_dataset(curr_climate_file)
    assert "t2m" in ds, "t2m variable not found in NetCDF"

    lon = ds.longitude.values
    lat = ds.latitude.values
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    temp = ds["t2m"]
    lon2d, lat2d = np.meshgrid(lon, lat)

    center_lat = (lat[0] + lat[-1]) / 2

    if custom_lapse_rate and calibrate_lapse_rate:
        raise ValueError("Cannot use both custom_lapse_rate and calibrate_lapse_rate=True.")

    if custom_lapse_rate:
        lapse_rate_all = np.array(custom_lapse_rate) / 1000.0
    elif not calibrate_lapse_rate:
        lapse_rate_all = lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem

    geop = xr.open_dataset(geopotential_path)
    assert "z" in geop, "Missing 'z' in geopotential file"

    z0 = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z0[i, j] = Z.values.item() / 9.81
            except:
                z0[i, j] = np.nan

    dx = np.abs(lon[1] - lon[0])
    dy = np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = CRS.from_epsg(4326)

    for i, timestep in enumerate(tqdm(time, desc="Downscaling temperature")):
        date = pd.to_datetime(str(timestep))
        out_name = f"temperature_downscaled_{date.strftime('%Y%m%dT%H%M')}.tif"
        out_path = os.path.join(output_folder_T, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {out_name} (already exists).")
            continue

        temp_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
        month_index = date.month - 1

        # Dynamic lapse rate estimation
        if calibrate_lapse_rate:
            T_vals = temp_raw.flatten()
            Z_vals = z0.flatten()
            valid = ~np.isnan(T_vals) & ~np.isnan(Z_vals)
            if np.sum(valid) < 5:
                lapse_rate = lapse_rate_sohem[month_index] if center_lat < 0 else lapse_rate_nohem[month_index]
            else:
                slope, _, _, _, _ = linregress(Z_vals[valid], T_vals[valid])
                lapse_rate = -slope  # lapse rate is -dT/dz
        else:
            lapse_rate = lapse_rate_all[month_index]

        t_0 = temp_raw - lapse_rate * (0 - z0)

        t0_resampled = np.empty_like(dem, dtype=np.float32)
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
        temperature_downscaled[dem_mask] = np.nan

        dem_meta.update({"dtype": "float32", "count": 1})
        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(temperature_downscaled.astype(np.float32), 1)

    print(f"\nDownscaling complete. Files saved in: {output_folder_T}")



# def downscale_SW(dem_path, curr_climate_file, output_folder_SW, z_700=3000, S0=1370.0, custom_lapse_rate=None, dem_nodata=None):
    

#     a, b, c = 611.21, 17.502, 240.97
#     geopotential_path = './auxiliary_data/geopotential3.nc'
#     working_directory = os.path.dirname(os.path.dirname(os.path.dirname(curr_climate_file)))
#     slope_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*slope*.tif'))[0]
#     aspect_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*aspect*.tif'))[0]
#     os.makedirs(output_folder_SW, exist_ok=True)

#     # Load DEM
#     with rasterio.open(dem_path) as dem_src:
#         dem = dem_src.read(1)
#         dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
#         dem_meta = dem_src.meta.copy()
#         dem_crs = dem_src.crs
#         dem_transform = dem_src.transform
#         ny, nx = dem.shape
#         x_coords = np.arange(nx) * dem_transform.a + dem_transform.c
#         y_coords = np.arange(ny) * dem_transform.e + dem_transform.f

#     with rasterio.open(slope_path) as slope_src:
#         slope_rad = np.radians(slope_src.read(1))
#     with rasterio.open(aspect_path) as aspect_src:
#         aspect_rad = np.radians(aspect_src.read(1))

#     # Lapse rates and coefficients
#     lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
#     lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
#     vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
#     vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

#     ds = xr.open_dataset(curr_climate_file)
#     sw = ds["ssrd"]
#     time = ds.valid_time.values if "valid_time" in ds else ds.time.values
#     lon, lat = ds.longitude.values, ds.latitude.values
#     lon2d, lat2d = np.meshgrid(lon, lat)
    
   
      
#     month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    
   
    
#     out_nc = os.path.join(output_folder_SW, f"shortwave_downscaled_{month_tag}.nc")
    
#     if os.path.exists(out_nc):
#         print(f"Output already exists: {out_nc}. Skipping downscaling.")
#         return
    
#     center_lat = (lat[0] + lat[-1]) / 2
#     lapse_rate_all = np.array(custom_lapse_rate) / 1000.0 if custom_lapse_rate else (lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem)
#     vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem
#     lat_mean_rad = np.radians(center_lat)

#     # Reference elevation (z0) from geopotential
#     geop = xr.open_dataset(geopotential_path)
#     z0 = np.zeros_like(lat2d, dtype=np.float32)
#     for i in range(lat2d.shape[0]):
#         for j in range(lat2d.shape[1]):
#             try:
#                 Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
#                 z0[i, j] = Z.values.item() / 9.81
#             except:
#                 z0[i, j] = np.nan

#     dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
#     era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
#     era_crs = rasterio.crs.CRS.from_epsg(4326)

#     Qsi_all = []
#     time_list = []

#     for i, timestep in enumerate(tqdm(time, desc="Downscaling shortwave radiation")):
#         date = pd.to_datetime(str(timestep))
#         month_index = date.month - 1
#         lapse_rate = lapse_rate_all[month_index]
#         vp_coeff = vp_coeff_all[month_index]
#         d_t_lapse_rate = vp_coeff * c / b
        
        

#         t_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
#         d_raw = dew.isel(valid_time=i).values if "valid_time" in dew.dims else dew.isel(time=i).values

#         t_0 = t_raw - lapse_rate * (0 - z0)
#         d_0 = d_raw - d_t_lapse_rate * (0 - z0)
#         T_700 = t_0 - lapse_rate * (z_700 - z0) - 273.15
#         D_700 = d_0 - d_t_lapse_rate * (z_700 - z0) - 273.15

#         es = a * np.exp((b * T_700) / (T_700 + c))
#         e = a * np.exp((b * D_700) / (D_700 + c))
#         RH_700 = np.clip(100 * e / es, 0, 100)
#         cloud_frac = np.clip(0.832 * np.exp((RH_700 - 100) / 41.6), 0, 1)

#         hour = date.hour + date.minute / 60
#         delta = -23.44 * np.pi / 180 * np.cos(2 * np.pi * (date.dayofyear + 10) / 365)
#         omega = np.pi * (hour - 12) / 12
#         cosZ = np.clip(np.sin(lat_mean_rad) * np.sin(delta) + np.cos(lat_mean_rad) * np.cos(delta) * np.cos(omega), 0, 1)
#         phi = np.arcsin(np.clip(np.cos(delta) * np.sin(omega) / max(np.sin(np.arccos(cosZ)), 1e-6), -1, 1))
#         cos_i = np.clip(np.cos(slope_rad) * cosZ + np.sin(slope_rad) * np.sqrt(1 - cosZ**2) * np.cos(phi - aspect_rad), 0, 1)

#         trans_dir = (0.6 + 0.2 * cosZ) * (1.0 - cloud_frac)
#         trans_dif = (0.3 + 0.1 * cosZ) * cloud_frac

#         cloud_resampled = np.empty_like(dem, dtype=np.float32)
#         trans_dir_resampled = np.empty_like(dem, dtype=np.float32)
#         trans_dif_resampled = np.empty_like(dem, dtype=np.float32)

#         reproject(cloud_frac, cloud_resampled, src_transform=era_transform, src_crs=era_crs,
#                   dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

#         reproject(np.full_like(cloud_frac, trans_dir), trans_dir_resampled, src_transform=era_transform, src_crs=era_crs,
#                   dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

#         reproject(np.full_like(cloud_frac, trans_dif), trans_dif_resampled, src_transform=era_transform, src_crs=era_crs,
#                   dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

#         Qsi = S0 * (trans_dir_resampled * cos_i + trans_dif_resampled * cosZ)
#         Qsi[dem_mask] = np.nan

#         Qsi_all.append(Qsi[np.newaxis, ...])
#         time_list.append(date)
    
#     write_downscaled_to_netcdf(
#         variables_dict={
#             "SW": (Qsi_all, "W m-2", "Downscaled incoming shortwave radiation")
#         },
#         time_list=time_list,
#         dem_shape=dem.shape,
#         dem_transform=dem_transform,
#         dem_crs=dem_crs,
#         out_nc=out_nc
#     )



def downscale_SW_original(dem_path, curr_climate_file, output_folder_SW, z_700=3000, S0=1370.0, custom_lapse_rate=None, calibrate_lapse_rate=False, dem_nodata=None):
    a, b, c = 611.21, 17.502, 240.97
    geopotential_path = './auxiliary_data/geopotential3.nc'
    working_directory = os.path.dirname(os.path.dirname(os.path.dirname(curr_climate_file)))
    slope_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*slope*.tif'))[0]
    aspect_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*aspect*.tif'))[0]
    os.makedirs(output_folder_SW, exist_ok=True)

    # Load DEM
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        ny, nx = dem.shape
        x_coords = np.arange(nx) * dem_transform.a + dem_transform.c
        y_coords = np.arange(ny) * dem_transform.e + dem_transform.f

    with rasterio.open(slope_path) as slope_src:
        slope_rad = np.radians(slope_src.read(1))
    with rasterio.open(aspect_path) as aspect_src:
        aspect_rad = np.radians(aspect_src.read(1))

    # Lapse rates and coefficients
    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
    vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

    ds = xr.open_dataset(curr_climate_file)
    temp, dew = ds["t2m"], ds["d2m"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)

    month_tag = pd.to_datetime(str(time[0])).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_SW, f"shortwave_downscaled_{month_tag}.nc")
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

    center_lat = (lat[0] + lat[-1]) / 2

    if custom_lapse_rate and calibrate_lapse_rate:
        raise ValueError("Cannot use both custom_lapse_rate and calibrate_lapse_rate=True.")

    if custom_lapse_rate:
        lapse_rate_all = np.array(custom_lapse_rate) / 1000.0
    elif not calibrate_lapse_rate:
        lapse_rate_all = lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem

    vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem
    lat_mean_rad = np.radians(center_lat)

    # Reference elevation (z0) from geopotential
    geop = xr.open_dataset(geopotential_path)
    z0 = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z0[i, j] = Z.values.item() / 9.81
            except:
                z0[i, j] = np.nan

    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = rasterio.crs.CRS.from_epsg(4326)

    Qsi_all = []
    time_list = []

    for i, timestep in enumerate(tqdm(time, desc="Downscaling shortwave radiation")):
        date = pd.to_datetime(str(timestep))
        month_index = date.month - 1

        # Calibrate lapse rate dynamically
        if calibrate_lapse_rate:
            T_vals = temp.isel(valid_time=i).values.flatten() if "valid_time" in temp.dims else temp.isel(time=i).values.flatten()
            Z_vals = z0.flatten()
            valid = ~np.isnan(T_vals) & ~np.isnan(Z_vals)
            if np.sum(valid) < 5:
                lapse_rate = lapse_rate_sohem[month_index] if center_lat < 0 else lapse_rate_nohem[month_index]
            else:
                slope, _, _, _, _ = linregress(Z_vals[valid], T_vals[valid])
                lapse_rate = -slope
        else:
            lapse_rate = lapse_rate_all[month_index]

        vp_coeff = vp_coeff_all[month_index]
        d_t_lapse_rate = vp_coeff * c / b

        t_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
        d_raw = dew.isel(valid_time=i).values if "valid_time" in dew.dims else dew.isel(time=i).values

        t_0 = t_raw - lapse_rate * (0 - z0)
        d_0 = d_raw - d_t_lapse_rate * (0 - z0)
        T_700 = t_0 - lapse_rate * (z_700 - z0) - 273.15
        D_700 = d_0 - d_t_lapse_rate * (z_700 - z0) - 273.15

        es = a * np.exp((b * T_700) / (T_700 + c))
        e = a * np.exp((b * D_700) / (D_700 + c))
        RH_700 = np.clip(100 * e / es, 0, 100)
        cloud_frac = np.clip(0.832 * np.exp((RH_700 - 100) / 41.6), 0, 1)

        hour = date.hour + date.minute / 60
        delta = -23.44 * np.pi / 180 * np.cos(2 * np.pi * (date.dayofyear + 10) / 365)
        omega = np.pi * (hour - 12) / 12
        cosZ = np.clip(np.sin(lat_mean_rad) * np.sin(delta) + np.cos(lat_mean_rad) * np.cos(delta) * np.cos(omega), 0, 1)
        phi = np.arcsin(np.clip(np.cos(delta) * np.sin(omega) / max(np.sin(np.arccos(cosZ)), 1e-6), -1, 1))
        cos_i = np.clip(np.cos(slope_rad) * cosZ + np.sin(slope_rad) * np.sqrt(1 - cosZ**2) * np.cos(phi - aspect_rad), 0, 1)

        trans_dir = (0.6 + 0.2 * cosZ) * (1.0 - cloud_frac)
        trans_dif = (0.3 + 0.1 * cosZ) * cloud_frac

        trans_dir_resampled = np.empty_like(dem, dtype=np.float32)
        trans_dif_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(np.full_like(cloud_frac, trans_dir), trans_dir_resampled, src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)
        reproject(np.full_like(cloud_frac, trans_dif), trans_dif_resampled, src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

        Qsi = S0 * (trans_dir_resampled * cos_i + trans_dif_resampled * cosZ)
        Qsi[dem_mask] = np.nan
        Qsi_all.append(Qsi[np.newaxis, ...])
        time_list.append(date)

    units = "W m-2"
    write_downscaled_to_netcdf(
        variables_dict={
            "SW": (Qsi_all, units, "Downscaled incoming shortwave radiation")
        },
        time_list=time_list,
        dem_shape=dem.shape,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        out_nc=out_nc
    )



def downscale_RH(dem_path, curr_climate_file, output_folder_RH, custom_lapse_rate=None, calibrate_lapse_rate=False, dem_nodata=None):
    a, b, c = 611.21, 17.502, 240.97
    geopotential_path = './auxiliary_data/geopotential3.nc'
    os.makedirs(output_folder_RH, exist_ok=True)

    # Lapse rates
    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
    vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

    # Load DEM
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    ds = xr.open_dataset(curr_climate_file)
    temp, dew = ds["t2m"], ds["d2m"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)

    month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_RH, f"relative_humidity_{month_tag}.nc")
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

    center_lat = (lat[0] + lat[-1]) / 2

    if custom_lapse_rate and calibrate_lapse_rate:
        raise ValueError("Cannot use both custom_lapse_rate and calibrate_lapse_rate=True.")

    if custom_lapse_rate:
        lapse_rate_all = np.array(custom_lapse_rate) / 1000.0
    elif not calibrate_lapse_rate:
        lapse_rate_all = lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem

    vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem

    geop = xr.open_dataset(geopotential_path)
    z0 = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z0[i, j] = Z.values.item() / 9.81
            except:
                z0[i, j] = np.nan

    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = CRS.from_epsg(4326)

    RH_all = []
    time_list = []

    for i, timestep in enumerate(tqdm(time, desc="Downscaling relative humidity")):
        date = pd.to_datetime(str(timestep))
        month_index = date.month - 1

        # Dynamic lapse rate estimation
        if calibrate_lapse_rate:
            T_vals = temp.isel(valid_time=i).values.flatten() if "valid_time" in temp.dims else temp.isel(time=i).values.flatten()
            Z_vals = z0.flatten()
            valid = ~np.isnan(T_vals) & ~np.isnan(Z_vals)
            if np.sum(valid) < 5:
                lapse_rate = lapse_rate_sohem[month_index] if center_lat < 0 else lapse_rate_nohem[month_index]
            else:
                slope, _, _, _, _ = linregress(Z_vals[valid], T_vals[valid])
                lapse_rate = -slope
        else:
            lapse_rate = lapse_rate_all[month_index]

        vp_coeff = vp_coeff_all[month_index]
        d_t_lapse_rate = vp_coeff * c / b

        t_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
        d_raw = dew.isel(valid_time=i).values if "valid_time" in dew.dims else dew.isel(time=i).values

        t_0 = t_raw - lapse_rate * (0 - z0)
        d_0 = d_raw - d_t_lapse_rate * (0 - z0)

        t0_resampled = np.empty_like(dem, dtype=np.float32)
        d0_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(t_0, t0_resampled, src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

        reproject(d_0, d0_resampled, src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

        T_down = t0_resampled - lapse_rate * (dem - 0) - 273.15
        D_down = d0_resampled - d_t_lapse_rate * (dem - 0) - 273.15

        es = a * np.exp((b * T_down) / (T_down + c))
        e = a * np.exp((b * D_down) / (D_down + c))
        RH = np.clip(100 * e / es, 0, 100)
        RH[dem_mask] = np.nan

        RH_all.append(RH[np.newaxis, ...])
        time_list.append(date)

    write_downscaled_to_netcdf(
        variables_dict={
            "RH": (RH_all, "%", "Downscaled relative humidity")
        },
        time_list=time_list,
        dem_shape=dem.shape,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        out_nc=out_nc
    )


def downscale_Precipitation(dem_path, curr_climate_file, output_folder_P, custom_gamma=None, dem_nodata=None):
    

    geopotential_path = './auxiliary_data/geopotential3.nc'
    os.makedirs(output_folder_P, exist_ok=True)

    gamma_nohem = np.array([0.35, 0.35, 0.35, 0.30, 0.25, 0.20, 0.20, 0.20, 0.20, 0.25, 0.30, 0.35]) / 1000.0
    gamma_sohem = np.array([0.25, 0.20, 0.20, 0.20, 0.20, 0.25, 0.30, 0.35, 0.35, 0.35, 0.30, 0.25]) / 1000.0

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        ny, nx = dem.shape
        x_coords = np.arange(nx) * dem_transform.a + dem_transform.c
        y_coords = np.arange(ny) * dem_transform.e + dem_transform.f

    ds = xr.open_dataset(curr_climate_file)
    precip = ds["tp"] if "tp" in ds else ds["precip"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    
    month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_P, f"precipitation_{month_tag}.nc")
    
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

    center_lat = (lat[0] + lat[-1]) / 2
    gamma_all = np.array(custom_gamma) / 1000.0 if custom_gamma else (gamma_sohem if center_lat < 0 else gamma_nohem)

    geop = xr.open_dataset(geopotential_path)
    z0 = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z0[i, j] = Z.values.item() / 9.81
            except:
                z0[i, j] = np.nan

    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = CRS.from_epsg(4326)

    precip_all = []
    time_list = []

    for i, timestep in enumerate(tqdm(time, desc="Downscaling precipitation")):
        date = pd.to_datetime(str(timestep))
        month_index = date.month - 1
        gamma = gamma_all[month_index]

        precip_raw = precip.isel(valid_time=i).values if "valid_time" in precip.dims else precip.isel(time=i).values

        p0_resampled = np.empty_like(dem, dtype=np.float32)
        z0_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(precip_raw, p0_resampled,
                  src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs,
                  resampling=Resampling.bilinear)

        reproject(z0, z0_resampled,
                  src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs,
                  resampling=Resampling.bilinear)

        dz = dem - z0_resampled
        precip_downscaled = p0_resampled * ((1 + gamma * dz) / (1 + np.abs(gamma * dz)))

        precip_downscaled[dem_mask] = np.nan
        precip_all.append(precip_downscaled[np.newaxis, ...])
        time_list.append(date)



    write_downscaled_to_netcdf(
        variables_dict={
            "P": (precip_all, "mm", "Downscaled relative humidity")
        },
        time_list=time_list,
        dem_shape=dem.shape,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        out_nc=out_nc
    )

def downscale_Wind(dem_path, curr_climate_file, output_folder_W, slope_weight=0.5, dem_nodata=None):

    os.makedirs(output_folder_W, exist_ok=True)
    working_directory = os.path.dirname(os.path.dirname(os.path.dirname(curr_climate_file)))
    curvature_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*curvature*.tif'))[0]

    curvature_weight = 1 - slope_weight

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        ny, nx = dem.shape
        x_coords = np.arange(nx) * dem_transform.a + dem_transform.c
        y_coords = np.arange(ny) * dem_transform.e + dem_transform.f

    with rasterio.open(curvature_path) as curv_src:
        curvature = curv_src.read(1)

    slope_u = np.gradient(dem, axis=1) / dem_transform[0]
    slope_v = np.gradient(dem, axis=0) / dem_transform[0]
    slope = np.sqrt(np.arctan((slope_u ** 2 + slope_v ** 2)))
    aspect = 3 * np.pi / 2 - np.arctan2(slope_v, slope_u)

    ds = xr.open_dataset(curr_climate_file)
    assert "u10" in ds and "v10" in ds, "Missing 'u10' or 'v10' in NetCDF"

    u10 = ds["u10"]
    v10 = ds["v10"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    
    month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_W, f"wind_speed_direction_{month_tag}.nc")
    
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return
    
    
    lon, lat = ds.longitude.values, ds.latitude.values
    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = CRS.from_epsg(4326)

    wind_speed_all = []
    wind_dir_all = []
    time_list = []

    for i, timestep in enumerate(tqdm(time, desc="Downscaling wind speed and direction")):
        date = pd.to_datetime(str(timestep))
        u_raw = u10.isel(valid_time=i).values if "valid_time" in u10.dims else u10.isel(time=i).values
        v_raw = v10.isel(valid_time=i).values if "valid_time" in v10.dims else v10.isel(time=i).values

        wind_u_resampled = np.empty_like(dem, dtype=np.float32)
        wind_v_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(u_raw, wind_u_resampled, src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)
        reproject(v_raw, wind_v_resampled, src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs, resampling=Resampling.bilinear)

        wind_speed = np.sqrt(wind_u_resampled**2 + wind_v_resampled**2)
        wind_direction = 3 * np.pi / 2 - np.arctan2(wind_v_resampled, wind_u_resampled)

        slope_wind_direction = slope * np.cos(wind_direction - aspect)

        min_slope = np.nanmin(slope_wind_direction)
        max_slope = np.nanmax(slope_wind_direction)
        range_slope = max_slope - min_slope
        slope_norm = (slope_wind_direction - min_slope) / range_slope if range_slope > 0 else np.zeros_like(slope_wind_direction) - 0.5

        min_curv = np.nanmin(curvature)
        max_curv = np.nanmax(curvature)
        range_curv = max_curv - min_curv
        curvature_norm = (curvature - min_curv) / range_curv if range_curv > 0 else np.zeros_like(curvature)

        slope_weighted = slope_weight * slope_norm
        curvature_weighted = curvature_weight * curvature_norm
        sum_weights = slope_weighted + curvature_weighted
        sum_weights[sum_weights == 0] = 1.0
        slope_final = slope_weighted / sum_weights
        curv_final = curvature_weighted / sum_weights

        wind_weighting_factor = 1 + slope_final + curv_final
        wind_speed_adjusted = wind_speed * wind_weighting_factor
        wind_direction_deg = np.degrees(wind_direction)

        wind_speed_adjusted[dem_mask] = np.nan
        wind_direction_deg[dem_mask] = np.nan

        wind_speed_all.append(wind_speed_adjusted[np.newaxis, ...])
        wind_dir_all.append(wind_direction_deg[np.newaxis, ...])
        time_list.append(date)


    write_downscaled_to_netcdf(
        variables_dict={
            "wind_speed": (wind_speed_all, "m s-1", "Downscaled wind speed"),
            "wind_direction": (wind_dir_all, "degrees from north", "Downscaled wind direction")
        },
        time_list=time_list,
        dem_shape=dem.shape,
        dem_transform=dem_transform,
        dem_crs=dem_crs,
        out_nc=out_nc
    )


def downscale_LW(dem_path, curr_climate_file, output_folder_LW, z_700=3000, custom_lapse_rate=None, calibrate_lapse_rate=False, dem_nodata=None):
    """
    Downscale longwave radiation using the MicroMet method,
    including optional lapse rate calibration.
    """
    geopotential_path = './auxiliary_data/geopotential3.nc'
    
    # Default lapse rates
    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
    vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

    os.makedirs(output_folder_LW, exist_ok=True)

    # Constants
    sigma = 5.67e-8
    a, b, c = 611.21, 17.502, 240.97

    # Load DEM
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)

    ds = xr.open_dataset(curr_climate_file)
    T = ds["t2m"]
    D = ds["d2m"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)

    center_lat = (lat[0] + lat[-1]) / 2

    if custom_lapse_rate and calibrate_lapse_rate:
        raise ValueError("Cannot use both custom_lapse_rate and calibrate_lapse_rate=True.")

    if custom_lapse_rate:
        lapse_rate_all = np.array(custom_lapse_rate) / 1000.0
    elif not calibrate_lapse_rate:
        lapse_rate_all = lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem

    vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem

    # Elevation-dependent emissivity parameters
    geop = xr.open_dataset(geopotential_path)
    z = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z[i, j] = Z.values.item() / 9.81
            except:
                z[i, j] = np.nan

    z1, z2 = 200, 3000
    X1, X2 = 0.35, 0.51
    Y1, Y2 = 0.100, 0.130
    Z1, Z2 = 0.224, 1.100

    def interpolate_by_elevation(zval, c1, c2):
        return np.where(
            zval <= z1, c1,
            np.where(zval >= z2, c2, c1 + (zval - z1) * (c2 - c1) / (z2 - z1))
        )

    Xs = interpolate_by_elevation(z, X1, X2)
    Ys = interpolate_by_elevation(z, Y1, Y2)
    Zs = interpolate_by_elevation(z, Z1, Z2)

    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = CRS.from_epsg(4326)

    for i, timestep in enumerate(tqdm(time, desc="Downscaling longwave radiation")):
        date = pd.to_datetime(str(timestep))
        month_index = date.month - 1

        # Dynamic lapse rate estimation
        if calibrate_lapse_rate:
            T_vals = T.isel(valid_time=i).values.flatten() if "valid_time" in T.dims else T.isel(time=i).values.flatten()
            Z_vals = z.flatten()
            valid = ~np.isnan(T_vals) & ~np.isnan(Z_vals)
            if np.sum(valid) < 5:
                lapse_rate = lapse_rate_sohem[month_index] if center_lat < 0 else lapse_rate_nohem[month_index]
            else:
                slope, _, _, _, _ = linregress(Z_vals[valid], T_vals[valid])
                lapse_rate = -slope
        else:
            lapse_rate = lapse_rate_all[month_index]

        vp_coeff = vp_coeff_all[month_index]
        d_t_lapse_rate = vp_coeff * c / b

        t_raw = T.isel(valid_time=i).values if "valid_time" in T.dims else T.isel(time=i).values
        d_raw = D.isel(valid_time=i).values if "valid_time" in D.dims else D.isel(time=i).values

        t_0 = t_raw - lapse_rate * (0 - z)
        d_0 = d_raw - d_t_lapse_rate * (0 - z)
        T_700 = t_0 - lapse_rate * (z_700 - z) - 273.15
        D_700 = d_0 - d_t_lapse_rate * (z_700 - z) - 273.15

        es = a * np.exp((b * T_700) / (T_700 + c))
        e = a * np.exp((b * D_700) / (D_700 + c))
        RH_700 = np.clip(100 * e / es, 0, 100)
        cloud_frac = np.clip(0.832 * np.exp((RH_700 - 100) / 41.6), 0, 1)

        out_name = f"LW_downscaled_{date.strftime('%Y%m%dT%H%M')}.tif"
        out_path = os.path.join(output_folder_LW, out_name)
        if os.path.exists(out_path):
            print(f"Skipping {out_name} (already exists).")
            continue

        T_now = T.isel(valid_time=i).values if "valid_time" in T.dims else T.isel(time=i).values
        D_now = D.isel(valid_time=i).values if "valid_time" in D.dims else D.isel(time=i).values

        e = a * np.exp((b * (D_now - 273.15)) / ((D_now - 273.15) + c))  # vapor pressure in Pa

        eps_atm = (1.083 * (1 + Zs * cloud_frac**2)) * (1 - Xs * np.exp(-Ys * e / T_now))
        Qli = eps_atm * sigma * T_now**4

        Qli_resampled = np.empty_like(dem, dtype=np.float32)
        reproject(
            source=Qli,
            destination=Qli_resampled,
            src_transform=era_transform,
            src_crs=era_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

        Qli_resampled[dem_mask] = np.nan
        dem_meta.update({"dtype": "float32", "count": 1})

        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(Qli_resampled.astype(np.float32), 1)

    print(f"Downscaling complete. Files saved in: {output_folder_LW}")
















