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

    
def downscale_Temperature(dem_path, curr_climate_file, output_folder_T, custom_lapse_rate=None):
    geopotential_path = './auxiliary_data/geopotential3.nc'

    # Default lapse rates per hemisphere
    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0

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

    lon = ds.longitude.values
    lat = ds.latitude.values
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    temp = ds["t2m"]
    lon2d, lat2d = np.meshgrid(lon, lat)

    center_lat = (lat[0] + lat[-1]) / 2
    if custom_lapse_rate:
        lapse_rate_all = np.array(custom_lapse_rate) / 1000.0
    else:
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
        lapse_rate = lapse_rate_all[month_index]

        t_0 = temp_raw + lapse_rate * (0 - z0)

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
        dem_meta.update({"dtype": "float32", "count": 1})

        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(temperature_downscaled.astype(np.float32), 1)

    print(f"\nDownscaling complete. Files saved in: {output_folder_T}")


def downscale_SW(dem_path, curr_climate_file, output_folder_SW, z_700=3000, S0=1370.0, custom_lapse_rate=None):
    a, b, c = 611.21, 17.502, 240.97
    geopotential_path = './auxiliary_data/geopotential3.nc'
    working_directory = os.path.dirname(os.path.dirname(os.path.dirname(curr_climate_file)))
    slope_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*slope*.tif'))[0]
    aspect_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*aspect*.tif'))[0]
    os.makedirs(output_folder_SW, exist_ok=True)

    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
    vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
    with rasterio.open(slope_path) as slope_src:
        slope_rad = np.radians(slope_src.read(1))
    with rasterio.open(aspect_path) as aspect_src:
        aspect_rad = np.radians(aspect_src.read(1))

    ds = xr.open_dataset(curr_climate_file)
    temp, dew = ds["t2m"], ds["d2m"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    center_lat = (lat[0] + lat[-1]) / 2
    lapse_rate_all = np.array(custom_lapse_rate) / 1000.0 if custom_lapse_rate else (lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem)
    vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem
    lat_mean_rad = np.radians(center_lat)

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

    for i, timestep in enumerate(tqdm(time, desc="Downscaling shortwave radiation")):
        date = pd.to_datetime(str(timestep))
        out_name = f"SW_downscaled_{date.strftime('%Y%m%dT%H%M')}.tif"
        out_path = os.path.join(output_folder_SW, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {out_name} (already exists).")
            continue

        month_index = date.month - 1
        lapse_rate = lapse_rate_all[month_index]
        vp_coeff = vp_coeff_all[month_index]
        d_t_lapse_rate = vp_coeff * c / b

        t_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
        d_raw = dew.isel(valid_time=i).values if "valid_time" in dew.dims else dew.isel(time=i).values

        t_0 = t_raw + lapse_rate * (0 - z0)
        d_0 = d_raw + d_t_lapse_rate * (0 - z0)
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

        cloud_resampled = np.empty_like(dem, dtype=np.float32)
        trans_dir_resampled = np.empty_like(dem, dtype=np.float32)
        trans_dif_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(cloud_frac, cloud_resampled,
                  src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs,
                  resampling=Resampling.bilinear)

        reproject(np.full_like(cloud_frac, trans_dir), trans_dir_resampled,
                  src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs,
                  resampling=Resampling.bilinear)

        reproject(np.full_like(cloud_frac, trans_dif), trans_dif_resampled,
                  src_transform=era_transform, src_crs=era_crs,
                  dst_transform=dem_transform, dst_crs=dem_crs,
                  resampling=Resampling.bilinear)

        Qsi = S0 * (trans_dir_resampled * cos_i + trans_dif_resampled * cosZ)

        dem_meta.update({"dtype": "float32", "count": 1})
        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(Qsi.astype(np.float32), 1)

    print(f"Downscaling complete. Files saved in: {output_folder_SW}")
    
    
def downscale_RH(dem_path, curr_climate_file, output_folder_RH, custom_lapse_rate=None):
    a, b, c = 611.21, 17.502, 240.97
    geopotential_path = './auxiliary_data/geopotential3.nc'
    os.makedirs(output_folder_RH, exist_ok=True)

    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    vp_coeff_nohem = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
    vp_coeff_sohem = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    ds = xr.open_dataset(curr_climate_file)
    temp, dew = ds["t2m"], ds["d2m"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)
    center_lat = (lat[0] + lat[-1]) / 2
    lapse_rate_all = np.array(custom_lapse_rate) / 1000.0 if custom_lapse_rate else (lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem)
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

    for i, timestep in enumerate(tqdm(time, desc="Downscaling relative humidity")):
        date = pd.to_datetime(str(timestep))
        out_name = f"RH_downscaled_{date.strftime('%Y%m%dT%H%M')}.tif"
        out_path = os.path.join(output_folder_RH, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {out_name} (already exists).")
            continue

        month_index = date.month - 1
        lapse_rate = lapse_rate_all[month_index]
        vp_coeff = vp_coeff_all[month_index]
        d_t_lapse_rate = vp_coeff * c / b

        t_raw = temp.isel(valid_time=i).values if "valid_time" in temp.dims else temp.isel(time=i).values
        d_raw = dew.isel(valid_time=i).values if "valid_time" in dew.dims else dew.isel(time=i).values

        t_0 = t_raw + lapse_rate * (0 - z0)
        d_0 = d_raw + d_t_lapse_rate * (0 - z0)

        t0_resampled = np.empty_like(dem, dtype=np.float32)
        d0_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(
            source=t_0,
            destination=t0_resampled,
            src_transform=era_transform,
            src_crs=era_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )
        reproject(
            source=d_0,
            destination=d0_resampled,
            src_transform=era_transform,
            src_crs=era_crs,
            dst_transform=dem_transform,
            dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

        T_down = t0_resampled - lapse_rate * (dem - 0) - 273.15
        D_down = d0_resampled - d_t_lapse_rate * (dem - 0) - 273.15

        es = a * np.exp((b * T_down) / (T_down + c))
        e = a * np.exp((b * D_down) / (D_down + c))
        RH = np.clip(100 * e / es, 0, 100)

        dem_meta.update({"dtype": "float32", "count": 1})
        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(RH.astype(np.float32), 1)

    print(f"Downscaling complete. Files saved in: {output_folder_RH}")


def downscale_Precipitation(dem_path, curr_climate_file, output_folder_P, custom_gamma=None):
    geopotential_path = './auxiliary_data/geopotential3.nc'
    os.makedirs(output_folder_P, exist_ok=True)

    gamma_nohem = np.array([0.35, 0.35, 0.35, 0.30, 0.25, 0.20, 0.20, 0.20, 0.20, 0.25, 0.30, 0.35]) / 1000.0
    gamma_sohem = np.array([0.25, 0.20, 0.20, 0.20, 0.20, 0.25, 0.30, 0.35, 0.35, 0.35, 0.30, 0.25]) / 1000.0

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1)
        dem_meta = dem_src.meta.copy()
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    ds = xr.open_dataset(curr_climate_file)
    precip = ds["tp"] if "tp" in ds else ds["precip"]
    time = ds.valid_time.values if "valid_time" in ds else ds.time.values
    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)

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

    for i, timestep in enumerate(tqdm(time, desc="Downscaling precipitation")):
        date = pd.to_datetime(str(timestep))
        out_name = f"precip_downscaled_{date.strftime('%Y%m%dT%H%M')}.tif"
        out_path = os.path.join(output_folder_P, out_name)

        if os.path.exists(out_path):
            print(f"Skipping {out_name} (already exists).")
            continue

        month_index = date.month - 1
        gamma = gamma_all[month_index]

        precip_raw = precip.isel(valid_time=i).values if "valid_time" in precip.dims else precip.isel(time=i).values

        p0 = precip_raw
        z0_field = z0

        p0_resampled = np.empty_like(dem, dtype=np.float32)
        z0_resampled = np.empty_like(dem, dtype=np.float32)

        reproject(
            source=p0,
            destination=p0_resampled,
            src_transform=era_transform, src_crs=era_crs,
            dst_transform=dem_transform, dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

        reproject(
            source=z0_field,
            destination=z0_resampled,
            src_transform=era_transform, src_crs=era_crs,
            dst_transform=dem_transform, dst_crs=dem_crs,
            resampling=Resampling.bilinear
        )

        dz = dem - z0_resampled
        precip_downscaled = p0_resampled * ((1 + gamma * dz) / (1 + np.abs(gamma * dz)))

        dem_meta.update({"dtype": "float32", "count": 1})
        with rasterio.open(out_path, 'w', **dem_meta) as dst:
            dst.write(precip_downscaled.astype(np.float32), 1)

    print(f"Downscaling complete. Files saved in: {output_folder_P}")























