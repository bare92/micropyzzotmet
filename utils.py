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
import glob
from rasterio.crs import CRS
import gzip
import shutil
from joblib import Parallel, delayed
from rasterio.warp import reproject, calculate_default_transform, Resampling
import rioxarray
from pyproj import Transformer
from rasterio.enums import Resampling
from affine import Affine
from tempfile import TemporaryDirectory
import zarr

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



def lon_to_360(lon):
    return lon % 360

def check_extent_alignment(min_val, max_val, res):
    size = max_val - min_val
    steps = size / res
    if not np.isclose(steps, round(steps)):
        suggested_max = min_val + round(steps) * res
        return False, suggested_max
    return True, None

def create_reference_grid(extent, resolution, crs):
    xmin, ymin, xmax, ymax = extent
    width = int(round((xmax - xmin) / resolution))
    height = int(round((ymax - ymin) / resolution))

    transform = Affine.translation(xmin, ymax) * Affine.scale(resolution, -resolution)
    coords = {
        "y": np.linspace(ymax - resolution / 2, ymin + resolution / 2, height),
        "x": np.linspace(xmin + resolution / 2, xmax - resolution / 2, width)
    }
    dummy_data = np.full((height, width), np.nan)

    da = xr.DataArray(
        dummy_data,
        coords=coords,
        dims=("y", "x"),
        name="dummy"
    )
    da.rio.write_transform(transform, inplace=True)
    da.rio.write_crs(crs, inplace=True)
    return da, width, height

def download_and_save_dem_from_config(config):
    if config["dem_file"] is not None:
        return config["dem_file"]

    extent = config.get("download_dem_extent")
    epsg = config.get("download_dem_epsg", 4326)
    resolution_m = config.get("download_dem_resolution", 30)
    base_folder = config["working_directory"]
    output_folder = os.path.join(base_folder, "inputs", "dem")
    dem_nodata = config.get("dem_nodata", -9999)
    os.makedirs(output_folder, exist_ok=True)

    output_filename = config.get("output_filename_dem", "downloaded_dem.tif")
    output_path = os.path.join(output_folder, output_filename)
    pat = config["earthdatahub_pat"]

    # Transform input extent to EPSG:4326
    transformer_to_4326 = Transformer.from_crs(f"EPSG:{epsg}", "EPSG:4326", always_xy=True)
    lon_min, lat_min = transformer_to_4326.transform(extent["lon_min"], extent["lat_min"])
    lon_max, lat_max = transformer_to_4326.transform(extent["lon_max"], extent["lat_max"])

    # Apply 0.5° buffer for download
    buffer_deg = 0.5
    lon_min_buf = lon_min - buffer_deg
    lon_max_buf = lon_max + buffer_deg
    lat_min_buf = lat_min - buffer_deg
    lat_max_buf = lat_max + buffer_deg

    # Load Copernicus DEM
    url = f"https://edh:{pat}@data.earthdatahub.destine.eu/copernicus-dem/GLO-30-v0.zarr"
    ds = xr.open_dataset(
        url,
        engine="zarr",
        chunks={},
        decode_coords="all",
        mask_and_scale=False
    )

    lat_name = 'lat' if 'lat' in ds.dims else 'latitude'
    lon_name = 'lon' if 'lon' in ds.dims else 'longitude'

    if ds[lon_name].max() > 180:
        lon_min_buf = lon_to_360(lon_min_buf)
        lon_max_buf = lon_to_360(lon_max_buf)

    lat_vals = ds[lat_name].values
    lat_slice = slice(lat_max_buf, lat_min_buf) if lat_vals[0] > lat_vals[-1] else slice(lat_min_buf, lat_max_buf)
    lon_slice = slice(lon_min_buf, lon_max_buf)

    ds_subset = ds.sel({lat_name: lat_slice, lon_name: lon_slice})
    if ds_subset.dsm.size == 0:
        raise ValueError("Selected area contains no data. Check coordinate bounds and try again.")

    ds_subset = ds_subset.rename({lat_name: 'latitude', lon_name: 'longitude'})
    da = ds_subset['dsm']
    da.rio.write_crs("EPSG:4326", inplace=True)

    # Build reference target grid from config-defined extent
    xmin = extent["lon_min"]
    xmax = extent["lon_max"]
    ymin = extent["lat_min"]
    ymax = extent["lat_max"]
    target_extent = (xmin, ymin, xmax, ymax)

    target_grid, width, height = create_reference_grid(
        extent=target_extent,
        resolution=resolution_m,
        crs=f"EPSG:{epsg}"
    )

    # Reproject using reproject_match
    da_matched = da.rio.reproject_match(target_grid, resampling=Resampling.nearest)

    # Final check of extent alignment
    left, bottom, right, top = target_extent
    x_aligned, suggested_right = check_extent_alignment(left, right, resolution_m)
    y_aligned, suggested_top = check_extent_alignment(bottom, top, resolution_m)

    if not x_aligned or not y_aligned:
        raise ValueError(
            f" The crop extent is not aligned with resolution {resolution_m}m.\n"
            f"Suggested bounds (EPSG:{epsg}):\n"
            f"  X: {left} to {suggested_right}\n"
            f"  Y: {bottom} to {suggested_top}\n"
            f"Please update your extent or resolution to match the grid."
        )

    # Save
    da_matched = da_matched.where(~np.isnan(da_matched), other=dem_nodata)

    # Set nodata metadata and save
    da_matched.rio.write_nodata(dem_nodata, inplace=True)
    da_matched.rio.to_raster(output_path)
    print(f" DEM downloaded, matched to config-defined grid and saved to: {output_path}")
    return output_path



def load_era_data(era_path, variables, start_date=None, end_date=None):
    era_ds = xr.open_dataset(era_path)

    # Optionally select variables and time range
    era_ds = era_ds[variables]
    if start_date and end_date and "time" in era_ds.dims:
        era_ds = era_ds.sel(time=slice(start_date, end_date))

    return era_ds



def compute_slope_aspect(dem_path, working_directory):
    output_dir = os.path.join(working_directory, 'inputs', 'dem')
    os.makedirs(output_dir, exist_ok=True)
    slope_path = os.path.join(output_dir, 'slope.tif')
    aspect_path = os.path.join(output_dir, 'aspect.tif')

    with rasterio.open(dem_path) as src:
        if src.crs.to_epsg() == 4326:
            # Reproject to UTM for slope/aspect computation
            lon, lat = (src.bounds.left + src.bounds.right)/2, (src.bounds.top + src.bounds.bottom)/2
            zone = int((lon + 180) / 6) + 1
            utm_epsg = 32600 + zone if lat >= 0 else 32700 + zone

            transform, width, height = calculate_default_transform(
                src.crs, f'EPSG:{utm_epsg}', src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': f'EPSG:{utm_epsg}', 'transform': transform,
                'width': width, 'height': height
            })

            with TemporaryDirectory() as tmpdir:
                projected_path = os.path.join(tmpdir, "reprojected_dem.tif")
                with rasterio.open(projected_path, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=f'EPSG:{utm_epsg}',
                        resampling=Resampling.bilinear
                    )
                
                # Now run gdaldem on projected_path
                tmp_slope = os.path.join(tmpdir, "slope_utm.tif")
                tmp_aspect = os.path.join(tmpdir, "aspect_utm.tif")
                os.system(f'gdaldem slope "{projected_path}" "{tmp_slope}" -of GTiff')
                os.system(f'gdaldem aspect "{projected_path}" "{tmp_aspect}" -of GTiff')

                # Reproject slope/aspect back to original DEM grid
                for input_file, output_file in zip([tmp_slope, tmp_aspect], [slope_path, aspect_path]):
                    with rasterio.open(input_file) as src_tmp, rasterio.open(dem_path) as dst_ref:
                        kwargs_out = dst_ref.meta.copy()
                        kwargs_out.update(dtype=src_tmp.dtypes[0], count=1)
                        with rasterio.open(output_file, 'w', **kwargs_out) as dst:
                            reproject(
                                source=src_tmp.read(1),
                                destination=rasterio.band(dst, 1),
                                src_transform=src_tmp.transform,
                                src_crs=src_tmp.crs,
                                dst_transform=dst_ref.transform,
                                dst_crs=dst_ref.crs,
                                resampling=Resampling.nearest
                            )
        else:
            # DEM already projected: run gdaldem directly
            slope_cmd = f'gdaldem slope "{dem_path}" "{slope_path}" -of GTiff'
            aspect_cmd = f'gdaldem aspect "{dem_path}" "{aspect_path}" -of GTiff'
            os.system(slope_cmd)
            os.system(aspect_cmd)

    print(f"Slope and aspect saved to {output_dir}")
    return slope_path, aspect_path


from tempfile import TemporaryDirectory
from rasterio.warp import calculate_default_transform, reproject, Resampling

def compute_topographic_curvature(dem_path, working_directory, L=1000, dem_nodata=None):
    output_dir = os.path.join(working_directory, 'inputs', 'dem')
    os.makedirs(output_dir, exist_ok=True)
    curvature_path = os.path.join(output_dir, 'curvature.tif')

    if os.path.exists(curvature_path):
        print(f"Curvature already exists at {curvature_path}. Skipping.")
        return curvature_path

    with rasterio.open(dem_path) as src:
        dem_crs = src.crs
        if dem_crs.to_epsg() == 4326:
            # Project to UTM for curvature calculation
            lon, lat = (src.bounds.left + src.bounds.right)/2, (src.bounds.top + src.bounds.bottom)/2
            zone = int((lon + 180) / 6) + 1
            utm_epsg = 32600 + zone if lat >= 0 else 32700 + zone

            transform, width, height = calculate_default_transform(
                src.crs, f"EPSG:{utm_epsg}", src.width, src.height, *src.bounds)
            kwargs = src.meta.copy()
            kwargs.update({
                'crs': f"EPSG:{utm_epsg}",
                'transform': transform,
                'width': width,
                'height': height
            })

            with TemporaryDirectory() as tmpdir:
                projected_path = os.path.join(tmpdir, "projected_dem.tif")
                with rasterio.open(projected_path, 'w', **kwargs) as dst:
                    reproject(
                        source=rasterio.band(src, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=f"EPSG:{utm_epsg}",
                        resampling=Resampling.bilinear
                    )

                # Compute curvature on projected DEM
                with rasterio.open(projected_path) as proj_src:
                    dem = proj_src.read(1).astype(np.float32)
                    transform = proj_src.transform
                    meta = proj_src.meta.copy()

        else:
            with rasterio.open(dem_path) as src:
                dem = src.read(1).astype(np.float32)
                transform = src.transform
                meta = src.meta.copy()

    if dem_nodata is not None:
        dem[dem == dem_nodata] = np.nan

    dx = transform.a
    dy = -transform.e
    cell_size = 0.5 * (dx + dy)

    kernel_diag = np.array([[1, 0, 1],
                            [0, -4, 0],
                            [1, 0, 1]], dtype=np.float32) / (np.sqrt(2) * 4 * cell_size)

    kernel_cross = np.array([[0, 1, 0],
                             [1, -4, 1],
                             [0, 1, 0]], dtype=np.float32) / (4 * cell_size)

    mask = np.isnan(dem)
    dem_filled = np.where(mask, np.nanmean(dem), dem)

    c_diag = convolve(dem_filled, kernel_diag, mode='mirror')
    c_cross = convolve(dem_filled, kernel_cross, mode='mirror')
    curvature = c_diag + c_cross
    curvature[mask] = np.nan

    curve_max = max(0.001, np.nanmax(np.abs(curvature)))
    curvature /= (2.0 * curve_max)

    # If DEM was originally EPSG:4326, reproject curvature back
    if dem_crs.to_epsg() == 4326:
        with rasterio.open(dem_path) as ref_src:
            dst_meta = ref_src.meta.copy()
            dst_meta.update(dtype='float32', count=1, nodata=np.nan)
            with rasterio.open(curvature_path, 'w', **dst_meta) as dst:
                reproject(
                    source=curvature,
                    destination=rasterio.band(dst, 1),
                    src_transform=transform,
                    src_crs=f"EPSG:{utm_epsg}",
                    dst_transform=ref_src.transform,
                    dst_crs=ref_src.crs,
                    resampling=Resampling.bilinear
                )
    else:
        meta.update(dtype='float32', count=1, nodata=np.nan)
        with rasterio.open(curvature_path, 'w', **meta) as dst:
            dst.write(curvature.astype(np.float32), 1)

    print(f"Curvature saved to {curvature_path}")
    return curvature_path




def write_downscaled_to_file(
    variables_dict,
    time_list,
    dem_shape,
    dem_transform,
    dem_crs,
    out_nc,
    scale_factor=0.01,
    dtype="int32",
    complevel=4,
    outformat="nc",
    compression=True,
):
    
    height, width = dem_shape
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    dataset_vars = {}
    decimals = int(abs(np.log10(scale_factor)))

    for var_name, (data_list, units, description) in variables_dict.items():
        data_stack = np.concatenate(data_list, axis=0)

        if compression:
            data_stack = np.round(data_stack, decimals=decimals).astype(np.float32)

        da = xr.DataArray(
            data_stack,
            dims=["time", "y", "x"],
            coords={"time": time_list, "y": y_coords, "x": x_coords},
            attrs={"units": units, "description": description},
        )

        dataset_vars[var_name] = da

    ds_out = xr.Dataset(dataset_vars)
    ds_out = (
        ds_out
        .rio.write_transform(dem_transform)
        .rio.write_crs(dem_crs)
        .rio.set_spatial_dims(x_dim="x", y_dim="y")
        .rio.write_coordinate_system()
    )

    os.makedirs(os.path.dirname(out_nc), exist_ok=True)

    if outformat == "nc":
        if compression:
            encoding = {
                var: {
                    "zlib": True,
                    "complevel": complevel,
                    "dtype": dtype,
                    "scale_factor": scale_factor,
                }
                for var in dataset_vars
            }
            ds_out.to_netcdf(out_nc, encoding=encoding)
        else:
            ds_out.to_netcdf(out_nc)

    elif outformat == "zarr":
        compressor = zarr.Blosc(cname="zstd", clevel=complevel, shuffle=2)
        encoding = {
            var: {"compressor": compressor, "scale_factor": scale_factor, "dtype": dtype}
            for var in dataset_vars
        }
        ds_out.to_zarr(out_nc.replace(".nc", ".zarr"), encoding=encoding)

    print(f"\nSaved dataset: {out_nc if outformat=='nc' else out_nc.replace('.nc', '.zarr')}")

    
    


def convert_micromet_to_s3m_inputs(
    micromet_output_dir: str,
    output_dir: str,
    dem_path: str,
    nodata_value: float = -9999.0,
    n_jobs: int = 4
):
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and reproject DEM to WGS84 ---
    with rasterio.open(dem_path) as src:
        terrain_src = src.read(1).astype(np.float32)
        src_crs = src.crs
        transform = src.transform
        height, width = src.height, src.width
        bounds = src.bounds

    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, width, height, *bounds
    )

    terrain = np.full((dst_height, dst_width), nodata_value, dtype=np.float32)
    reproject(
        source=terrain_src,
        destination=terrain,
        src_transform=transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear
    )

    # --- Create x, y coordinates and flip if needed ---
    x_coords = np.arange(dst_width) * dst_transform.a + dst_transform.c + dst_transform.a / 2
    y_coords = np.arange(dst_height) * dst_transform.e + dst_transform.f + dst_transform.e / 2

    if y_coords[1] > y_coords[0]:  # flip to top-to-bottom
        y_coords = y_coords[::-1]
        terrain = terrain[::-1]

    lon2d, lat2d = np.meshgrid(x_coords, y_coords)

    # --- Load Micromet variables ---
    var_map = {
        "Temperature": ("t2m", "AirTemperature"),
        "SW": ("SW", "IncRadiation"),
        "RH": ("RH", "RelHumidity"),
        "P": ("P", "Rain"),
    }

    var_data = {}
    time_index = None

    for folder, (var_in, var_out) in var_map.items():
        paths = sorted(glob.glob(os.path.join(micromet_output_dir, folder, "*.nc")))
        if not paths:
            continue
        ds = xr.open_mfdataset(paths, combine="by_coords")
        var_array = ds[var_in]
        if time_index is None:
            time_index = pd.to_datetime(var_array.time.values)
        var_data[var_out] = var_array

    if time_index is None or time_index.empty:
        raise RuntimeError("No time data found in Micromet outputs")

    # --- Write one NetCDF.gz file per timestep ---
    def _write_one_s3m_file(i, t):
        import rioxarray  # needed in subprocess
        date_str = pd.to_datetime(t).strftime("%Y%m%d%H%M")
        filename_nc = os.path.join(output_dir, f"MeteoData_{date_str}.nc")
        filename_gz = filename_nc + ".gz"
    
        if os.path.exists(filename_gz):
            return
    
        data_vars = {}
    
        # Interpolation is done using original y_coords
        for var_name in ["Rain", "AirTemperature", "IncRadiation", "RelHumidity"]:
            if var_name in var_data:
                data = var_data[var_name].isel(time=i).interp(
                    x=x_coords, y=y_coords, method="nearest"
                ).values.astype(np.float32)
                data[np.isnan(data)] = nodata_value
                if var_name == "AirTemperature":
                    data[data != nodata_value] = data[data != nodata_value] - 273.15
            else:
                data = np.full((dst_height, dst_width), nodata_value, dtype=np.float32)
    
            # Always flip vertically to match terrain
            data = data[::-1, :]
    
            da = xr.DataArray(
                data,
                dims=("y", "x"),
                coords={"x": x_coords, "y": y_coords[::-1]},
                attrs={"coordinates": "longitude latitude"}
            )
            data_vars[var_name] = da
    
        # Flip static fields
        data_vars["terrain"] = xr.DataArray(
            terrain[::-1, :],
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords[::-1]},
            attrs={"coordinates": "longitude latitude"}
        )
        data_vars["longitude"] = xr.DataArray(
            lon2d[::-1, :],
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords[::-1]}
        )
        data_vars["latitude"] = xr.DataArray(
            lat2d[::-1, :],
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords[::-1]},
            attrs={"_FillValue": nodata_value}
        )
    
        # Build dataset
        ds_out = xr.Dataset(data_vars)
        ds_out = ds_out.rio.write_transform(dst_transform)
        ds_out = ds_out.rio.write_crs(dst_crs)
    
        ds_out.attrs.update({
            "ncols": dst_width,
            "nrows": dst_height,
            "nodata_value": int(nodata_value),
            "xllcorner": float(dst_transform.c),
            "yllcorner": float(dst_transform.f + dst_height * dst_transform.e),
            "cellsize": float(dst_transform.a)
        })
    
        ds_out.to_netcdf(filename_nc)
    
        with open(filename_nc, 'rb') as f_in, gzip.open(filename_gz, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
        os.remove(filename_nc)



    # --- Run in parallel ---
    Parallel(n_jobs=n_jobs)(
        delayed(_write_one_s3m_file)(i, t) for i, t in enumerate(time_index)
    )

    print(f"\n 3M .nc.gz export complete: {output_dir} ")



