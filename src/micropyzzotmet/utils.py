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
from netrc import netrc
from urllib.parse import quote


def get_earthdatahub_credentials(machine="earthdatahub.com"):
    """
    Load EarthDataHub credentials from ~/.netrc.

    Parameters
    ----------
    machine : str, default "earthdatahub.com"
        Machine entry name in ~/.netrc.

    Returns
    -------
    tuple[str, str]
        (login, password/token) read from ~/.netrc.

    Raises
    ------
    ValueError
        If credentials cannot be found for the requested machine.
    """
    try:
        auth = netrc().authenticators(machine)
    except Exception as exc:
        raise ValueError(
            f"Unable to read ~/.netrc for machine '{machine}'."
        ) from exc

    if auth is None:
        raise ValueError(
            f"No ~/.netrc credentials found for machine '{machine}'."
        )

    login, _, password = auth
    if not login or not password:
        raise ValueError(
            f"Invalid ~/.netrc credentials for machine '{machine}': missing login or password."
        )

    return login, password


def build_earthdatahub_url(dataset_path, pat=None, machine="earthdatahub.com"):
    """
    Build an authenticated EarthDataHub URL for xarray/fsspec access.

    Parameters
    ----------
    dataset_path : str
        Dataset path under data.earthdatahub.destine.eu (without leading slash).
    pat : str, optional
        Explicit PAT token. If None, credentials are read from ~/.netrc.
    machine : str, default "earthdatahub.com"
        Machine entry name in ~/.netrc.

    Returns
    -------
    str
        HTTPS URL embedding credentials for authenticated access.
    """
    if pat:
        login = "edh"
        password = pat
    else:
        login, password = get_earthdatahub_credentials(machine=machine)

    safe_login = quote(str(login), safe="")
    safe_password = quote(str(password), safe="")
    dataset_path = dataset_path.lstrip("/")

    return f"https://{safe_login}:{safe_password}@data.earthdatahub.destine.eu/{dataset_path}"


def parse_yes_no_flag(value, var_name=""):
    """
  Convert a 'y' / 'n' string flag into a boolean value.

  This helper function is intended for parsing configuration files
  (e.g. JSON or command-line inputs) where yes/no options are provided
  as single-character strings.

  Parameters
  ----------
  value : str
      Input string flag. Must be either ``'y'`` or ``'n'``.
  var_name : str, optional
      Name of the variable being parsed. Used only to improve
      error-message clarity.

  Returns
  -------
  bool
      ``True`` if value is ``'y'``, ``False`` if value is ``'n'``.

  Raises
  ------
  ValueError
      If the input value is not ``'y'`` or ``'n'``.
  """
    if value == "y":
        return True
    elif value == "n":
        return False
    else:
        raise ValueError(f"Invalid value for '{var_name}': {value}. Expected 'y' or 'n'.")


def create_full_micromet_folder_structure(base_path="."):
    
    """
Create the standard Micromet project folder structure.

The function creates the following directories (if they do not exist):

- ``inputs/climate`` : climate forcing data
- ``inputs/dem``     : DEM and derived terrain layers
- ``outputs``        : Micromet / downscaling outputs

Parameters
----------
base_path : str or pathlib.Path, default "."
    Root directory where the Micromet folder structure
    will be created.

Returns
-------
None
    Creates directories on disk.
"""

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
    """
    Load a JSON configuration file.

    Parameters
    ----------
    config_path : str or pathlib.Path
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Dictionary containing the parsed configuration parameters.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def load_dem(dem_path):
    """
    Load a DEM raster from disk.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to the DEM GeoTIFF.

    Returns
    -------
    dem_data : numpy.ndarray
        DEM elevation values (2D array).
    dem_meta : dict
        Raster metadata (driver, dtype, CRS, etc.).
    dem_transform : affine.Affine
        Affine transform describing the DEM georeferencing.
    """

    with rasterio.open(dem_path) as src:
        dem_data = src.read(1)
        dem_meta = src.meta
        dem_transform = src.transform
    return dem_data, dem_meta, dem_transform



def lon_to_360(lon):
    """
    Convert longitude from [-180, 180] range to [0, 360] range.

    Parameters
    ----------
    lon : float or array-like
        Longitude value(s) in degrees.

    Returns
    -------
    float or array-like
        Longitude value(s) wrapped to the [0, 360] domain.
    """

    return lon % 360

def check_extent_alignment(min_val, max_val, res):
    """
    Check whether a spatial extent is aligned with a given resolution.

    This function verifies that ``(max_val - min_val)`` is an integer
    multiple of the grid resolution.

    Parameters
    ----------
    min_val : float
        Minimum coordinate value (e.g. xmin or ymin).
    max_val : float
        Maximum coordinate value (e.g. xmax or ymax).
    res : float
        Target grid resolution.

    Returns
    -------
    aligned : bool
        True if the extent is aligned with the resolution.
    suggested_max : float or None
        Suggested adjusted maximum value if misaligned,
        otherwise ``None``.
    """

    size = max_val - min_val
    steps = size / res
    if not np.isclose(steps, round(steps)):
        suggested_max = min_val + round(steps) * res
        return False, suggested_max
    return True, None

def create_reference_grid(extent, resolution, crs):
    """
    Create a reference xarray grid for reprojection and resampling.

    The grid is defined by a bounding box, resolution, and CRS, and
    contains a dummy data array filled with NaNs. It is primarily used
    as a target grid for ``rio.reproject_match``.

    Parameters
    ----------
    extent : tuple
        Spatial extent defined as (xmin, ymin, xmax, ymax).
    resolution : float
        Grid resolution (same units as CRS).
    crs : str or rasterio.crs.CRS
        Coordinate reference system.

    Returns
    -------
    da : xarray.DataArray
        Dummy DataArray with spatial coordinates and CRS.
    width : int
        Number of grid columns.
    height : int
        Number of grid rows.
    """

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
    """
    Download, reproject, and save a DEM based on configuration settings.

    If a DEM file is already specified in the configuration, that path
    is returned directly. Otherwise, the function downloads Copernicus
    GLO-30 DEM data, subsets it to the requested extent, reprojects it
    to the target CRS and resolution, and saves it to disk.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing DEM download parameters
        (extent, CRS, resolution, output path, authentication token).

    Returns
    -------
    str
        Path to the DEM file on disk.

    Raises
    ------
    ValueError
        If the selected spatial extent contains no DEM data or
        if the extent is not aligned with the requested resolution.
    """

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
    pat = config.get("earthdatahub_pat")

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
    url = build_earthdatahub_url(
        "copernicus-dem/GLO-30-v0.zarr",
        pat=pat,
        machine=config.get("earthdatahub_machine", "earthdatahub.com")
    )
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
    """
    Load ERA5 / ERA5-Land data from NetCDF.

    Parameters
    ----------
    era_path : str or pathlib.Path
        Path to the ERA NetCDF file.
    variables : list of str
        Names of variables to extract from the dataset.
    start_date, end_date : str or datetime-like, optional
        Time range selection. If provided, data are sliced in time.

    Returns
    -------
    xarray.Dataset
        Subset of the ERA dataset containing selected variables
        and time range.
    """

    era_ds = xr.open_dataset(era_path)

    # Optionally select variables and time range
    era_ds = era_ds[variables]
    if start_date and end_date and "time" in era_ds.dims:
        era_ds = era_ds.sel(time=slice(start_date, end_date))

    return era_ds



def compute_slope_aspect(dem_path, working_directory):
    """
    Compute slope and aspect rasters from a DEM.

    If the DEM is in geographic coordinates (EPSG:4326), it is first
    reprojected to an appropriate UTM zone before computing slope and
    aspect using GDAL. Results are then reprojected back to the original
    DEM grid.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to the DEM GeoTIFF.
    working_directory : str or pathlib.Path
        Project working directory where output rasters will be saved.

    Returns
    -------
    slope_path : str
        Path to the slope GeoTIFF.
    aspect_path : str
        Path to the aspect GeoTIFF.
    """

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
    """
    Compute a normalized topographic curvature index from a DEM.

    Curvature is estimated using a combination of diagonal and
    cross-shaped finite-difference kernels and normalized to
    approximately [-0.5, 0.5].

    If the DEM is in geographic coordinates, the computation is
    performed in a projected CRS (UTM) and reprojected back.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to the DEM GeoTIFF.
    working_directory : str or pathlib.Path
        Project working directory.
    L : float, optional
        Characteristic length scale (currently informational).
    dem_nodata : float or int, optional
        No-data value in the DEM.

    Returns
    -------
    str
        Path to the curvature GeoTIFF.
    """

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



def write_downscaled_to_netcdf(
    variables_dict,
    time_list,
    dem_shape,
    dem_transform,
    dem_crs,
    out_nc,
    nodata_value=-9999,
    mode="w"
):
    """
    Write multiple downscaled variables to a CF-compliant NetCDF file.

    This utility function is used by the downscaling routines to export
    gridded variables on the DEM grid, including full spatial
    referencing compatible with GDAL and QGIS.

    Parameters
    ----------
    variables_dict : dict
        Dictionary of variables in the form:
        ``{var_name: (data_list, units, description)}``.
    time_list : list of datetime-like
        Time coordinate values.
    dem_shape : tuple
        Shape of the DEM grid (rows, columns).
    dem_transform : affine.Affine
        DEM affine transform.
    dem_crs : rasterio.crs.CRS
        DEM coordinate reference system.
    out_nc : str or pathlib.Path
        Output NetCDF file path.
    nodata_value : int, default -9999
        No-data value written to the NetCDF.
    mode : {"w", "a"}, default "w"
        NetCDF write mode.

    Returns
    -------
    None
        Writes a NetCDF file to disk.
    """


    height, width = dem_shape
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    dataset_vars = {}

    for var_name, (data_list, units, description) in variables_dict.items():
        data_stack = np.concatenate(data_list, axis=0).astype(np.int16, copy=False)

        # NOTE: do NOT put _FillValue in attrs (xarray reserves it for encoding)
        da = xr.DataArray(
            data_stack,
            dims=["time", "y", "x"],
            coords={"time": time_list, "y": y_coords, "x": x_coords},
            attrs={"units": units, "description": description, "nodata": int(nodata_value)}
        )
        dataset_vars[var_name] = da

    ds_out = xr.Dataset(dataset_vars)

    # Write georeferencing
    # ds_out = ds_out.rio.write_transform(dem_transform)
    # ds_out = ds_out.rio.write_crs(dem_crs)
    
    ds_out = xr.Dataset(dataset_vars)
    ds_out = ds_out.rio.write_transform(dem_transform)
    ds_out = ds_out.rio.write_crs(dem_crs)

    # Ensure CF grid mapping is attached to each variable
    # (rio.write_crs typically creates "spatial_ref")
    for var_name in dataset_vars.keys():
        ds_out[var_name].attrs["grid_mapping"] = "spatial_ref"

    encoding = {
        var_name: {
            "dtype": "int16",
            "_FillValue": np.int16(nodata_value),
            "zlib": False,
            "complevel": 0
        }
        for var_name in dataset_vars.keys()
    }

    os.makedirs(os.path.dirname(out_nc), exist_ok=True)

    # Important: mode="a" overwrites variables (doesn't truly append time).
    # Use this writer only for whole-file writes (mode="w") unless you know what you're doing.
    ds_out.to_netcdf(
        out_nc,
        mode=mode,
        format="NETCDF4",
        unlimited_dims=["time"],
        encoding=encoding
    )

    print(f"\nSaved NetCDF: {out_nc}")



from .additional_outputs import convert_micromet_to_s3m_inputs





