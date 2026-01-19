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
    """
    Check whether a NetCDF file exists and can be opened by xarray.

    This is used to skip already-downloaded or already-generated files.
    Returns True if the file can be opened, otherwise False.
    """
    try:
        xr.open_dataset(file_path).close()
        return True
    except:
        return False


def lon_to_360(lon):
    """
    Convert a longitude value from [-180, 180] to [0, 360] convention.

    Parameters
    ----------
    lon : float
        Longitude in degrees.

    Returns
    -------
    float
        Longitude expressed in [0, 360].
    """
    return lon if lon >= 0 else lon + 360


def get_tiff_extent_latlon(tiff_path, buffer=0.4):
    """
    Get the geographic extent of a raster (GeoTIFF) in lat/lon (EPSG:4326).

    The raster bounds are read from its native CRS and (if needed) reprojected
    to EPSG:4326. A small buffer can be applied to enlarge the bounding box.

    Parameters
    ----------
    tiff_path : str
        Path to a GeoTIFF defining the reference area.
    buffer : float, optional
        Buffer (in degrees) added to each side of the bounding box.

    Returns
    -------
    tuple
        (lat_min, lat_max, lon_min, lon_max) in degrees.
    """
    with rasterio.open(tiff_path) as src:
        bounds = src.bounds
        src_crs = src.crs
        if src_crs.to_epsg() != 4326:
            lon_min, lat_min, lon_max, lat_max = transform_bounds(
                src_crs, "EPSG:4326", bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        else:
            lon_min, lat_min, lon_max, lat_max = bounds.left, bounds.bottom, bounds.right, bounds.top

    return lat_min - buffer, lat_max + buffer, lon_min - buffer, lon_max + buffer


def aggregate_to_daily(ds):
    """
    Aggregate an ERA5-Land hourly dataset to daily frequency.

    Rules applied:
    - Temperature / winds / pressure variables are averaged over the day.
    - Precipitation ('tp') is kept as the last value of the day (cumulative convention).
    - Unknown variables default to daily mean.

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with a 'valid_time' coordinate.

    Returns
    -------
    xarray.Dataset
        Daily-aggregated dataset.
    """
    mean_vars = ['t2m', 'd2m', 'u10', 'v10', 'sp', 'strd', 'ssrd']
    sum_vars = []

    agg_dict = {}
    for var in ds.data_vars:
        if var == 'tp':
            # Keep the last hourly value of the day
            agg_dict[var] = ds[var].resample(valid_time="1D").last()
        elif var in mean_vars:
            agg_dict[var] = ds[var].resample(valid_time="1D").mean()
        elif var in sum_vars:
            agg_dict[var] = ds[var].resample(valid_time="1D").sum()
        else:
            agg_dict[var] = ds[var].resample(valid_time="1D").mean()

    return xr.Dataset(agg_dict)


def process_month(ds, start, output_dir, surface_vars, aggregate_daily=False):
    """
    Extract and post-process one month of ERA5-Land data and save it to NetCDF.

    This function:
    - selects the month time window from the larger dataset
    - subsets the desired surface variables
    - converts longitude convention from [0, 360] to [-180, 180] and sorts longitudes
    - converts precipitation from meters to millimeters and optionally derives hourly increments
    - converts radiation (ssrd/strd) from cumulative J m-2 to W m-2 using time differences
    - optionally aggregates data to daily frequency
    - writes CRS metadata and exports to NetCDF

    Parameters
    ----------
    ds : xarray.Dataset
        Opened ERA5-Land dataset (already spatially subsetted).
    start : pandas.Timestamp
        First day of the month to process.
    output_dir : str
        Folder where output NetCDF files are saved.
    surface_vars : list of str
        Variables to keep (e.g. t2m, d2m, tp, ssrd...).
    aggregate_daily : bool, optional
        If True, dataset is aggregated to daily values.

    Returns
    -------
    None
        The function writes output to disk.
    """
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

    # Handle precipitation
    if 'tp' in ds_surface:
        tp_mm = ds_surface['tp'] * 1000
        if not aggregate_daily:
            tp_mm_hourly = tp_mm.groupby('valid_time.dayofyear').map(
                lambda group: xr.concat(
                    [group.isel(valid_time=0), group.diff('valid_time')],
                    dim='valid_time'
                )
            )
            tp_mm_hourly.attrs['units'] = 'mm'
            tp_mm_hourly.attrs['description'] = 'Converted from m to mm; Converted to mm/hour from daily cumulative'
            ds_surface['tp'] = tp_mm_hourly
        else:
            tp_mm.attrs['units'] = 'mm'
            tp_mm.attrs['description'] = 'Converted from m to mm'
            ds_surface['tp'] = tp_mm

    # Handle radiation (shortwave and longwave)
    for var in ['ssrd', 'strd']:
        if var in ds_surface:
            rad_j = ds_surface[var]
            if not aggregate_daily:
                rad_hourly = rad_j.groupby('valid_time.dayofyear').map(
                    lambda group: xr.concat(
                        [group.isel(valid_time=0), group.diff('valid_time')],
                        dim='valid_time'
                    )
                )
                rad_wm2 = rad_hourly / 3600
                rad_wm2.attrs['units'] = 'W m-2'
                rad_wm2.attrs['description'] = 'Converted from J m-2 to W m-2 using hourly cumulative diff'
                ds_surface[var] = rad_wm2
            else:
                rad_daily = rad_j.resample(valid_time="1D").last() / 86400
                rad_daily.attrs['units'] = 'W m-2'
                rad_daily.attrs['description'] = 'Daily average from cumulative J m-2 (last of day / 86400)'
                ds_surface[var] = rad_daily

    if aggregate_daily:
        ds_surface = aggregate_to_daily(ds_surface)

    ds_surface.latitude.attrs.update(units='degrees_north', standard_name='latitude', axis='Y')
    ds_surface.longitude.attrs.update(units='degrees_east', standard_name='longitude', axis='X')

    # NOTE: requires rioxarray; if it's missing, move this inside a try/except or mock it for docs
    ds_surface = ds_surface.rio.write_crs("EPSG:4326", inplace=True)

    ds_surface.to_netcdf(surface_file)


def get_era5(start_date, end_date, refrence_area_path, output_dir, PAT, jobs_download=1, aggregate_daily=False):
    """
    Download and subset ERA5-Land data from a Zarr store and save it as monthly NetCDF files.

    This function:
    - reads a reference GeoTIFF to determine the lat/lon bounding box
    - opens the ERA5-Land Zarr dataset
    - subsets the data spatially and temporally
    - dispatches monthly processing in parallel (one file per month)

    Parameters
    ----------
    start_date : str or datetime-like
        Start of the requested time period.
    end_date : str or datetime-like
        End of the requested time period.
    refrence_area_path : str
        Path to a reference GeoTIFF used to define the spatial extent.
    output_dir : str
        Output directory where monthly NetCDF files are written.
    PAT : str
        Personal Access Token used to access the EarthDataHub endpoint.
    jobs_download : int, optional
        Number of parallel jobs (currently not used; joblib uses all cores with n_jobs=-1).
    aggregate_daily : bool, optional
        If True, output files contain daily aggregated variables.

    Returns
    -------
    None
        The function writes NetCDF outputs to disk.
    """
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
        delayed(process_month)(
            ds, start, output_dir, variables, aggregate_daily=aggregate_daily
        )
        for start in time_ranges
    )

    print("ERA5 download complete.")
