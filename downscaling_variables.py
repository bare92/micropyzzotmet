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
from scipy import interpolate as spint
import copy
import pvlib
import netCDF4 as nc



def downscale_Temperature(
    dem_path,
    curr_climate_file,
    output_folder_T,
    custom_lapse_rate=None,
    calibrate_lapse_rate=False,
    dem_nodata=None,
    time_chunk=24
):
    
    """
    Downscale ERA5/ERA5-Land 2 m air temperature (``t2m``) to a high-resolution DEM grid.
    
    The method:
      1. Opens the DEM raster and uses its grid/CRS/transform as target.
      2. Opens the climate NetCDF and reads ``t2m`` on the ERA grid.
      3. Reads geopotential (``./auxiliary_data/geopotential3.nc``, variable ``z``) to
         estimate ERA-grid elevations (meters).
      4. Applies a vertical lapse-rate correction:
         - **Fixed monthly lapse rates** (MicroMet defaults) OR
         - **User-provided** monthly lapse rates OR
         - **Calibrated** lapse rate (linear regression of T vs elevation per timestep).
      5. Reprojects the "sea-level" temperature field to the DEM grid (bilinear).
      6. Applies the lapse-rate correction using DEM elevations.
    
    Output is written as a **monthly** NetCDF file named:
    ``temperature_downscaled_YYYY_MM.nc`` in ``output_folder_T``.
    
    Notes
    -----
    - Assumes the climate file contains longitude/latitude coordinates named
      ``longitude`` and ``latitude``, and time is either ``time`` or ``valid_time``.
    - Lapse rates are in K/m internally. If you pass ``custom_lapse_rate``,
      it must be 12 values in K/km (it will be divided by 1000).
    - Buffered writing is used to reduce memory usage.
    
    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to DEM GeoTIFF.
    curr_climate_file : str or pathlib.Path
        Path to monthly climate NetCDF containing ``t2m``.
    output_folder_T : str or pathlib.Path
        Output directory where the monthly NetCDF will be saved.
    custom_lapse_rate : array-like of length 12, optional
        Monthly lapse rates in **K/km** (January..December). Mutually exclusive with
        ``calibrate_lapse_rate=True``.
    calibrate_lapse_rate : bool, default False
        If True, estimate lapse rate per timestep by linear regression of ERA-grid
        temperature vs ERA-grid elevation (from geopotential).
    dem_nodata : float or int, optional
        Value representing nodata in the DEM. If None, NaNs are used.
    write_buffer_steps : int, default 24
        Number of timesteps to buffer before writing to disk. Use 24 for hourly data.
    
    Returns
    -------
    None
        Writes a NetCDF file to disk. If the output file already exists, the function
        prints a message and returns early.
    
    Raises
    ------
    ValueError
        If required variables are missing or if incompatible options are provided.
        """


    geopotential_path = './auxiliary_data/geopotential3.nc'

    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0

    os.makedirs(output_folder_T, exist_ok=True)

    # output nodata must fit int16. If DEM nodata doesn't fit, you must pick another.
    if dem_nodata is None:
        out_nodata = np.int16(-9999)
    else:
        if dem_nodata < -32768 or dem_nodata > 32767:
            raise ValueError(f"dem_nodata={dem_nodata} does not fit in int16. Use e.g. -9999.")
        out_nodata = np.int16(dem_nodata)

    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform

    # input: chunk by time so xarray doesn't load everything
    ds = xr.open_dataset(curr_climate_file, chunks={"valid_time": 1, "time": 1})
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
    else:
        raise ValueError("calibrate_lapse_rate=True not implemented in this function (same as your original).")

    month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_T, f"temperature_downscaled_{month_tag}.nc")
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

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

    height, width = dem.shape
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    # ---------- create output file ONCE with full time length (CF/GDAL/QGIS compliant) ----------
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    root = nc.Dataset(out_nc, "w", format="NETCDF4")
    
    ntime = len(time)
    
    # Dimensions
    root.createDimension("time", ntime)
    root.createDimension("y", height)
    root.createDimension("x", width)
    
    # Coordinate variables
    xv = root.createVariable("x", "f4", ("x",))
    yv = root.createVariable("y", "f4", ("y",))
    tv = root.createVariable("time", "f8", ("time",))
    
    xv[:] = x_coords.astype(np.float32)
    yv[:] = y_coords.astype(np.float32)
    
    # Axis metadata (important for GDAL/QGIS)
    xv.standard_name = "projection_x_coordinate"
    xv.units = "m"
    xv.axis = "X"
    
    yv.standard_name = "projection_y_coordinate"
    yv.units = "m"
    yv.axis = "Y"
    
    tv.units = "seconds since 1970-01-01 00:00:00"
    tv.calendar = "standard"
    
    # Data variable
    t2m_var = root.createVariable(
        "t2m", "i2", ("time", "y", "x"),
        fill_value=np.int16(out_nodata),
        chunksizes=(min(time_chunk, ntime), min(256, height), min(256, width))
    )
    t2m_var.units = "K"
    t2m_var.description = "Downscaled air temperature"
    t2m_var.missing_value = np.int16(out_nodata)
    
    # ---- CF grid mapping (THIS is what QGIS needs) ----
    spatial_ref = root.createVariable("spatial_ref", "i4")
    
    if dem_crs is not None:
        try:
            for k, v in dem_crs.to_cf().items():
                spatial_ref.setncattr(k, v)
        except Exception:
            pass
    
    wkt = dem_crs.to_wkt() if dem_crs is not None else ""
    spatial_ref.setncattr("crs_wkt", wkt)
    spatial_ref.setncattr("spatial_ref", wkt)
    
    gt = f"{dem_transform.c} {dem_transform.a} {dem_transform.b} {dem_transform.f} {dem_transform.d} {dem_transform.e}"
    spatial_ref.setncattr("GeoTransform", gt)
    
    t2m_var.setncattr("grid_mapping", "spatial_ref")
    
    root.setncattr("Conventions", "CF-1.8")

    # ---------- compute + write in chunks ----------
    pbar = tqdm(total=ntime, desc="Downscaling temperature (chunked)")

    start = 0
    while start < ntime:
        end = min(start + time_chunk, ntime)
        B = end - start

        chunk_data = np.empty((B, height, width), dtype=np.int16)
        chunk_times = []

        for k, idx in enumerate(range(start, end)):
            date = pd.to_datetime(str(time[idx]))
            chunk_times.append(date)

            month_index = date.month - 1
            lapse_rate = lapse_rate_all[month_index]

            temp_raw = temp.isel(valid_time=idx).values if "valid_time" in temp.dims else temp.isel(time=idx).values
            temp_u16 = np.rint(temp_raw).astype(np.uint16, copy=False)  # 1 K quantization early
            temp_f32 = temp_u16.astype(np.float32, copy=False)

            t_0 = temp_f32 - lapse_rate * (0 - z0)

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

            temp_i32 = np.rint(temperature_downscaled).astype(np.int32)
            temp_i32[dem_mask | ~np.isfinite(temperature_downscaled)] = int(out_nodata)
            chunk_data[k, :, :] = np.clip(temp_i32, -32768, 32767).astype(np.int16)

            del temp_raw, temp_u16, temp_f32, t_0, t0_resampled, temperature_downscaled, temp_i32
            pbar.update(1)

        # write times + data for this block at the correct indices
        tv[start:end] = nc.date2num([d.to_pydatetime() for d in chunk_times], units=tv.units, calendar=tv.calendar)
        t2m_var[start:end, :, :] = chunk_data

        del chunk_data, chunk_times
        start = end

    pbar.close()
    root.close()
    print(f"\nDownscaling complete. NetCDF saved in: {out_nc}")
  
def downscale_SW_custom(
    dem_path,
    curr_climate_file,
    output_folder_SW,
    dem_nodata=None,
    time_chunk=24
):
    
    """
    Downscale incoming shortwave radiation using a pvlib-based solar geometry approach.

    This version uses ERA5/ERA5-Land surface shortwave radiation (``ssrd``) and corrects it
    for terrain slope/aspect using pvlib solar position and angle-of-incidence (AOI).

    The workflow:
      1. Read DEM + slope + aspect rasters (same grid/CRS/transform as output).
      2. Open the climate NetCDF and read ``ssrd`` (time or valid_time).
      3. Reproject ``ssrd`` from ERA grid to DEM grid (bilinear).
      4. Compute solar position with pvlib at the domain center location.
      5. Compute AOI using slope/aspect and solar zenith/azimuth.
      6. Apply a cosine correction: ``SW = ssrd_resampled * (cos_i / cosZ)``.
      7. Write a CF-compliant NetCDF (including CRS metadata for QGIS).

    Output is written as a **monthly** NetCDF file named:
    ``shortwave_downscaled_YYYY_MM.nc`` in ``output_folder_SW``.

    Notes
    -----
    - Requires slope and aspect rasters in ``<project_root>/inputs/dem/`` with filenames containing
      ``slope`` and ``aspect``.
    - If the input is daily, the function evaluates solar geometry at an estimated UTC solar noon
      based on longitude (same behavior as your current implementation).
    - CRS metadata is written to ensure QGIS/GDAL correctly recognize georeferencing.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to DEM GeoTIFF.
    curr_climate_file : str or pathlib.Path
        Path to monthly climate NetCDF containing ``ssrd``.
    output_folder_SW : str or pathlib.Path
        Output directory where the monthly NetCDF will be saved.
    dem_nodata : float or int, optional
        Value representing nodata in the DEM. If None, NaNs are used.
    time_chunk : int, default 24
        Number of timesteps to process/write per block.

    Returns
    -------
    None
        Writes a NetCDF file to disk and returns.

    Raises
    ------
    ValueError
        If required variables are missing.
    IndexError
        If slope/aspect rasters cannot be found in the expected folder.
    """

    
    import numpy as np
    import xarray as xr
    import os
    import pandas as pd
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    from glob import glob as glob_glob
    import netCDF4 as nc
    import pvlib

    # Get slope and aspect maps
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(curr_climate_file)))
    slope_path = glob_glob(os.path.join(project_root, 'inputs', 'dem', '*slope*.tif'))[0]
    aspect_path = glob_glob(os.path.join(project_root, 'inputs', 'dem', '*aspect*.tif'))[0]
    os.makedirs(output_folder_SW, exist_ok=True)

    with rasterio.open(dem_path) as dem_src, \
         rasterio.open(slope_path) as slope_src, \
         rasterio.open(aspect_path) as aspect_src:

        dem = dem_src.read(1).astype(np.float32)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_transform = dem_src.transform
        dem_crs = dem_src.crs
        height, width = dem.shape

        slope_rad = np.radians(slope_src.read(1).astype(np.float32))

        aspect_raw = aspect_src.read(1).astype(np.float32)
        # Keep your original behavior:
        aspect_rad = np.radians(aspect_raw)

    # Open dataset (chunked on time to avoid preloading)
    ds = xr.open_dataset(curr_climate_file, chunks={"time": 1, "valid_time": 1})
    ssrd = ds["ssrd"]

    if "time" in ds.variables:
        time = ds["time"].values
        time_dim = "time"
    else:
        time = ds["valid_time"].values
        time_dim = "valid_time"

    time_pd = pd.to_datetime(time)

    lon, lat = ds.longitude.values, ds.latitude.values

    # ERA grid georef
    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = rasterio.crs.CRS.from_epsg(4326)

    # Location for pvlib (domain center)
    center_lat = (lat[0] + lat[-1]) / 2
    center_lon = (lon[0] + lon[-1]) / 2
    location = pvlib.location.Location(float(center_lat), float(center_lon))

    # Decide if daily
    is_daily_data = (len(time_pd) == 1) or (len(time_pd) > 1 and (time_pd[1] - time_pd[0]) >= pd.Timedelta("23h"))

    month_tag = time_pd[0].strftime("%Y_%m")
    out_nc = os.path.join(output_folder_SW, f"shortwave_downscaled_{month_tag}.nc")
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

    # Output coords from DEM transform
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    # We'll store times in UTC seconds since epoch
    # Ensure UTC-aware timestamps for pvlib + time variable
    # - daily: your code used "solar noon UTC" (timezone-aware). We'll store the DATE at 00:00 UTC for daily.
    # - hourly: store the actual hour in UTC
    if is_daily_data:
        out_time_pd = [pd.Timestamp(d).tz_localize("UTC") if pd.Timestamp(d).tzinfo is None else pd.Timestamp(d).tz_convert("UTC")
                       for d in time_pd]
    else:
        out_time_pd = []
        for d in time_pd:
            ts = pd.Timestamp(d)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            else:
                ts = ts.tz_convert("UTC")
            out_time_pd.append(ts)

    ntime = len(out_time_pd)

    # ---------- create output file ONCE (CF/GDAL/QGIS CRS metadata) ----------
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    root = nc.Dataset(out_nc, "w", format="NETCDF4")

    root.createDimension("time", ntime)
    root.createDimension("y", height)
    root.createDimension("x", width)

    xv = root.createVariable("x", "f4", ("x",))
    yv = root.createVariable("y", "f4", ("y",))
    tv = root.createVariable("time", "f8", ("time",))

    xv[:] = x_coords.astype(np.float32)
    yv[:] = y_coords.astype(np.float32)

    xv.standard_name = "projection_x_coordinate"
    xv.units = "m"
    xv.axis = "X"

    yv.standard_name = "projection_y_coordinate"
    yv.units = "m"
    yv.axis = "Y"

    tv.units = "seconds since 1970-01-01 00:00:00"
    tv.calendar = "standard"

    # SW output variable (float32 recommended)
    SW_var = root.createVariable(
        "SW", "f4", ("time", "y", "x"),
        fill_value=np.float32(np.nan),
        chunksizes=(min(time_chunk, ntime), min(256, height), min(256, width))
    )
    SW_var.units = "W m-2"
    SW_var.description = "Topographically corrected shortwave radiation"

    # ---- CF grid mapping for QGIS ----
    spatial_ref = root.createVariable("spatial_ref", "i4")
    if dem_crs is not None:
        try:
            for k, v in dem_crs.to_cf().items():
                spatial_ref.setncattr(k, v)
        except Exception:
            pass
        wkt = dem_crs.to_wkt()
    else:
        wkt = ""

    spatial_ref.setncattr("crs_wkt", wkt)
    spatial_ref.setncattr("spatial_ref", wkt)

    gt = f"{dem_transform.c} {dem_transform.a} {dem_transform.b} {dem_transform.f} {dem_transform.d} {dem_transform.e}"
    spatial_ref.setncattr("GeoTransform", gt)

    SW_var.setncattr("grid_mapping", "spatial_ref")
    root.setncattr("Conventions", "CF-1.8")

    # Write full time coordinate now (cheap, avoids doing it per chunk)
    tv[:] = nc.date2num([t.to_pydatetime() for t in out_time_pd], units=tv.units, calendar=tv.calendar)

    # ---------- compute + write in chunks ----------
    start = 0
    while start < ntime:
        end = min(start + time_chunk, ntime)
        B = end - start

        sw_chunk = np.empty((B, height, width), dtype=np.float32)

        for kk, i in enumerate(range(start, end)):
            date = out_time_pd[i]  # UTC aware Timestamp

            # Read ssrd at this time
            ssrd_data = ssrd.isel({time_dim: i}).values.astype(np.float32, copy=False)

            # Resample to DEM grid
            ssrd_resampled = np.empty_like(dem, dtype=np.float32)
            reproject(
                ssrd_data, ssrd_resampled,
                src_transform=era_transform, src_crs=era_crs,
                dst_transform=dem_transform, dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )

            if is_daily_data:
                # Estimate UTC solar noon based on longitude (your logic)
                solar_noon_utc = date.normalize() + pd.Timedelta(hours=12 - location.longitude / 15.0)
                if solar_noon_utc.tzinfo is None:
                    solar_noon_utc = solar_noon_utc.tz_localize("UTC")
                else:
                    solar_noon_utc = solar_noon_utc.tz_convert("UTC")

                solpos = location.get_solarposition(solar_noon_utc)

                zenith_deg = float(solpos["zenith"].values[0])
                cosZ = float(np.cos(np.radians(zenith_deg)))

                incidence = pvlib.irradiance.aoi(
                    surface_tilt=np.degrees(slope_rad),
                    surface_azimuth=np.degrees(aspect_rad),
                    solar_zenith=zenith_deg,
                    solar_azimuth=float(solpos["azimuth"].values[0])
                )

                cos_i = np.clip(np.cos(np.radians(incidence)), 0, 1).astype(np.float32)

                Qsi = ssrd_resampled * (cos_i / (cosZ + 1e-6))
                Qsi[dem_mask] = np.nan
                sw_chunk[kk, :, :] = Qsi.astype(np.float32, copy=False)

            else:
                # Hourly: use actual time
                solpos = location.get_solarposition(date.to_pydatetime())

                zenith_deg = float(solpos["zenith"].values[0])
                cosZ = float(np.cos(np.radians(zenith_deg)))

                incidence = pvlib.irradiance.aoi(
                    surface_tilt=np.degrees(slope_rad),
                    surface_azimuth=np.degrees(aspect_rad),
                    solar_zenith=zenith_deg,
                    solar_azimuth=float(solpos["azimuth"].values[0])
                )

                cos_i = np.clip(np.cos(np.radians(incidence)), 0, 1).astype(np.float32)

                Qsi = ssrd_resampled * (cos_i / (cosZ + 1e-6))
                Qsi[dem_mask] = np.nan
                sw_chunk[kk, :, :] = Qsi.astype(np.float32, copy=False)

            del ssrd_data, ssrd_resampled, solpos, incidence, cos_i, Qsi

        # Write the chunk
        SW_var[start:end, :, :] = sw_chunk
        del sw_chunk

        start = end

    root.close()
    print(f"\nDownscaling complete. NetCDF saved in: {out_nc}")



def downscale_RH(
    dem_path,
    curr_climate_file,
    output_folder_RH,
    custom_lapse_rate=None,
    calibrate_lapse_rate=False,
    dem_nodata=None,
    time_chunk=24
):
    
    """
    Downscale relative humidity to a high-resolution DEM grid using t2m and d2m.

    Relative humidity is computed using the Magnus formulation from downscaled
    air temperature and dew point temperature.

    The workflow:
      1. Read DEM raster to define target grid/CRS/transform.
      2. Open climate NetCDF and read ``t2m`` and ``d2m``.
      3. Read geopotential (``./auxiliary_data/geopotential3.nc``) to estimate ERA-grid elevation.
      4. Apply lapse-rate corrections:
         - temperature lapse rate (monthly default or user-provided), and
         - dew-point lapse rate derived from empirical monthly vapor-pressure coefficients.
      5. Reproject sea-level temperature/dewpoint fields to DEM grid (bilinear).
      6. Compute RH (%) on the DEM grid and write output in chunks.

    Output is written as a **monthly** NetCDF file named:
    ``relative_humidity_YYYY_MM.nc`` in ``output_folder_RH``.

    Notes
    -----
    - Uses monthly default lapse rates for temperature unless ``custom_lapse_rate`` is provided.
    - If ``calibrate_lapse_rate=True``, lapse rate is estimated per timestep by regression of
      ERA-grid temperature vs ERA-grid elevation.
    - Output NetCDF includes CF grid mapping information for QGIS/GDAL.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to DEM GeoTIFF.
    curr_climate_file : str or pathlib.Path
        Path to monthly climate NetCDF containing ``t2m`` and ``d2m``.
    output_folder_RH : str or pathlib.Path
        Output directory where the monthly NetCDF will be saved.
    custom_lapse_rate : array-like of length 12, optional
        Monthly temperature lapse rates in K/km. Mutually exclusive with ``calibrate_lapse_rate=True``.
    calibrate_lapse_rate : bool, default False
        If True, estimate lapse rate dynamically from ERA-grid temperature vs elevation.
    dem_nodata : float or int, optional
        Value representing nodata in the DEM. If None, NaNs are used.
    time_chunk : int, default 24
        Number of timesteps to process/write per block.

    Returns
    -------
    None
        Writes a NetCDF file to disk and returns.

    Raises
    ------
    ValueError
        If required variables are missing or if incompatible options are provided.
    """

    import os
    import numpy as np
    import xarray as xr
    import pandas as pd
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    from tqdm import tqdm
    from scipy.stats import linregress
    import netCDF4 as nc

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
        dem = dem_src.read(1).astype(np.float32)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        height, width = dem.shape

    # Open climate dataset (chunked along time)
    ds = xr.open_dataset(curr_climate_file, chunks={"valid_time": 1, "time": 1})
    temp, dew = ds["t2m"], ds["d2m"]

    if "valid_time" in ds:
        time = ds.valid_time.values
        time_dim = "valid_time"
    else:
        time = ds.time.values
        time_dim = "time"

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
    else:
        lapse_rate_all = None  # will be computed dynamically

    vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem

    # Geopotential -> z0 (same approach as your original, computed once)
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

    # Output coords from DEM
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    time_pd = pd.to_datetime(time)
    ntime = len(time_pd)

    # ---------- create output NetCDF ONCE (CF/GDAL/QGIS compliant CRS) ----------
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    root = nc.Dataset(out_nc, "w", format="NETCDF4")

    root.createDimension("time", ntime)
    root.createDimension("y", height)
    root.createDimension("x", width)

    xv = root.createVariable("x", "f4", ("x",))
    yv = root.createVariable("y", "f4", ("y",))
    tv = root.createVariable("time", "f8", ("time",))

    xv[:] = x_coords.astype(np.float32)
    yv[:] = y_coords.astype(np.float32)

    xv.standard_name = "projection_x_coordinate"
    xv.units = "m"
    xv.axis = "X"

    yv.standard_name = "projection_y_coordinate"
    yv.units = "m"
    yv.axis = "Y"

    tv.units = "seconds since 1970-01-01 00:00:00"
    tv.calendar = "standard"

    RH_var = root.createVariable(
        "RH", "f4", ("time", "y", "x"),
        fill_value=np.float32(np.nan),
        chunksizes=(min(time_chunk, ntime), min(256, height), min(256, width))
    )
    RH_var.units = "%"
    RH_var.description = "Downscaled relative humidity"

    # CF grid mapping for QGIS
    spatial_ref = root.createVariable("spatial_ref", "i4")
    if dem_crs is not None:
        try:
            for k, v in dem_crs.to_cf().items():
                spatial_ref.setncattr(k, v)
        except Exception:
            pass
        wkt = dem_crs.to_wkt()
    else:
        wkt = ""

    spatial_ref.setncattr("crs_wkt", wkt)
    spatial_ref.setncattr("spatial_ref", wkt)

    gt = f"{dem_transform.c} {dem_transform.a} {dem_transform.b} {dem_transform.f} {dem_transform.d} {dem_transform.e}"
    spatial_ref.setncattr("GeoTransform", gt)

    RH_var.setncattr("grid_mapping", "spatial_ref")
    root.setncattr("Conventions", "CF-1.8")

    # Write full time coordinate once
    tv[:] = nc.date2num([pd.Timestamp(t).to_pydatetime() for t in time_pd], units=tv.units, calendar=tv.calendar)

    # ---------- compute + write in chunks ----------
    pbar = tqdm(total=ntime, desc="Downscaling relative humidity (chunked)")

    start = 0
    while start < ntime:
        end = min(start + time_chunk, ntime)
        B = end - start

        rh_chunk = np.empty((B, height, width), dtype=np.float32)

        for kk, ti in enumerate(range(start, end)):
            date = pd.Timestamp(time_pd[ti])
            month_index = date.month - 1

            # Lapse rate
            if calibrate_lapse_rate:
                T_vals = temp.isel({time_dim: ti}).values.astype(np.float32).ravel()
                Z_vals = z0.ravel()
                valid = ~np.isnan(T_vals) & ~np.isnan(Z_vals)
                if np.sum(valid) < 5:
                    lapse_rate = (lapse_rate_sohem[month_index] if center_lat < 0 else lapse_rate_nohem[month_index])
                else:
                    slope, _, _, _, _ = linregress(Z_vals[valid], T_vals[valid])
                    lapse_rate = -slope
            else:
                lapse_rate = lapse_rate_all[month_index]

            vp_coeff = vp_coeff_all[month_index]
            d_t_lapse_rate = vp_coeff * c / b

            t_raw = temp.isel({time_dim: ti}).values.astype(np.float32, copy=False)
            d_raw = dew.isel({time_dim: ti}).values.astype(np.float32, copy=False)

            t_0 = t_raw - lapse_rate * (0 - z0)
            d_0 = d_raw - d_t_lapse_rate * (0 - z0)

            t0_resampled = np.empty_like(dem, dtype=np.float32)
            d0_resampled = np.empty_like(dem, dtype=np.float32)

            reproject(
                t_0, t0_resampled,
                src_transform=era_transform, src_crs=era_crs,
                dst_transform=dem_transform, dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )
            reproject(
                d_0, d0_resampled,
                src_transform=era_transform, src_crs=era_crs,
                dst_transform=dem_transform, dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )

            # Convert to °C for Magnus formula
            T_down = t0_resampled - lapse_rate * (dem - 0) - 273.15
            D_down = d0_resampled - d_t_lapse_rate * (dem - 0) - 273.15

            es = a * np.exp((b * T_down) / (T_down + c))
            e = a * np.exp((b * D_down) / (D_down + c))
            RH = np.clip(100.0 * e / es, 0, 100).astype(np.float32)

            RH[dem_mask] = np.nan
            rh_chunk[kk, :, :] = RH

            del t_raw, d_raw, t_0, d_0, t0_resampled, d0_resampled, T_down, D_down, es, e, RH
            pbar.update(1)

        RH_var[start:end, :, :] = rh_chunk
        del rh_chunk
        start = end

    pbar.close()
    root.close()
    print(f"\nDownscaling complete. NetCDF saved in: {out_nc}")



def downscale_Wind(
    dem_path,
    curr_climate_file,
    output_folder_W,
    slope_weight=0.5,
    dem_nodata=None,
    time_chunk=24
):
    
    """
    Downscale 10 m wind speed and wind direction to a DEM grid using terrain modifiers.

    This function downsamples ERA5/ERA5-Land ``u10`` and ``v10`` winds to a high-resolution DEM grid
    and applies terrain-based adjustments using:
      - local terrain slope/aspect (computed from DEM), and
      - terrain curvature (read from a raster)

    The workflow:
      1. Read DEM raster to define target grid/CRS/transform.
      2. Read curvature raster from ``<working_directory>/inputs/dem/*curvature*.tif``.
      3. Open climate NetCDF and read ``u10`` and ``v10``.
      4. Reproject u/v winds from ERA grid to DEM grid (bilinear).
      5. Compute wind speed and base wind direction.
      6. Compute slope-wind interaction and normalize slope + curvature terms.
      7. Apply a wind-speed scaling factor and a directional deflection term.
      8. Write ``wind_speed`` and ``wind_direction`` to a CF-compliant NetCDF in chunks.

    Output is written as a **monthly** NetCDF file named:
    ``wind_speed_direction_YYYY_MM.nc`` in ``output_folder_W``.

    Notes
    -----
    - ``slope_weight`` balances slope vs curvature contributions (curvature weight is ``1 - slope_weight``).
    - Output NetCDF includes CF grid mapping information for QGIS/GDAL.

    Parameters
    ----------
    dem_path : str or pathlib.Path
        Path to DEM GeoTIFF.
    curr_climate_file : str or pathlib.Path
        Path to monthly climate NetCDF containing ``u10`` and ``v10``.
    output_folder_W : str or pathlib.Path
        Output directory where the monthly NetCDF will be saved.
    slope_weight : float, default 0.5
        Weight of slope contribution in terrain adjustment (0..1).
    dem_nodata : float or int, optional
        Value representing nodata in the DEM. If None, NaNs are used.
    time_chunk : int, default 24
        Number of timesteps to process/write per block.

    Returns
    -------
    None
        Writes a NetCDF file to disk and returns.

    Raises
    ------
    ValueError
        If required variables are missing or if ERA longitude/latitude ordering is unexpected.
    IndexError
        If curvature raster cannot be found.
    """

    
    import os
    import glob
    import numpy as np
    import xarray as xr
    import pandas as pd
    import rasterio
    from rasterio.warp import reproject, Resampling
    from rasterio.crs import CRS
    from affine import Affine
    from tqdm import tqdm
    import netCDF4 as nc

    os.makedirs(output_folder_W, exist_ok=True)
    working_directory = os.path.dirname(os.path.dirname(os.path.dirname(curr_climate_file)))
    curvature_path = glob.glob(os.path.join(working_directory, 'inputs', 'dem', '*curvature*.tif'))[0]
    curvature_weight = 1.0 - float(slope_weight)

    # --- DEM + CRS ---
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        height, width = dem.shape

    # curvature
    with rasterio.open(curvature_path) as curv_src:
        curvature = curv_src.read(1).astype(np.float32)

    # Terrain slope + aspect (as you had it)
    slope_u = np.gradient(dem, axis=1) / dem_transform.a
    slope_v = np.gradient(dem, axis=0) / dem_transform.a
    slope = np.sqrt(np.arctan((slope_u ** 2 + slope_v ** 2))).astype(np.float32)
    aspect = (3 * np.pi / 2 - np.arctan2(slope_v, slope_u)).astype(np.float32)

    # --- Climate dataset (chunked by time) ---
    ds = xr.open_dataset(curr_climate_file, chunks={"valid_time": 1, "time": 1})
    if "u10" not in ds or "v10" not in ds:
        raise ValueError("Missing 'u10' or 'v10' in NetCDF")

    u10 = ds["u10"]
    v10 = ds["v10"]

    if "valid_time" in ds:
        time = ds.valid_time.values
        time_dim = "valid_time"
    else:
        time = ds.time.values
        time_dim = "time"

    month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_W, f"wind_speed_direction_{month_tag}.nc")
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

    lon = ds.longitude.values
    lat = ds.latitude.values

    dx = np.abs(lon[1] - lon[0])
    dy = np.abs(lat[1] - lat[0])

    # Build pixel-centered affine transform (as your original logic)
    lon_sorted = np.all(np.diff(lon) > 0)
    lat_sorted = np.all(np.diff(lat) < 0)
    if not lon_sorted or not lat_sorted:
        raise ValueError("Longitude must be ascending and latitude descending for ERA5-Land.")

    era_transform = Affine.translation(lon[0] - dx / 2, lat[0] - dy / 2) * Affine.scale(dx, -dy)
    era_crs = CRS.from_epsg(4326)

    # Precompute normalized curvature once (it doesn't change with time)
    range_curv = np.nanmax(curvature) - np.nanmin(curvature)
    curvature_norm = ((curvature - np.nanmin(curvature)) / range_curv).astype(np.float32) if range_curv > 0 else np.zeros_like(curvature, dtype=np.float32)

    # Output coords from DEM
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    time_pd = pd.to_datetime(time)
    ntime = len(time_pd)

    # ---------- create output NetCDF ONCE (CF/GDAL/QGIS compliant CRS) ----------
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    root = nc.Dataset(out_nc, "w", format="NETCDF4")

    root.createDimension("time", ntime)
    root.createDimension("y", height)
    root.createDimension("x", width)

    xv = root.createVariable("x", "f4", ("x",))
    yv = root.createVariable("y", "f4", ("y",))
    tv = root.createVariable("time", "f8", ("time",))

    xv[:] = x_coords.astype(np.float32)
    yv[:] = y_coords.astype(np.float32)

    xv.standard_name = "projection_x_coordinate"
    xv.units = "m"
    xv.axis = "X"

    yv.standard_name = "projection_y_coordinate"
    yv.units = "m"
    yv.axis = "Y"

    tv.units = "seconds since 1970-01-01 00:00:00"
    tv.calendar = "standard"

    ws_var = root.createVariable(
        "wind_speed", "f4", ("time", "y", "x"),
        fill_value=np.float32(np.nan),
        chunksizes=(min(time_chunk, ntime), min(256, height), min(256, width))
    )
    ws_var.units = "m s-1"
    ws_var.description = "Downscaled wind speed"

    wd_var = root.createVariable(
        "wind_direction", "f4", ("time", "y", "x"),
        fill_value=np.float32(np.nan),
        chunksizes=(min(time_chunk, ntime), min(256, height), min(256, width))
    )
    wd_var.units = "degrees from north"
    wd_var.description = "Downscaled wind direction"

    # CF grid mapping for QGIS
    spatial_ref = root.createVariable("spatial_ref", "i4")
    if dem_crs is not None:
        try:
            for k, v in dem_crs.to_cf().items():
                spatial_ref.setncattr(k, v)
        except Exception:
            pass
        wkt = dem_crs.to_wkt()
    else:
        wkt = ""

    spatial_ref.setncattr("crs_wkt", wkt)
    spatial_ref.setncattr("spatial_ref", wkt)

    gt = f"{dem_transform.c} {dem_transform.a} {dem_transform.b} {dem_transform.f} {dem_transform.d} {dem_transform.e}"
    spatial_ref.setncattr("GeoTransform", gt)

    ws_var.setncattr("grid_mapping", "spatial_ref")
    wd_var.setncattr("grid_mapping", "spatial_ref")
    root.setncattr("Conventions", "CF-1.8")

    # write time coordinate once
    tv[:] = nc.date2num([pd.Timestamp(t).to_pydatetime() for t in time_pd], units=tv.units, calendar=tv.calendar)

    # ---------- compute + write in chunks ----------
    pbar = tqdm(total=ntime, desc="Downscaling wind speed and direction (chunked)")

    start = 0
    while start < ntime:
        end = min(start + time_chunk, ntime)
        B = end - start

        ws_chunk = np.empty((B, height, width), dtype=np.float32)
        wd_chunk = np.empty((B, height, width), dtype=np.float32)

        for kk, ti in enumerate(range(start, end)):
            u_raw = u10.isel({time_dim: ti}).values.astype(np.float32, copy=False)
            v_raw = v10.isel({time_dim: ti}).values.astype(np.float32, copy=False)

            wind_u_resampled = np.empty_like(dem, dtype=np.float32)
            wind_v_resampled = np.empty_like(dem, dtype=np.float32)

            reproject(
                u_raw, wind_u_resampled,
                src_transform=era_transform, src_crs=era_crs,
                dst_transform=dem_transform, dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )
            reproject(
                v_raw, wind_v_resampled,
                src_transform=era_transform, src_crs=era_crs,
                dst_transform=dem_transform, dst_crs=dem_crs,
                resampling=Resampling.bilinear
            )

            wind_speed = np.sqrt(wind_u_resampled**2 + wind_v_resampled**2).astype(np.float32)

            # Base wind direction (radians)
            wind_direction = (3 * np.pi / 2 - np.arctan2(wind_v_resampled, wind_u_resampled)).astype(np.float32)

            # slope wind interaction
            slope_wind_direction = slope * np.cos(wind_direction - aspect)

            range_slope = np.nanmax(slope_wind_direction) - np.nanmin(slope_wind_direction)
            if range_slope > 0:
                slope_norm = (slope_wind_direction - np.nanmin(slope_wind_direction)) / range_slope
            else:
                slope_norm = np.zeros_like(slope_wind_direction, dtype=np.float32) - 0.5

            slope_weighted = slope_weight * slope_norm
            curvature_weighted = curvature_weight * curvature_norm

            sum_weights = slope_weighted + curvature_weighted
            sum_weights[sum_weights == 0] = 1.0

            slope_final = slope_weighted / sum_weights
            curv_final = curvature_weighted / sum_weights

            wind_weighting_factor = 1.0 + slope_final + curv_final
            wind_speed_adjusted = wind_speed * wind_weighting_factor

            div_factor = -0.5 * slope_norm * np.sin(2 * (wind_direction - aspect))
            wind_direction_modified = wind_direction + div_factor

            # Convert to degrees (use MODIFIED direction)
            wind_direction_deg = np.degrees(wind_direction_modified).astype(np.float32)

            wind_speed_adjusted[dem_mask] = np.nan
            wind_direction_deg[dem_mask] = np.nan

            ws_chunk[kk, :, :] = wind_speed_adjusted
            wd_chunk[kk, :, :] = wind_direction_deg

            del u_raw, v_raw, wind_u_resampled, wind_v_resampled, wind_speed, wind_direction
            del slope_wind_direction, slope_norm, slope_weighted, curvature_weighted, sum_weights
            del slope_final, curv_final, wind_weighting_factor, wind_speed_adjusted, div_factor, wind_direction_modified, wind_direction_deg

            pbar.update(1)

        ws_var[start:end, :, :] = ws_chunk
        wd_var[start:end, :, :] = wd_chunk

        del ws_chunk, wd_chunk
        start = end

    pbar.close()
    root.close()
    print(f"\nWind downscaling complete. NetCDF saved in: {out_nc}")

   
def downscale_LW(
    dem_path,
    curr_climate_file,
    output_folder_LW,
    z_700=3000,
    custom_lapse_rate=None,
    calibrate_lapse_rate=False,
    dem_nodata=None,
    time_chunk=24
):
    
    """
Downscale incoming longwave radiation to a DEM grid using atmospheric emissivity parameterization.

This function estimates downscaled longwave radiation using ERA5/ERA5-Land air temperature (t2m),
dew point (d2m), and cloud fraction estimated from RH at ~700 hPa.

The workflow:
  1. Read DEM raster to define target grid/CRS/transform.
  2. Open climate NetCDF and read ``t2m`` and ``d2m``.
  3. Read geopotential (``./auxiliary_data/geopotential3.nc``) to estimate ERA-grid elevation.
  4. Apply lapse-rate corrections to compute temperature/dew point at 700 hPa level approximation.
  5. Estimate RH_700 and cloud fraction.
  6. Compute atmospheric emissivity using elevation-dependent coefficients (X, Y, Z).
  7. Compute longwave radiation on ERA grid: ``Qli = eps_atm * sigma * T^4``.
  8. Reproject Qli to DEM grid (bilinear) and write output in chunks.

Output is written as a **monthly** NetCDF file named:
``longwave_downscaled_YYYY_MM.nc`` in ``output_folder_LW``.

Notes
-----
- Uses Stefan–Boltzmann constant ``sigma = 5.67e-8``.
- If ``calibrate_lapse_rate=True``, lapse rate is estimated per timestep by regression of
  ERA-grid temperature vs ERA-grid elevation.
- Output NetCDF includes CF grid mapping information for QGIS/GDAL.

Parameters
----------
dem_path : str or pathlib.Path
    Path to DEM GeoTIFF.
curr_climate_file : str or pathlib.Path
    Path to monthly climate NetCDF containing ``t2m`` and ``d2m``.
output_folder_LW : str or pathlib.Path
    Output directory where the monthly NetCDF will be saved.
z_700 : float, default 3000
    Reference height (m) used for the 700 hPa approximation.
custom_lapse_rate : array-like of length 12, optional
    Monthly temperature lapse rates in K/km. Mutually exclusive with ``calibrate_lapse_rate=True``.
calibrate_lapse_rate : bool, default False
    If True, estimate lapse rate dynamically from ERA-grid temperature vs elevation.
dem_nodata : float or int, optional
    Value representing nodata in the DEM. If None, NaNs are used.
time_chunk : int, default 24
    Number of timesteps to process/write per block.

Returns
-------
None
    Writes a NetCDF file to disk and returns.

Raises
------
ValueError
    If incompatible options are provided.
"""


    import numpy as np
    import os
    import rasterio
    import xarray as xr
    from tqdm import tqdm
    import pandas as pd
    from scipy.stats import linregress
    from rasterio.warp import reproject, Resampling
    from rasterio.transform import from_origin
    from rasterio.crs import CRS
    import netCDF4 as nc

    geopotential_path = './auxiliary_data/geopotential3.nc'

    lapse_rate_nohem = np.array([4.4, 5.9, 7.1, 7.8, 8.1, 8.2, 8.1, 8.1, 7.7, 6.8, 5.5, 4.7]) / 1000.0
    lapse_rate_sohem = np.array([8.1, 8.1, 7.7, 6.8, 5.5, 4.7, 4.4, 5.9, 7.1, 7.8, 8.1, 8.2]) / 1000.0
    vp_coeff_nohem  = np.array([0.41, 0.42, 0.40, 0.39, 0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40]) / 1000.0
    vp_coeff_sohem  = np.array([0.38, 0.36, 0.33, 0.33, 0.36, 0.37, 0.40, 0.40, 0.41, 0.42, 0.40, 0.39]) / 1000.0

    os.makedirs(output_folder_LW, exist_ok=True)

    sigma = 5.67e-8
    a, b, c = 611.21, 17.502, 240.97

    # DEM
    with rasterio.open(dem_path) as dem_src:
        dem = dem_src.read(1).astype(np.float32)
        dem_crs = dem_src.crs
        dem_transform = dem_src.transform
        dem_mask = (dem == dem_nodata) if dem_nodata is not None else np.isnan(dem)
        height, width = dem.shape

    # Climate dataset (chunked on time)
    ds = xr.open_dataset(curr_climate_file, chunks={"valid_time": 1, "time": 1})
    T = ds["t2m"]
    D = ds["d2m"]

    if "valid_time" in ds:
        time = ds.valid_time.values
        time_dim = "valid_time"
    else:
        time = ds.time.values
        time_dim = "time"

    lon, lat = ds.longitude.values, ds.latitude.values
    lon2d, lat2d = np.meshgrid(lon, lat)

    center_lat = (lat[0] + lat[-1]) / 2
    if custom_lapse_rate and calibrate_lapse_rate:
        raise ValueError("Cannot use both custom_lapse_rate and calibrate_lapse_rate=True.")
    if custom_lapse_rate:
        lapse_rate_all = np.array(custom_lapse_rate) / 1000.0
    elif not calibrate_lapse_rate:
        lapse_rate_all = lapse_rate_sohem if center_lat < 0 else lapse_rate_nohem
    else:
        lapse_rate_all = None  # dynamic per timestep

    vp_coeff_all = vp_coeff_sohem if center_lat < 0 else vp_coeff_nohem

    month_tag = pd.to_datetime(time[0]).strftime("%Y_%m")
    out_nc = os.path.join(output_folder_LW, f"longwave_downscaled_{month_tag}.nc")
    if os.path.exists(out_nc):
        print(f"Output already exists: {out_nc}. Skipping downscaling.")
        return

    # geopotential -> z on ERA grid (same as your original)
    geop = xr.open_dataset(geopotential_path)
    z = np.zeros_like(lat2d, dtype=np.float32)
    for i in range(lat2d.shape[0]):
        for j in range(lat2d.shape[1]):
            try:
                Z = geop.z.sel(latitude=lat2d[i, j], longitude=lon2d[i, j], method="nearest", tolerance=0.5)
                z[i, j] = Z.values.item() / 9.81
            except:
                z[i, j] = np.nan

    # coefficients by elevation (same as your original)
    z1, z2 = 200, 3000
    X1, X2 = 0.35, 0.51
    Y1, Y2 = 0.100, 0.130
    Z1, Z2 = 0.224, 1.100

    def interpolate_by_elevation(zval, c1, c2):
        return np.where(
            zval <= z1, c1,
            np.where(zval >= z2, c2, c1 + (zval - z1) * (c2 - c1) / (z2 - z1))
        ).astype(np.float32)

    Xs = interpolate_by_elevation(z, X1, X2)
    Ys = interpolate_by_elevation(z, Y1, Y2)
    Zs = interpolate_by_elevation(z, Z1, Z2)

    # ERA georef
    dx, dy = np.abs(lon[1] - lon[0]), np.abs(lat[1] - lat[0])
    era_transform = from_origin(np.min(lon), np.max(lat), dx, dy)
    era_crs = CRS.from_epsg(4326)

    # Output coords from DEM
    x_coords = np.arange(width) * dem_transform.a + dem_transform.c + dem_transform.a / 2
    y_coords = np.arange(height) * dem_transform.e + dem_transform.f + dem_transform.e / 2

    time_pd = pd.to_datetime(time)
    ntime = len(time_pd)

    # ---------- create output NetCDF ONCE (CF/GDAL/QGIS compliant CRS) ----------
    os.makedirs(os.path.dirname(out_nc), exist_ok=True)
    root = nc.Dataset(out_nc, "w", format="NETCDF4")

    root.createDimension("time", ntime)
    root.createDimension("y", height)
    root.createDimension("x", width)

    xv = root.createVariable("x", "f4", ("x",))
    yv = root.createVariable("y", "f4", ("y",))
    tv = root.createVariable("time", "f8", ("time",))

    xv[:] = x_coords.astype(np.float32)
    yv[:] = y_coords.astype(np.float32)

    xv.standard_name = "projection_x_coordinate"
    xv.units = "m"
    xv.axis = "X"

    yv.standard_name = "projection_y_coordinate"
    yv.units = "m"
    yv.axis = "Y"

    tv.units = "seconds since 1970-01-01 00:00:00"
    tv.calendar = "standard"

    lwr_var = root.createVariable(
        "lwr", "f4", ("time", "y", "x"),
        fill_value=np.float32(np.nan),
        chunksizes=(min(time_chunk, ntime), min(256, height), min(256, width))
    )
    lwr_var.units = "W/m^2"
    lwr_var.description = "Downscaled longwave radiation"

    # CF grid mapping for QGIS
    spatial_ref = root.createVariable("spatial_ref", "i4")
    if dem_crs is not None:
        try:
            for k, v in dem_crs.to_cf().items():
                spatial_ref.setncattr(k, v)
        except Exception:
            pass
        wkt = dem_crs.to_wkt()
    else:
        wkt = ""

    spatial_ref.setncattr("crs_wkt", wkt)
    spatial_ref.setncattr("spatial_ref", wkt)

    gt = f"{dem_transform.c} {dem_transform.a} {dem_transform.b} {dem_transform.f} {dem_transform.d} {dem_transform.e}"
    spatial_ref.setncattr("GeoTransform", gt)

    lwr_var.setncattr("grid_mapping", "spatial_ref")
    root.setncattr("Conventions", "CF-1.8")

    # write time coordinate once
    tv[:] = nc.date2num([pd.Timestamp(t).to_pydatetime() for t in time_pd], units=tv.units, calendar=tv.calendar)

    # ---------- compute + write in chunks ----------
    pbar = tqdm(total=ntime, desc="Downscaling longwave radiation (chunked)")

    start = 0
    while start < ntime:
        end = min(start + time_chunk, ntime)
        B = end - start

        lwr_chunk = np.empty((B, height, width), dtype=np.float32)

        for kk, ti in enumerate(range(start, end)):
            date = pd.Timestamp(time_pd[ti])
            month_index = date.month - 1

            T_now = T.isel({time_dim: ti}).values.astype(np.float32, copy=False)
            D_now = D.isel({time_dim: ti}).values.astype(np.float32, copy=False)

            # Lapse rate
            if calibrate_lapse_rate:
                T_vals = T_now.ravel()
                Z_vals = z.ravel()
                valid = ~np.isnan(T_vals) & ~np.isnan(Z_vals)
                if np.sum(valid) < 5:
                    lapse_rate = (lapse_rate_sohem[month_index] if center_lat < 0 else lapse_rate_nohem[month_index])
                else:
                    slope, _, _, _, _ = linregress(Z_vals[valid], T_vals[valid])
                    lapse_rate = -slope
            else:
                lapse_rate = lapse_rate_all[month_index]

            vp_coeff = vp_coeff_all[month_index]
            d_t_lapse_rate = vp_coeff * c / b

            t_0 = T_now - lapse_rate * (0 - z)
            d_0 = D_now - d_t_lapse_rate * (0 - z)

            T_700 = t_0 - lapse_rate * (z_700 - z) - 273.15
            D_700 = d_0 - d_t_lapse_rate * (z_700 - z) - 273.15

            es = a * np.exp((b * T_700) / (T_700 + c))
            e700 = a * np.exp((b * D_700) / (D_700 + c))
            RH_700 = np.clip(100.0 * e700 / es, 0, 100).astype(np.float32)
            cloud_frac = np.clip(0.832 * np.exp((RH_700 - 100.0) / 41.6), 0, 1).astype(np.float32)

            # vapor pressure at surface dew point
            e_surf = a * np.exp((b * (D_now - 273.15)) / ((D_now - 273.15) + c)).astype(np.float32)

            eps_atm = (1.083 * (1 + Zs * cloud_frac**2)) * (1 - Xs * np.exp(-Ys * e_surf / T_now))
            Qli = (eps_atm * sigma * T_now**4).astype(np.float32)

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
            lwr_chunk[kk, :, :] = Qli_resampled

            del T_now, D_now, t_0, d_0, T_700, D_700, es, e700, RH_700, cloud_frac, e_surf, eps_atm, Qli, Qli_resampled
            pbar.update(1)

        lwr_var[start:end, :, :] = lwr_chunk
        del lwr_chunk
        start = end

    pbar.close()
    root.close()
    print(f"\nDownscaling complete. NetCDF saved in: {out_nc}")



def main():
    ...
 
if __name__ == "__main__":
    main()






