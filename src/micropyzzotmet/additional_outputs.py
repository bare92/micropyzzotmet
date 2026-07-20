#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Additional output-generation utilities."""

from __future__ import annotations

import glob
import gzip
import os
import shutil

import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from affine import Affine
from joblib import Parallel, delayed
from pyproj import CRS as PyCRS
from pyproj import Transformer
from rasterio.crs import CRS
from rasterio.warp import Resampling, calculate_default_transform, reproject


def compute_extraterrestrial_radiation(latitude_deg, day_of_year):
    """
    Compute daily extraterrestrial radiation (Ra) in MJ m-2 day-1.

    The implementation follows FAO-56 daily formulation.

    Parameters
    ----------
    latitude_deg : float, numpy.ndarray, or xarray.DataArray
        Latitude in decimal degrees.
    day_of_year : int, numpy.ndarray, or xarray.DataArray
        Day of year (1..366).

    Returns
    -------
    same type as input broadcasting
        Daily extraterrestrial radiation in MJ m-2 day-1.
    """
    gsc = 0.0820  # MJ m-2 min-1

    # Work entirely in numpy to avoid per-operation xarray overhead.
    phi = np.deg2rad(np.asarray(latitude_deg, dtype=np.float64))
    j   = np.asarray(day_of_year, dtype=np.float64)

    factor = 2.0 * np.pi / 365.0
    dr    = 1.0 + 0.033 * np.cos(factor * j)
    delta = 0.409 * np.sin(factor * j - 1.39)

    # Precompute spatial-only trig once (not repeated per time step).
    sin_phi = np.sin(phi)
    cos_phi = np.cos(phi)
    tan_phi = np.tan(phi)

    if j.ndim >= 1 and phi.ndim >= 2:
        # (T,) × (Y, X) → (T, Y, X): expand axes explicitly so numpy
        # broadcasts without xarray name-based alignment overhead.
        ws_arg_c = np.clip(
            -tan_phi[np.newaxis] * np.tan(delta)[:, np.newaxis, np.newaxis],
            -1.0, 1.0,
        )
        ws     = np.arccos(ws_arg_c)
        # sin(arccos(x)) = sqrt(1 - x²): replaces a full sin() on the 3-D array.
        sin_ws = np.sqrt(np.maximum(1.0 - ws_arg_c ** 2, 0.0))
        sin_d  = np.sin(delta)[:, np.newaxis, np.newaxis]
        cos_d  = np.cos(delta)[:, np.newaxis, np.newaxis]
        ra = (
            (24.0 * 60.0 / np.pi) * gsc * dr[:, np.newaxis, np.newaxis]
            * (
                ws * sin_phi[np.newaxis] * sin_d
                + cos_phi[np.newaxis] * cos_d * sin_ws
            )
        )
    else:
        # Scalar / 1-D latitude: simple numpy broadcast.
        ws_arg_c = np.clip(-tan_phi * np.tan(delta), -1.0, 1.0)
        ws     = np.arccos(ws_arg_c)
        sin_ws = np.sqrt(np.maximum(1.0 - ws_arg_c ** 2, 0.0))
        ra = (
            (24.0 * 60.0 / np.pi) * gsc * dr
            * (ws * sin_phi * np.sin(delta) + cos_phi * np.cos(delta) * sin_ws)
        )
    return ra


def compute_reference_evapotranspiration(
    t_min_c,
    t_max_c,
    latitude_deg,
    day_of_year,
    method="hargreaves_samani",
    t_mean_c=None,
):
    """
        Compute reference evapotranspiration (ET0, mm/day) using Hargreaves-Samani.

        Equation
        --------
        ET0 = 0.0023 * Ra * (Tmean + 17.8) * sqrt(Tmax - Tmin)

    Parameters
    ----------
    t_min_c : float, numpy.ndarray, or xarray.DataArray
        Daily minimum air temperature in degC.
    t_max_c : float, numpy.ndarray, or xarray.DataArray
        Daily maximum air temperature in degC.
    latitude_deg : float, numpy.ndarray, or xarray.DataArray
        Latitude in decimal degrees.
    day_of_year : int, numpy.ndarray, or xarray.DataArray
        Day of year (1..366).
    method : str, default "hargreaves_samani"
        Method selector. Only HS/Hargreaves-Samani is supported.
    t_mean_c : float, numpy.ndarray, or xarray.DataArray, optional
        Daily mean air temperature in degC. If None, computed as (Tmax + Tmin)/2.

    Returns
    -------
    same type as input broadcasting
        ET0 in mm/day.
    """
    method_key = method.strip().lower()
    aliases = {
        "hargreaves": "hargreaves_samani",
        "hargreaves_samani": "hargreaves_samani",
        "hs": "hargreaves_samani",
    }
    if method_key not in aliases:
        raise ValueError("Unsupported ET method. Only HS (Hargreaves-Samani) is supported")

    # Extract to numpy early so all intermediate operations are plain numpy
    # (avoids xarray per-operation overhead for large DataArray inputs).
    is_da   = hasattr(t_min_c, "dims")
    t_min   = np.asarray(t_min_c)
    t_max   = np.asarray(t_max_c)
    t_range = np.maximum(t_max - t_min, 0.0)
    t_mean  = (t_max + t_min) / 2.0 if t_mean_c is None else np.asarray(t_mean_c)

    ra  = compute_extraterrestrial_radiation(latitude_deg=latitude_deg, day_of_year=day_of_year)
    et0 = 0.0023 * ra * (t_mean + 17.8) * np.sqrt(t_range)
    result = np.maximum(et0, 0.0)

    # Re-wrap as DataArray only at the boundary so the caller's .attrs access works.
    if is_da:
        return xr.DataArray(result, dims=t_min_c.dims, coords=t_min_c.coords)
    return result


def _month_tag_from_filename(path, prefix):
    base = os.path.basename(path)
    if not base.startswith(prefix) or not base.endswith(".nc"):
        return None
    return base[len(prefix):-3]


def _build_month_file_map(folder, prefix):
    month_map = {}
    for path in sorted(glob.glob(os.path.join(folder, f"{prefix}*.nc"))):
        tag = _month_tag_from_filename(path, prefix)
        if tag:
            month_map[tag] = path
    return month_map


def _to_celsius(temp_da):
    units = str(temp_da.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin"}:
        out = temp_da - 273.15
        out.attrs = dict(temp_da.attrs)
        out.attrs["units"] = "degC"
        return out
    return temp_da


def _latitude_grid_from_dataset(ds):
    if "spatial_ref" not in ds.variables:
        raise RuntimeError("Missing 'spatial_ref' variable needed to compute latitude grid for PET")

    wkt = ds["spatial_ref"].attrs.get("crs_wkt") or ds["spatial_ref"].attrs.get("spatial_ref")
    if not wkt:
        raise RuntimeError("Missing CRS WKT metadata in 'spatial_ref' needed to compute PET")

    src_crs = PyCRS.from_wkt(wkt)
    transformer = Transformer.from_crs(src_crs, PyCRS.from_epsg(4326), always_xy=True)

    x2d, y2d = np.meshgrid(ds["x"].values, ds["y"].values)
    _, lat2d = transformer.transform(x2d, y2d)
    return lat2d.astype(np.float32)


def generate_monthly_potential_evapotranspiration(working_directory, method="HS"):
    """
    Generate monthly PET NetCDF outputs in outputs/ET_{method}.

    Inputs are read from existing downscaled output folders:
    - Temperature_min (t_min)
    - Temperature_max (t_max)
    """
    method_key = str(method).strip().upper()
    if method_key != "HS":
        raise ValueError(f"Unsupported PET method '{method}'. Only HS is supported")

    outputs_root = os.path.join(working_directory, "outputs")
    folder_tmin = os.path.join(outputs_root, "Temperature_min")
    folder_tmax = os.path.join(outputs_root, "Temperature_max")

    tmin_map = _build_month_file_map(folder_tmin, "temperature_min_downscaled_")
    tmax_map = _build_month_file_map(folder_tmax, "temperature_max_downscaled_")
    common_tags = sorted(set(tmin_map) & set(tmax_map))

    if not common_tags:
        raise RuntimeError(
            "No common monthly t_min/t_max files found. "
            "Expected outputs in Temperature_min and Temperature_max folders."
        )

    out_folder = os.path.join(outputs_root, f"ET_{method_key}")
    os.makedirs(out_folder, exist_ok=True)

    for tag in common_tags:
        out_nc = os.path.join(out_folder, f"potential_evapotranspiration_{tag}.nc")
        if os.path.exists(out_nc):
            print(f"Output already exists: {out_nc}. Skipping PET.")
            continue

        with xr.open_dataset(tmin_map[tag]) as ds_tmin, xr.open_dataset(tmax_map[tag]) as ds_tmax:
            if "t_min" not in ds_tmin or "t_max" not in ds_tmax:
                raise RuntimeError("Expected t_min and t_max variables in monthly temperature files")

            tmin = _to_celsius(ds_tmin["t_min"]).astype(np.float32)
            tmax = _to_celsius(ds_tmax["t_max"]).astype(np.float32)

            time_coord = "time" if "time" in tmin.dims else "valid_time"
            if time_coord != "time":
                tmin = tmin.rename({time_coord: "time"})
                tmax = tmax.rename({time_coord: "time"})

            if "time" not in tmin.coords:
                raise RuntimeError("Temperature files must include a time coordinate for PET")

            day_of_year = xr.DataArray(
                pd.to_datetime(tmin["time"].values).dayofyear,
                dims=("time",),
                coords={"time": tmin["time"].values},
            )
            latitude = xr.DataArray(
                _latitude_grid_from_dataset(ds_tmin),
                dims=("y", "x"),
                coords={"y": ds_tmin["y"].values, "x": ds_tmin["x"].values},
            )

            pet = compute_reference_evapotranspiration(
                t_min_c=tmin,
                t_max_c=tmax,
                latitude_deg=latitude,
                day_of_year=day_of_year,
                method="hargreaves_samani",
            ).astype(np.float32)

            pet.attrs["units"] = "mm day-1"
            pet.attrs["description"] = f"Potential evapotranspiration ({method_key})"

            ds_out = xr.Dataset({"PET": pet})
            if "spatial_ref" in ds_tmin.variables:
                ds_out["spatial_ref"] = ds_tmin["spatial_ref"]
                ds_out["PET"].attrs["grid_mapping"] = "spatial_ref"
            ds_out.attrs["pet_method"] = method_key

            ds_out.to_netcdf(out_nc)

    print(f"PET monthly NetCDF export complete: {out_folder}")


def convert_micromet_to_s3m_inputs(
    micromet_output_dir: str,
    output_dir: str,
    dem_path: str,
    nodata_value: float = -9999.0,
    n_jobs: int = 4,
):
    """
    Convert Micromet downscaled outputs into S3M forcing files (one NetCDF per timestep, gzipped).

    This utility prepares meteorological forcing files for the S3M Fortran model by:
      1. Reprojecting the DEM to WGS84 (EPSG:4326) on a regular grid.
      2. Building an S3M-oriented grid where:
         - x increases eastward,
         - y increases northward, and
         - row 0 corresponds to the southernmost row (south -> north ordering).
      3. Reading Micromet output NetCDFs for each variable (possibly split across multiple files),
         extracting a single 2D slice per timestep.
      4. Reprojecting each 2D field from the Micromet grid/CRS to the S3M WGS84 grid using
         nearest-neighbor resampling.
      5. Writing one NetCDF per timestep with S3M-required variables:
         Rain, AirTemperature, IncRadiation, RelHumidity, plus static fields
         terrain, longitude, latitude.
      6. Compressing each NetCDF to .nc.gz and removing the uncompressed file.

    The function expects Micromet NetCDFs to be CF-like and include CRS metadata via a
    spatial_ref variable with a crs_wkt (or spatial_ref) attribute. Pixel georeferencing
    for Micromet inputs is reconstructed from 1D x and y coordinate vectors interpreted as
    pixel centers.
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- Load and reproject DEM to WGS84 (EPSG:4326) ---
    with rasterio.open(dem_path) as src:
        terrain_src = src.read(1).astype(np.float32)
        src_crs = src.crs
        src_transform = src.transform
        src_width, src_height = src.width, src.height
        bounds = src.bounds

    dst_crs = CRS.from_epsg(4326)
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *bounds
    )

    terrain = np.full((dst_height, dst_width), nodata_value, dtype=np.float32)
    reproject(
        source=terrain_src,
        destination=terrain,
        src_transform=src_transform,
        src_crs=src_crs,
        dst_transform=dst_transform,
        dst_crs=dst_crs,
        resampling=Resampling.bilinear,
    )

    # --- Build grid in S3M convention: row 0 = southernmost, y increases northward ---
    cellsize = float(abs(dst_transform.a))
    xllcorner = float(dst_transform.c)
    yllcorner = float(dst_transform.f + dst_height * dst_transform.e)

    x_coords = xllcorner + (np.arange(dst_width) + 0.5) * cellsize
    y_coords = yllcorner + (np.arange(dst_height) + 0.5) * cellsize

    # terrain from reproject() is north->south; flip to south->north
    terrain = terrain[::-1, :]

    lon2d, lat2d = np.meshgrid(x_coords, y_coords)

    # --- Map Micromet folder/var -> S3M var name ---
    var_map = {
        "Temperature": ("t2m", "AirTemperature"),
        "SW": ("SW", "IncRadiation"),
        "RH": ("RH", "RelHumidity"),
        "P": ("P", "Rain"),
    }

    var_paths = {}
    time_index = None

    for folder, (var_in, var_out) in var_map.items():
        paths = sorted(glob.glob(os.path.join(micromet_output_dir, folder, "*.nc")))
        if not paths:
            continue

        ds0 = xr.open_dataset(paths[0])
        if "time" not in ds0 and "valid_time" not in ds0:
            ds0.close()
            raise RuntimeError(f"No time coordinate found in {paths[0]}")
        ds0.close()

        if time_index is None:
            ds_time = xr.open_mfdataset(paths, combine="by_coords", chunks={})
            tcoord = "time" if "time" in ds_time.coords else ("valid_time" if "valid_time" in ds_time.coords else None)
            if tcoord is None:
                ds_time.close()
                raise RuntimeError(f"No time coordinate found across {folder} outputs")
            time_index = pd.to_datetime(ds_time[tcoord].values)
            ds_time.close()

        var_paths[var_out] = (paths, var_in)

    if time_index is None or len(time_index) == 0:
        raise RuntimeError("No time data found in Micromet outputs")

    def _get_da_crs_from_cf(ds):
        if "spatial_ref" in ds.variables:
            wkt = ds["spatial_ref"].attrs.get("crs_wkt", None) or ds["spatial_ref"].attrs.get("spatial_ref", None)
            if wkt:
                return CRS.from_wkt(wkt)
        wkt = ds.attrs.get("crs_wkt", None)
        if wkt:
            return CRS.from_wkt(wkt)
        return None

    def _affine_from_xy_centers(x, y):
        x = np.asarray(x)
        y = np.asarray(y)
        if x.size < 2 or y.size < 2:
            raise ValueError("Need at least 2 x and 2 y coordinates to build affine transform.")
        dx = float(x[1] - x[0])
        dy = float(y[1] - y[0])
        return Affine.translation(float(x[0]) - dx / 2.0, float(y[0]) - dy / 2.0) * Affine.scale(dx, dy)

    def _reproject_to_s3m_grid(src2d, src_transform_local, src_crs_local):
        dst = np.full((dst_height, dst_width), np.nan, dtype=np.float32)
        reproject(
            source=src2d.astype(np.float32, copy=False),
            destination=dst,
            src_transform=src_transform_local,
            src_crs=src_crs_local,
            dst_transform=dst_transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest,
        )
        dst = dst[::-1, :]
        dst[np.isnan(dst)] = nodata_value
        return dst

    def _write_one_s3m_file(i, t):
        date_str = pd.to_datetime(t).strftime("%Y%m%d%H%M")
        filename_nc = os.path.join(output_dir, f"MeteoData_{date_str}.nc")
        filename_gz = filename_nc + ".gz"

        if os.path.exists(filename_gz):
            return

        data_vars = {}

        for var_name in ["Rain", "AirTemperature", "IncRadiation", "RelHumidity"]:
            if var_name in var_paths:
                paths, var_in = var_paths[var_name]

                ds_in = xr.open_mfdataset(paths, combine="by_coords", chunks={})
                _time_chunks = {d: 1 for d in ds_in.dims if d in ("time", "valid_time")}
                ds_in = ds_in.chunk(_time_chunks)

                if "time" in ds_in.coords:
                    tcoord = "time"
                elif "valid_time" in ds_in.coords:
                    tcoord = "valid_time"
                else:
                    ds_in.close()
                    raise RuntimeError(f"No time coordinate in dataset for {var_name}")

                da2d = ds_in[var_in].isel({tcoord: i})

                if "x" not in da2d.coords or "y" not in da2d.coords:
                    ds_in.close()
                    raise RuntimeError(f"{var_name} missing x/y coordinates; cannot reproject.")

                src_crs_local = _get_da_crs_from_cf(ds_in)
                if src_crs_local is None:
                    ds_in.close()
                    raise RuntimeError(f"Could not determine CRS for {var_name} from CF 'spatial_ref'.")

                src_transform_local = _affine_from_xy_centers(da2d["x"].values, da2d["y"].values)
                src2d = da2d.values
                data = _reproject_to_s3m_grid(src2d, src_transform_local, src_crs_local)

                ds_in.close()

                if var_name == "AirTemperature":
                    mask = data != nodata_value
                    data[mask] = data[mask] - 273.15
            else:
                data = np.full((dst_height, dst_width), nodata_value, dtype=np.float32)

            data_vars[var_name] = xr.DataArray(
                data.astype(np.float32, copy=False),
                dims=("y", "x"),
                coords={"x": x_coords, "y": y_coords},
                attrs={"coordinates": "longitude latitude"},
            )

        data_vars["terrain"] = xr.DataArray(
            terrain.astype(np.float32, copy=False),
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
            attrs={"coordinates": "longitude latitude"},
        )
        data_vars["longitude"] = xr.DataArray(
            lon2d.astype(np.float32, copy=False),
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
        )
        data_vars["latitude"] = xr.DataArray(
            lat2d.astype(np.float32, copy=False),
            dims=("y", "x"),
            coords={"x": x_coords, "y": y_coords},
        )

        ds_out = xr.Dataset(data_vars)
        ds_out.attrs.update(
            {
                "ncols": int(dst_width),
                "nrows": int(dst_height),
                "nodata_value": float(nodata_value),
                "xllcorner": float(xllcorner),
                "yllcorner": float(yllcorner),
                "cellsize": float(cellsize),
                "crs": "EPSG:4326",
            }
        )

        ds_out.to_netcdf(filename_nc)

        with open(filename_nc, "rb") as f_in, gzip.open(filename_gz, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)

        os.remove(filename_nc)

    Parallel(n_jobs=n_jobs)(
        delayed(_write_one_s3m_file)(i, t) for i, t in enumerate(time_index)
    )

    print(f"\nS3M .nc.gz export complete: {output_dir}")
