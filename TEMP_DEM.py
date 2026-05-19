#!/usr/bin/env python3

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin
from netCDF4 import Dataset


# ============================================================
# USER SETTINGS
# ============================================================

INPUT_NC = Path("/home/riccardo/Documents/09_tmp_data_check/Terrain_Data.nc/FULL_DOMAIN_Terrain_Data.nc")

OUTPUT_TIF = Path("/home/riccardo/Documents/09_tmp_data_check/FULL_DOMAIN_Terrain.tif")

TERRAIN_VAR = "Terrain"
LAT_VAR = "Latitude"
LON_VAR = "Longitude"

CRS = "EPSG:4326"

COMPRESS = "lzw"


# ============================================================
# FUNCTIONS
# ============================================================

def get_global_attr(ds, name, default=None):
    if hasattr(ds, name):
        return getattr(ds, name)
    return default


def read_variable(ds, var_name):
    if var_name not in ds.variables:
        raise KeyError(f"Variable not found in NetCDF: {var_name}")

    data = ds.variables[var_name][:]

    if np.ma.isMaskedArray(data):
        fill_value = getattr(ds.variables[var_name], "_FillValue", None)
        if fill_value is None:
            fill_value = np.nan
        data = data.filled(fill_value)

    return np.asarray(data)


def main():
    if not INPUT_NC.exists():
        raise FileNotFoundError(f"Input file not found: {INPUT_NC}")

    with Dataset(INPUT_NC, "r") as ds:
        terrain = read_variable(ds, TERRAIN_VAR).astype(np.float32)

        xllcorner = float(get_global_attr(ds, "xllcorner"))
        yllcorner = float(get_global_attr(ds, "yllcorner"))
        cellsize = float(get_global_attr(ds, "cellsize"))

        ncols = int(get_global_attr(ds, "ncols", terrain.shape[1]))
        nrows = int(get_global_attr(ds, "nrows", terrain.shape[0]))

        nodata = float(get_global_attr(ds, "nodata_value", -9999))

        if terrain.shape != (nrows, ncols):
            raise ValueError(
                f"Terrain shape {terrain.shape} does not match "
                f"nrows/ncols from metadata: {(nrows, ncols)}"
            )

        # Optional orientation check using Latitude / Longitude grids
        if LAT_VAR in ds.variables:
            lat = read_variable(ds, LAT_VAR)
            lat_top_left = lat[0, 0]
            lat_bottom_left = lat[-1, 0]

            # If latitude increases downward, row 0 is south,
            # so flip vertically to make GeoTIFF north-up.
            if lat_top_left < lat_bottom_left:
                print("[INFO] Flipping Terrain vertically")
                terrain = np.flipud(terrain)

        if LON_VAR in ds.variables:
            lon = read_variable(ds, LON_VAR)
            lon_top_left = lon[0, 0]
            lon_top_right = lon[0, -1]

            # If longitude decreases left-to-right, flip horizontally.
            if lon_top_left > lon_top_right:
                print("[INFO] Flipping Terrain horizontally")
                terrain = np.fliplr(terrain)

    # Convert NaN to nodata
    terrain = np.where(np.isfinite(terrain), terrain, nodata).astype(np.float32)

    # GeoTIFF needs upper-left corner.
    # NetCDF metadata gives lower-left corner.
    upper_left_x = xllcorner
    upper_left_y = yllcorner + nrows * cellsize

    transform = from_origin(
        upper_left_x,
        upper_left_y,
        cellsize,
        cellsize,
    )

    OUTPUT_TIF.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        OUTPUT_TIF,
        "w",
        driver="GTiff",
        height=nrows,
        width=ncols,
        count=1,
        dtype="float32",
        crs=CRS,
        transform=transform,
        nodata=nodata,
        compress=COMPRESS,
        tiled=True,
        BIGTIFF="IF_SAFER",
    ) as dst:
        dst.write(terrain, 1)
        dst.set_band_description(1, TERRAIN_VAR)

    print(f"[DONE] Saved GeoTIFF: {OUTPUT_TIF}")


if __name__ == "__main__":
    main()