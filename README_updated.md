# MicroPyzzotMet

**MicroPyzzotMet** is a Python package for downscaling meteorological variables over complex terrain using a high-resolution DEM and reanalysis forcing (currently focused on ERA5-Land via EarthDataHub). It implements a lightweight, MicroMet-inspired workflow for generating distributed atmospheric forcing fields for snow, cryosphere, hydrology, and mountain-environment applications.

The project now uses a **packaged `src/` layout** and an installable **CLI entrypoint**:

```bash
micropyzzotmet path/to/config.json
```

---

## What the package does

MicroPyzzotMet can:

- create a standard project folder structure under a user-defined working directory
- use an existing DEM or automatically download a Copernicus DEM subset
- derive terrain layers such as slope, aspect, and curvature
- download and spatially subset ERA5-Land data from EarthDataHub
- downscale selected meteorological variables to the DEM grid
- optionally export S3M-compatible forcing files

Supported downscaling modules currently include:

- air temperature
- shortwave radiation
- relative humidity
- precipitation
- wind
- longwave radiation

Outputs are written as monthly NetCDF files inside the working directory.

---

## Repository layout

```text
micropyzzotmet/
├── pyproject.toml
├── README.md
├── LICENSE.txt
├── micro_config_DEMO_MAIPO.json
├── micro_config_alps.json
├── setup_micropyzzotmet_env.sh
├── run_micromet_DEMO_MAIPO.sh
├── run_micromet_alps.sh
├── auxiliary_data/
│   └── geopotential3.nc
├── docs/
├── JOSS/
└── src/
    └── micropyzzotmet/
        ├── __init__.py
        ├── cli.py
        ├── main_micromet.py
        ├── get_era5_land.py
        ├── downscaling_variables.py
        └── utils.py
```

---

## Installation

### Recommended: create the virtual environment, then install the package

The safest way to run the full workflow is to use the provided setup script, because the code relies on a geospatial/scientific stack that includes GDAL, rasterio, xarray, zarr, dask/fsspec-related tooling, and command-line GDAL utilities such as `gdaldem`.

From the repository root:

```bash
chmod +x setup_micropyzzotmet_env.sh
./setup_micropyzzotmet_env.sh
```

Then activate the environment and install the package in editable mode:

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate microenv
pip install -e .
```

You can check that the CLI is available with:

```bash
micropyzzotmet --help
```

### Alternative: install into an existing environment

If you already have a working Python/geospatial environment, you can install directly from `pyproject.toml`:

```bash
pip install -e .
```

This route is best for development. For full production runs, make sure the environment also provides:

- Python 3.10+
- GDAL command-line tools (`gdaldem`)
- rasterio / rioxarray / pyproj
- xarray / zarr / netCDF4
- joblib / tqdm
- pvlib
- any additional cloud/Zarr dependencies required by the EarthDataHub workflow

---

## Quick start

### 1. Clone the repository

```bash
git clone https://github.com/bare92/micropyzzotmet.git
cd micropyzzotmet
```

### 2. Install the package

Use one of the installation methods above, then confirm the CLI exists:

```bash
micropyzzotmet --help
```

### 3. Prepare a configuration file

MicroPyzzotMet is driven by a JSON config file. Two example configs are included:

- `micro_config_DEMO_MAIPO.json`
- `micro_config_alps.json`

A typical config includes:

- `working_directory`: root folder where `inputs/` and `outputs/` are created
- `dem_file`: path to an existing DEM, or `null` to trigger DEM download
- `download_dem_extent`, `download_dem_epsg`, `download_dem_resolution`, `output_filename_dem`: DEM download settings used when `dem_file` is `null`
- `era_file`: currently used as a switch to skip automatic ERA5-Land download
- `earthdatahub_pat`: EarthDataHub personal access token
- `variables_to_downscale`: `"y"` / `"n"` flags for each variable
- `start_date`, `end_date`: run period
- `aggregate_daily`: whether ERA5-Land inputs are aggregated to daily values
- `time_chunk`: block size used by chunked downscaling/writing
- `dem_nodata`: no-data value for the DEM
- `custom_lapse_rates`: optional temperature / precipitation monthly values
- `jobs_parallel_downscale`: number of parallel jobs for variable processing
- `jobs_parallel_download`: download-parallelism setting exposed in the config
- `generate_s3m_input`: optional S3M export switch

A sanitized example:

```json
{
  "working_directory": "../DEMO_micromet_outputs",
  "dem_file": null,
  "download_dem_extent": {
    "lat_min": 6205000,
    "lat_max": 6342500,
    "lon_min": 366000,
    "lon_max": 428500
  },
  "download_dem_epsg": 32719,
  "download_dem_resolution": 50,
  "output_filename_dem": "downloaded_dem.tif",
  "era_file": null,
  "earthdatahub_pat": "<YOUR_EDH_PAT_HERE>",
  "variables_to_downscale": {
    "t_air": "y",
    "sw_radiation": "y",
    "relative_humidity": "y",
    "precipitation": "n",
    "wind": "n",
    "lw_radiation": "n"
  },
  "start_date": "2017-04-01",
  "end_date": "2017-07-31",
  "aggregate_daily": "y",
  "time_chunk": 24,
  "dem_nodata": -32768,
  "generate_s3m_input": "n",
  "custom_lapse_rates": {
    "temperature": {
      "monthly": [8.1, 7.9, 7.78, 7.76, 7.9, 8.0, 8.2, 8.4, 8.6, 8.7, 8.4, 8.32]
    },
    "precipitation": {
      "monthly": null
    }
  },
  "jobs_parallel_downscale": 4,
  "jobs_parallel_download": 1
}
```

> **Important:** never commit a real EarthDataHub PAT into version control.

### 4. Run the workflow

From the repository root:

```bash
micropyzzotmet micro_config_DEMO_MAIPO.json
```

You can also point to any other JSON config:

```bash
micropyzzotmet path/to/your_config.json
```

---

## Current workflow behavior

A few details are worth knowing for the current codebase:

### Run from the repository root

The current temperature downscaling code reads the auxiliary geopotential file from:

```text
./auxiliary_data/geopotential3.nc
```

For that reason, the safest way to run the package at the moment is **from the repository root**, not from an arbitrary working directory.

### `era_file` currently acts as a download switch

At present, the code checks whether `era_file` is `null`:

- if `era_file` is `null`, MicroPyzzotMet downloads monthly ERA5-Land files into `working_directory/inputs/climate`
- if `era_file` is **not** `null`, the automatic download step is skipped

The current pipeline then reads climate files from:

```text
<working_directory>/inputs/climate/*.nc
```

So if you skip the download, make sure your climate NetCDF files are already in that folder.

### Legacy helper scripts

The packaged entrypoint is now:

```bash
micropyzzotmet <config.json>
```

Some helper shell scripts in the repository may still use the older pattern:

```bash
python main_micromet.py ...
```

If you use those scripts, update them to call the CLI.

---

## Output structure

MicroPyzzotMet creates a standard folder tree under `working_directory`:

```text
working_directory/
├── inputs/
│   ├── climate/   # Monthly ERA5-Land NetCDF files
│   └── dem/       # DEM + slope / aspect / curvature
└── outputs/
    ├── Temperature/
    ├── SW/
    ├── RH/
    ├── P/
    ├── Wind/
    ├── LW/
    └── s3m/       # Optional, only if generate_s3m_input = "y"
```

Typical monthly outputs are written as NetCDF files inside the variable-specific folders.

---

## Example launcher script

A minimal shell launcher using the **current CLI** looks like this:

```bash
#!/bin/bash
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate microenv

cd /path/to/micropyzzotmet
micropyzzotmet micro_config_DEMO_MAIPO.json
```

---

## Documentation

Sphinx documentation sources are included in `docs/source/`.

To build the docs locally:

```bash
pip install -r docs/requirements.txt
make -C docs html
```

---

## Reference

If this package is relevant to your work, please also cite the original MicroMet paper:

- Liston, G. E., & Elder, K. (2006). *A Meteorological Distribution System for High-Resolution Terrestrial Modeling (MicroMet).* Journal of Hydrometeorology, 7(2), 217-234. https://doi.org/10.1175/JHM486.1

---

## Contact

For questions, bug reports, or collaboration, open an issue on the repository or contact the maintainer through the project GitHub page.
