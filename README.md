# MicroPyzzotMet

**MicroPyzzotMet** is a Python package for downscaling meteorological variables over complex terrain using a high-resolution DEM and reanalysis forcing (currently focused on ERA5-Land via EarthDataHub). It implements a lightweight, MicroMet-inspired workflow for generating distributed atmospheric forcing fields for snow, cryosphere, hydrology, and mountain-environment applications.

Documentation is available in the repository under `docs/` and through Read the Docs.

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
├── CONTRIBUTING.md
├── JOSS_RESUBMISSION_GUIDE.md
├── LICENSE.txt
├── auto_run_Paloma.py
├── auxiliary_data/
│   └── geopotential3.nc
├── docs/
├── JOSS/
├── option_files/
│   ├── micro_config_DEMO_MAIPO.json
│   └── micro_config_alps.json
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

### Install from PyPI

If a public release is available on PyPI, you can install it with:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install micropyzzotmet
```

At the moment, if `pip install micropyzzotmet` returns `No matching distribution found`, the package is not yet available on the public PyPI index for your environment, and you should use the source installation route below.

### Install from source

This is the currently reliable installation path for this repository.

Create a virtual environment, activate it, and install the package:

```bash
# Create a virtual environment
python3 -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install the package
pip install -e .
```

You can verify that the CLI is available with:

```bash
micropyzzotmet --help
```

### Optional extras

If you are working from a local clone and want development or documentation tools, use editable installation with extras.

For development tools:

```bash
pip install -e ".[dev]"
```

For documentation tools:

```bash
pip install -e ".[docs]"
```

### Environment requirements

For full production runs, make sure the environment also provides:

- Python 3.10+
- GDAL command-line tools (`gdaldem`)
- rasterio / rioxarray / pyproj
- xarray / zarr / netCDF4
- joblib / tqdm
- pvlib
- fsspec / s3fs / dask (required for EarthDataHub Zarr access)

### EarthDataHub credentials

MicroPyzzotMet downloads ERA5-Land and DEM data from [EarthDataHub](https://earthdatahub.destine.eu). You need a personal access token (PAT).

You can provide it in two ways:

**Option A — in the config file** (recommended):
```json
"earthdatahub_pat": "<YOUR_EDH_PAT_HERE>"
```

**Option B — via `~/.netrc`**:
```bash
cat > ~/.netrc << 'EOF'
machine earthdatahub.com
login YOUR_EDH_USERNAME
password YOUR_EDH_PASSWORD
EOF
chmod 600 ~/.netrc
```

> **Important:** never commit a real EarthDataHub PAT or `~/.netrc` credentials into version control.

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

MicroPyzzotMet is driven by a JSON config file. Two example configs are included in `option_files/`:

- `option_files/micro_config_DEMO_MAIPO.json`
- `option_files/micro_config_alps.json`

A typical config includes:

- `working_directory`: root folder where `inputs/` and `outputs/` are created
- `dem_file`: path to an existing DEM, or `null` to trigger DEM download
- `download_dem_extent`, `download_dem_epsg`, `download_dem_resolution`, `output_filename_dem`: DEM download settings used when `dem_file` is `null`
- `era_file`: currently used as a switch to skip automatic ERA5-Land download
- `earthdatahub_pat`: EarthDataHub personal access token (alternative to `~/.netrc`)
- `earthdatahub_machine`: netrc machine name (default: `"earthdatahub.com"`, optional)
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
    "precipitation": "y",
    "wind": "y",
    "lw_radiation": "y"
  },
  "start_date": "2017-04-01",
  "end_date": "2017-07-31",
  "aggregate_daily": "y",
  "time_chunk": 24,
  "dem_nodata": -32768,
  "generate_s3m_input": "y",
  "custom_lapse_rates": {
    "temperature": {
      "monthly": [8.1, 7.9, 7.78, 7.76, 7.9, 8.0, 8.2, 8.4, 8.6, 8.7, 8.4, 8.32]
    },
    "precipitation": {
      "monthly": null
    }
  },
  "jobs_parallel_downscale": 4,
  "jobs_parallel_download": 4
}
```

### 4. Run the workflow

From the repository root:

```bash
micropyzzotmet option_files/micro_config_DEMO_MAIPO.json
```

You can also point to any other JSON config:

```bash
micropyzzotmet path/to/your_config.json
```

## Usage examples

### Example 1: Run an included example configuration

```bash
micropyzzotmet option_files/micro_config_DEMO_MAIPO.json
```

This runs the full workflow using the example Maipo configuration.

### Example 2: Run your own configuration file

```bash
micropyzzotmet /absolute/path/to/config.json
```

Use this when your project configuration is stored outside the repository.

### Example 3: Check that the CLI is installed correctly

```bash
micropyzzotmet --help
```

This is the fastest way to confirm that installation completed successfully and that the console entry point is available.

### Example 4: Install and use the released version from PyPI

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install micropyzzotmet
micropyzzotmet /path/to/config.json
```

Use this only when a public PyPI release is available.
It is intended for supported Python versions only.

### Example 5: Development workflow from a local clone

```bash
git clone https://github.com/bare92/micropyzzotmet.git
cd micropyzzotmet
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
micropyzzotmet option_files/micro_config_DEMO_MAIPO.json
```

This is the recommended route if you want to modify code, run tests, or contribute changes.

---

## Current workflow behavior

A few details are worth knowing for the current codebase:

### Run from the repository root

The current temperature downscaling code reads the auxiliary geopotential file from:

```text
./auxiliary_data/geopotential3.nc
```

For that reason, the safest way to run the package at the moment is **from the repository root**, not from an arbitrary working directory.

This means that, even if a future PyPI installation path is available, some workflows may still be easiest when launched from a repository checkout until all repository-relative paths are fully removed from the runtime code.

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

A minimal shell launcher using the **current CLI** and `.venv` looks like this:

```bash
#!/bin/bash
set -e

source /path/to/micropyzzotmet/.venv/bin/activate
cd /path/to/micropyzzotmet
micropyzzotmet option_files/micro_config_DEMO_MAIPO.json
```

---

## Documentation

Sphinx documentation sources are included in `docs/source/`.

Hosted documentation is configured through Read the Docs.

To build the docs locally:

```bash
pip install -e ".[docs]"
make -C docs html
```

To run tests locally:

```bash
pip install -e ".[dev]"
pytest
```

The repository also includes GitHub Actions CI and a Read the Docs configuration file:

- `.github/workflows/tests.yml`
- `.github/workflows/publish.yml`
- `.readthedocs.yml`

---

## Reference

If this package is relevant to your work, please also cite the original MicroMet paper:

- Liston, G. E., & Elder, K. (2006). *A Meteorological Distribution System for High-Resolution Terrestrial Modeling (MicroMet).* Journal of Hydrometeorology, 7(2), 217-234. https://doi.org/10.1175/JHM486.1

---

## Contact

For questions, bug reports, or collaboration, open an issue on the repository or contact the maintainer through the project GitHub page.
