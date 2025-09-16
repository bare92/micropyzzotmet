# MicroPyzzotMet: A Python Implementation of MicroMet

**MicroPyzzotMet** is a beginner-friendly Python tool that replicates the functionality of the MicroMet model (Liston and Elder, 2006), designed to downscale meteorological variables over complex terrain using high-resolution DEMs and ERA5-Land (will be expanded) data. It produces distributed atmospheric forcing variables suitable for use in models (e.g., snow models like SnowModel, s3m, fsm2..).

---

## Features

* Automatic ERA5-Land data download and spatial subset using EartDatHub functionalities (https://earthdatahub.destine.eu/)
* Modular downscaling of:

  * Air Temperature
  * Shortwave Radiation
  * Relative Humidity
  * Precipitation
  * Wind
  * Longwave Radiation
* Uses custom, default, or computed lapse rates
* Generates monthly NetCDF
* Parallel processing support

---

## Quick Start

### 1. Clone the repository

```bash
git clone [https://github.com/yourusername/micropezzottomet.git](https://github.com/bare92/micropyzzotmet)
cd micropyzzotmet
```

## Environment Setup (via `setup_micropyzzotmet_env.sh`)

You can set up all required dependencies using the provided script:

```bash
# From the project root
chmod +x setup_micropyzzotmet_env.sh
./setup_micropyzzotmet_env.sh
```
Follow any on-screen instructions that the script prints (e.g., activating a conda/virtualenv).

---

### 3. Prepare your config file

Edit or create a JSON file (see example below or `micro_config_MAIPO.json`). This specifies input paths, time range, variables to downscale, and lapse rates.

### 4. Run the model

```bash
python main_micromet.py path/to/your_config.json
```

---

## Directory Structure

The model automatically creates this folder structure inside your `working_directory`:

```
working_directory/
├── inputs/
│   ├── climate/           # ERA5-Land NetCDF files
│   └── dem/               # DEM and derived slope/aspect/curvature
└── outputs/
    ├── Temperature/
    ├── SW/              # Shortwave radiation
    ├── RH/              # Relative humidity
    ├── P/               # Precipitation
    └── Wind/            # Wind speed/direction
    └── LW/              # Longwave Radiation
```

---

## Configuration File Structure (based on `micro_config_MAIPO_AUTODEM.json`)

This project reads a JSON configuration. The structure mirrors the example below (values here are examples; replace with your own as needed). Sensitive fields are shown as placeholders.

```json
{
  "working_directory": "/mnt/CEPH_PROJECTS/SNOWCOP/Riccardo/MAIPO_Downscaled_micromet",
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
  "earthdatahub_pat": "<provide_here>",
  "variables_to_downscale": {
    "t_air": "y",
    "sw_radiation": "y",
    "relative_humidity": "n",
    "precipitation": "n",
    "wind": "n",
    "lw_radiation": "n"
  },
  "start_date": "2017-04-01",
  "end_date": "2023-03-31",
  "aggregate_daily": "y",
  "dem_nodata": -32768,
  "auto_calibrate_lapse_rate": "n",
  "custom_lapse_rates": {
    "temperature": {
      "monthly": [
        8.1,
        7.9,
        7.78,
        7.76,
        7.9,
        8,
        8.2,
        8.4,
        8.6,
        8.7,
        8.4,
        8.32
      ]
    },
    "precipitation": {
      "monthly": null
    }
  },
  "jobs_parallel_downscale": -1,
  "jobs_parallel_download": -1
}
```

* You must register and obtain a PAT from [earthdatahub.destine.eu](https://earthdatahub.destine.eu/)

---

### Bash file example

```bash
#!/bin/bash

# Exit if any command fails
set -e

# Activate conda environment
echo "Activating conda environment 'swe3'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swe3

# Path to your Python script and config
SCRIPT_PATH="./main_micromet.py"
CONFIG_PATH="./micro_config_MAIPO.json"

echo "Running MicroMet downscaling..."
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done DK."

```
---



## Example: Run downscaling on the Maipo area (via `run_micromet_MAIPO.sh`)

A ready-to-use script is included to run the downscaling workflow for the Maipo domain:

```bash
# From the project root
chmod +x run_micromet_MAIPO.sh
./run_micromet_MAIPO.sh
```

This script launches the pipeline using the Maipo configuration (see `micro_config_MAIPO_AUTODEM.json`). Adjust paths or environment details inside the script if your setup differs.



## References

* Liston, G.E., & Elder, K. (2006). A Meteorological Distribution System for High-Resolution Terrestrial Modeling (MicroMet). *Journal of Hydrometeorology*, 7(2), 217-234. [https://doi.org/10.1175/JHM486.1](https://doi.org/10.1175/JHM486.1)

---

## Acknowledgements


---

## To Do


---

## Contact


---

Happy downscaling!

---



