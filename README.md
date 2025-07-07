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
  * Longwave Radiation (to be implemented)
* Uses custom or default monthly lapse rates
* Generates GeoTIFFs (will be updated)
* Parallel processing support

---

## Quick Start

### 1. Clone the repository

```bash
git clone [https://github.com/yourusername/micropezzottomet.git](https://github.com/bare92/micropyzzotmet)
cd micropezzottomet
```

### 2. Install requirements

```bash
METTERE COSA FARE
```

!! Make sure you also have `GDAL` installed and accessible from command line (used for slope/aspect computation).

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

## Configuration File Example (`micro_config_MAIPO.json`)

```json
{
  "working_directory": "/path/to/workspace",
  "dem_file": "/path/to/dem.tif",
  "era_file": null,
  "earthdatahub_pat": "your_token_here",
  "variables_to_downscale": {
    "t_air": "y",
    "sw_radiation": "y",
    "relative_humidity": "y",
    "precipitation": "y",
    "wind": "y",
    "lw_radiation": "y"
  },
  "start_date": "2017-04-01",
  "end_date": "2018-03-31",
  "time_step": "24H",
  "dem_nodata": -32768,
  "jobs_parallel_downscale": -1,
  "jobs_parallel_download": -1,
  "custom_lapse_rates": {
    "temperature": {"monthly": null},
    "precipitation": {"monthly": null}
  }
}
```

* Use `null` for default lapse rates (Liston and Elder, 2006)
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

## References

* Liston, G.E., & Elder, K. (2006). A Meteorological Distribution System for High-Resolution Terrestrial Modeling (MicroMet). *Journal of Hydrometeorology*, 7(2), 217-234. [https://doi.org/10.1175/JHM486.1](https://doi.org/10.1175/JHM486.1)

---

## Acknowledgements

mo li si aggiunge

---

## To Do


---

## Contact
o criatore

---

Happy downscaling e ktm! 
