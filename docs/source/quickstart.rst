Quickstart
==========

This quickstart provides the minimal steps to run a complete
MicroPyzzotMet downscaling workflow, from cloning the repository to
running a full example (e.g. the Maipo domain).

1. Clone the repository
-----------------------

Clone the project from GitHub and move into the folder::

    git clone https://github.com/bare92/micropyzzotmet.git
    cd micropyzzotmet

2. Set up the environment
-------------------------

MicroPyzzotMet provides a setup script that installs all required
dependencies and prepares the environment. From the project root, run::

    chmod +x setup_micropyzzotmet_env.sh
    ./setup_micropyzzotmet_env.sh

Follow any instructions printed on screen (e.g., activating a conda or
virtual environment).

3. Prepare your configuration file
----------------------------------

MicroPyzzotMet is controlled by a JSON configuration file specifying:

- working directory
- DEM source (local file or auto-download)
- ERA5-Land input
- time period
- variables to downscale
- lapse rates
- parallelisation settings

An example is provided in ``micro_config_DEMO_MAIPO.json``.
Typical structure::

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
	  "earthdatahub_pat": "edh_pat_61df66e3f10aa2e2de793de541ed4c55259bc1f7c37b54303073a1dd38952db257a6f74b388cf9d44578d01ace9c3928",
	  "variables_to_downscale": {
	    "t_air": "y",
	    "sw_radiation": "n",
	    "relative_humidity": "n",
	    "precipitation": "n",
	    "wind": "n",
	    "lw_radiation": "n"
	  },
	  "start_date": "2017-04-01",
	  "end_date": "2017-05-31",
	  "aggregate_daily": "y",
	  "write_buffer_steps": 192,
	  "dem_nodata": -32768,
	  "auto_calibrate_lapse_rate": "n",
	  "custom_lapse_rates": {
	    "temperature": {
	      "monthly": [8.1, 7.9, 7.78, 7.76, 7.9, 8, 8.2, 8.4, 8.6, 8.7, 8.4, 8.32]
	    },
	    "precipitation": {
	      "monthly": null
	    }
	  },
	  "jobs_parallel_downscale": -1,
	  "jobs_parallel_download": -1
	}


.. note::

   You must obtain a Personal Access Token (PAT) from
   https://earthdatahub.destine.eu/ to download ERA5-Land data
   automatically.

4. Run the downscaling model
----------------------------

To run the model using your configuration file::

    python main_micromet.py path/to/your_config.json

This launches the entire downscaling workflow, including:

- DEM preparation (or auto-download)
- climate data download (if enabled)
- downscaling of all selected variables
- writing outputs in NetCDF format

5. Understanding the output structure
-------------------------------------

MicroPyzzotMet automatically creates a folder structure under
``working_directory``::

    working_directory/
    ├── inputs/
    │   ├── climate/           # ERA5-Land NetCDF files
    │   └── dem/               # DEM and derived slope/aspect/curvature
    └── outputs/
        ├── Temperature/
        ├── SW/                # Shortwave radiation
        ├── RH/                # Relative humidity
        ├── P/                 # Precipitation
        ├── Wind/              # Wind speed/direction
        └── LW/                # Longwave radiation

Example: running the Maipo workflow
-----------------------------------

A ready-to-use script is provided for the Maipo region. To run it::

    chmod +x run_micromet_MAIPO.sh
    ./run_micromet_MAIPO.sh

This script uses ``micro_config_MAIPO_AUTODEM.json`` and runs the full
workflow automatically.

Example bash script
-------------------

Below is an example of a user script that activates an environment and
runs MicroPyzzotMet::

	#!/bin/bash

	# Exit if any command fails
	set -e

	# Activate conda environment
	echo "Activating conda environment 'microenv'..."
	source $(conda info --base)/etc/profile.d/conda.sh
	conda activate microenv

	# Path to your Python script and config
	SCRIPT_PATH="./main_micromet.py"
	CONFIG_PATH="./micro_config_DEMO_MAIPO.json"

	echo "Running MicroMet downscaling..."
	python "$SCRIPT_PATH" "$CONFIG_PATH"

	echo "Done."



