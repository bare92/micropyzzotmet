Quickstart
==========

This quickstart provides the minimal steps to run a complete
MicroPyzzotMet workflow using the current CLI.

1. Clone the repository
-----------------------

Clone the project and enter the folder::

    git clone https://github.com/bare92/micropyzzotmet.git
    cd micropyzzotmet

2. Create and activate a virtual environment
--------------------------------------------

From the repository root::

    python3 -m venv .venv
    source .venv/bin/activate
    pip install --upgrade pip
    pip install -e .

Check the CLI is available::

    micropyzzotmet --help

3. Prepare your configuration file
----------------------------------

MicroPyzzotMet is configured through a JSON file.

Example configs are available in ``option_files/``:

- ``option_files/micro_config_DEMO_MAIPO.json``
- ``option_files/micro_config_alps.json``

A typical structure is::

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
      "jobs_parallel_download": 4
    }

.. note::

   To download DEM and ERA5-Land data automatically, provide credentials
   either via ``earthdatahub_pat`` in the config file or via ``~/.netrc``
   for machine ``earthdatahub.com``.

4. Run the downscaling workflow
-------------------------------

Run the workflow::

    micropyzzotmet option_files/micro_config_DEMO_MAIPO.json

You can pass any other JSON config path::

    micropyzzotmet path/to/your_config.json

5. Check outputs
----------------

The run writes data under ``working_directory``::

    working_directory/
    ├── inputs/
    │   ├── climate/
    │   └── dem/
    └── outputs/
        ├── Temperature/
        ├── SW/
        ├── RH/
        ├── P/
        ├── Wind/
        ├── LW/
        └── s3m/   # only when generate_s3m_input = "y"

Example launcher script
-----------------------

Minimal bash launcher::

    #!/bin/bash
    set -e

    source /path/to/micropyzzotmet/.venv/bin/activate
    cd /path/to/micropyzzotmet

    micropyzzotmet option_files/micro_config_DEMO_MAIPO.json



