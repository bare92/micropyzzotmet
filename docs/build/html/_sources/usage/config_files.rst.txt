Configuration files
===================

MicroPyzzotMet is driven by a single JSON configuration file, passed to
``main_micromet.py``::

    python main_micromet.py path/to/your_config.json

The configuration controls:

- Working directory and folder structure
- How the DEM is provided or downloaded
- How ERA5-Land data are obtained and pre-processed
- Which variables are downscaled
- Run period and processing options
- Parallelisation and optional S3M output

Below we describe each key and provide a complete example.

Basic structure
---------------

A minimal configuration contains:

- ``working_directory``: base folder where ``inputs/`` and ``outputs/`` are
  created (automatically by ``create_full_micromet_folder_structure``). :contentReference[oaicite:0]{index=0}
- DEM configuration: either point to an existing DEM (``dem_file``) or ask
  MicroPyzzotMet to download a Copernicus DEM patch using
  ``download_dem_*`` fields. :contentReference[oaicite:1]{index=1}
- ERA5 configuration: either point to existing NetCDF files (``era_file``) or
  trigger ERA5-Land download via ``earthdatahub_pat``. 
- ``variables_to_downscale``: choose which fields to process (temperature,
  radiation, etc.), using ``"y"`` / ``"n"`` flags. 
- Temporal range: ``start_date`` and ``end_date`` (inclusive).
- Optional advanced options: lapse-rates, parallel jobs, daily aggregation,
  S3M output, buffer size for NetCDF writes, etc. 


Example configuration
---------------------

Below is a complete example configuration (based on your current setup),
with a **placeholder** for the personal access token (PAT) that you should
obtain from ``earthdatahub.destine.eu`` and **never commit in plain text**:

.. code-block:: json

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
        "relative_humidity": "n",
        "precipitation": "n",
        "wind": "n",
        "lw_radiation": "n"
      },

      "start_date": "2017-04-01",
      "end_date": "2018-03-31",
      "aggregate_daily": "n",

      "dem_nodata": -32768,

      "auto_calibrate_lapse_rate": "n",

      "custom_lapse_rates": {
        "temperature": {
          "monthly": [8.1, 7.9, 7.78, 7.76, 7.9, 8.0, 8.2, 8.4, 8.6, 8.7, 8.4, 8.32]
        },
        "precipitation": {
          "monthly": null
        }
      },

      "jobs_parallel_downscale": -1,
      "jobs_parallel_download": -1,

      "write_buffer_steps": 24,

      "generate_s3m_input": "n"
    }

You can use this file as a template and adapt each block as described below.

Paths, working directory and folder structure
---------------------------------------------

**working_directory**

- Path to the base folder for the run.
- At the beginning of the run,
  :func:`create_full_micromet_folder_structure` creates:

  - ``<working_directory>/inputs/climate``
  - ``<working_directory>/inputs/dem``
  - ``<working_directory>/outputs`` 

**DEM configuration**

- ``dem_file``

  - Path to an existing DEM GeoTIFF.
  - If not ``null``, this DEM is used as the reference grid. :contentReference[oaicite:6]{index=6}
- ``download_dem_extent``, ``download_dem_epsg``, ``download_dem_resolution``,
  ``output_filename_dem``

  - Used only when ``dem_file`` is ``null``.
  - :func:`download_and_save_dem_from_config` downloads Copernicus DEM
    (GLO-30) via EarthDataHub, crops it to the given extent, reprojects to
    ``download_dem_epsg`` and resamples to ``download_dem_resolution`` (in
    metres). 
  - The result is written to
    ``<working_directory>/inputs/dem/<output_filename_dem>`` and returned
    as ``dem_file``.

**Important:** the extent coordinates must be consistent with
``download_dem_epsg``. If the extent is not aligned with the requested
resolution, an informative error suggests corrected bounds. :contentReference[oaicite:8]{index=8}

ERA5-Land configuration
-----------------------

You can either:

1. **Use existing ERA5(-Land) NetCDF files** (one per month) and point
   ``era_file`` to them (then the download step is skipped), or
2. **Let MicroPyzzotMet download ERA5-Land directly** from EarthDataHub.

In the current workflow, ``era_file`` is typically left as ``null`` and the
code:

- Calls :func:`get_era5` when ``era_file`` is ``null``. 
- Uses ``earthdatahub_pat`` to authenticate against the zarr store on
  EarthDataHub for both ERA5-Land and Copernicus DEM.

**Fields:**

- ``era_file``

  - Currently used mainly as a flag. When ``null``, ERA5-Land is downloaded
    into ``<working_directory>/inputs/climate`` as monthly NetCDF files
    ``era_YYYY_MM.nc`` (or ``era_YYYY_MM_daily.nc`` if daily aggregation is
    requested). 

- ``earthdatahub_pat``

  - Personal Access Token for EarthDataHub (string).
  - Required **both** for DEM download (Copernicus GLO-30) and ERA5-Land
    access. 
  - Do **not** commit real tokens into a public repository. Use environment
    variables, a private config file, or a local, untracked JSON.

- ``start_date``, ``end_date``

  - Time range of ERA5-Land data (strings in ``YYYY-MM-DD`` format).
  - Passed to :func:`get_era5` and used to build the list of monthly
    NetCDF files.

- ``aggregate_daily``

  - ``"y"`` / ``"n"`` flag.
  - When ``"y"``, the ERA5 stream is aggregated to daily values inside
    :func:`process_month` (means for most variables, special handling for
    precipitation and radiation). :contentReference[oaicite:12]{index=12}
  - Parsed via :func:`parse_yes_no_flag`. :contentReference[oaicite:13]{index=13}

Variables to downscale
----------------------

The ``variables_to_downscale`` block controls which downscaling routines
are executed. Each key is a ``"y"`` / ``"n"`` flag and is parsed by
:func:`parse_yes_no_flag`. 

Available keys:

- ``"t_air"``: 2 m air temperature (``t2m``) → stored in
  ``outputs/Temperature``. Uses monthly lapse rates or on-the-fly
  calibration and buffered NetCDF writing. 
- ``"sw_radiation"``: shortwave radiation (``ssrd``), topographically
  corrected using slope/aspect and solar geometry; output goes to
  ``outputs/SW``. 
- ``"relative_humidity"``: relative humidity, output ``outputs/RH``.
- ``"precipitation"``: precipitation (from ``tp``, after conversion and
  differencing), output ``outputs/P``. 
- ``"wind"``: wind speed/direction (``u10``, ``v10``) with topographic
  adjustment and curvature, output ``outputs/Wind``. 
- ``"lw_radiation"``: longwave radiation (from ``strd`` and downscaled
  temperature/humidity), output ``outputs/LW``. 

DEM and lapse-rate options
--------------------------

- ``dem_nodata``

  - Nodata value used when reading the DEM and in many downscaling steps
    to mask invalid cells. 

- ``auto_calibrate_lapse_rate``

  - ``"y"`` / ``"n"`` flag.
  - When ``"y"``, the temperature lapse rate is estimated for each
    timestep by linear regression of ERA5 ``t2m`` vs geopotential-derived
    height (``z``). 
  - When ``"n"``, a fixed monthly lapse rate is used (either the default
    MicroMet rates or custom ones).

- ``custom_lapse_rates``

  - Optional object with keys:

    - ``"temperature" : { "monthly": [12 values] }``
    - ``"precipitation" : { "monthly": [12 values] or null }``

  - The monthly arrays (K/km for temperature, gamma for precipitation)
    override the internal MicroMet parameters when ``auto_calibrate_lapse_rate``
    is ``"n"``. 

Parallelisation and performance
-------------------------------

- ``jobs_parallel_downscale``

  - Number of parallel jobs used by joblib when downscaling each variable
    over all monthly NetCDF files. ``-1`` means “use all available cores”. :contentReference[oaicite:23]{index=23}

- ``jobs_parallel_download``

  - Number of parallel jobs when post-processing ERA5 month by month in
    :func:`get_era5`. Again, ``-1`` uses all cores. :contentReference[oaicite:24]{index=24}

- ``write_buffer_steps`` (optional, default: 24)

  - Controls how many time steps are buffered in memory before being
    flushed to NetCDF when downscaling temperature. 
  - Smaller values → less memory, more frequent writes.
  - Larger values → more memory, fewer writes, potentially faster I/O.

Optional S3M output
-------------------

- ``generate_s3m_input`` (optional, default: ``"n"``)

  - When set to ``"y"``, after all downscaling is complete the code calls
    :func:`convert_micromet_to_s3m_inputs` to create one compressed
    NetCDF (``.nc.gz``) per timestep in ``outputs/s3m`` with the S3M
    variable naming convention (``Rain``, ``AirTemperature``,
    ``IncRadiation``, ``RelHumidity``, ``terrain``, ``longitude``,
    ``latitude``). 


