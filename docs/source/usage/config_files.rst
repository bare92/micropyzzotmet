Configuration files
===================

MicroPyzzotMet is driven by a JSON configuration file passed to the CLI::

    micropyzzotmet path/to/your_config.json

This page documents the keys currently used by the code.

Core fields
-----------

- ``working_directory``
  Base output folder where ``inputs/`` and ``outputs/`` are created.

- ``dem_file``
  Path to an existing DEM GeoTIFF, or ``null`` to download the DEM from
  EarthDataHub.

- ``download_dem_extent``
  Required when ``dem_file`` is ``null``. Object with ``lat_min``,
  ``lat_max``, ``lon_min``, ``lon_max`` in the CRS defined by
  ``download_dem_epsg``.

- ``download_dem_epsg``
  Target EPSG code for downloaded DEM.

- ``download_dem_resolution``
  Target DEM resolution in meters.

- ``output_filename_dem``
  Filename for downloaded DEM inside ``<working_directory>/inputs/dem``.

- ``era_file``
  Currently used as a switch:

  - ``null``: download ERA5-Land into ``<working_directory>/inputs/climate``.
  - non-null: skip automatic download. In that case, climate files are
    expected to already exist in ``<working_directory>/inputs/climate``.

Authentication fields
---------------------

- ``earthdatahub_pat``
  EarthDataHub personal access token.

- ``earthdatahub_machine`` (optional)
  Machine name for ``~/.netrc`` lookup. Default is ``earthdatahub.com``.

Use either ``earthdatahub_pat`` or a valid ``~/.netrc`` entry.

Variable and time controls
--------------------------

- ``variables_to_downscale``
  Object with ``"y"`` or ``"n"`` flags for:

  - ``t_air``
  - ``sw_radiation``
  - ``relative_humidity``
  - ``precipitation``
  - ``wind``
  - ``lw_radiation``

- ``start_date`` and ``end_date``
  Date range in ``YYYY-MM-DD`` format.

- ``aggregate_daily``
  ``"y"`` or ``"n"``. Controls daily aggregation for downloaded ERA5-Land data.

- ``time_chunk``
  Chunk size used during downscaling output writing. The current code defaults
  to ``24`` when not explicitly provided.

- ``dem_nodata``
  No-data value used for DEM masking.

Lapse-rate and parallel settings
--------------------------------

- ``custom_lapse_rates``
  Optional object with monthly values for:

  - ``temperature.monthly`` (12 values)
  - ``precipitation.monthly`` (12 values or ``null``)

- ``jobs_parallel_downscale``
  Parallel job count for per-variable monthly downscaling.

- ``jobs_parallel_download``
  Parallel job count for ERA5 monthly download and processing.

Optional S3M export
-------------------

- ``generate_s3m_input``
  If ``"y"``, generate S3M-compatible files in
  ``<working_directory>/outputs/s3m``.

Complete example
----------------

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
      "earthdatahub_machine": "earthdatahub.com",
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

.. warning::

   Never commit a real EarthDataHub PAT into version control.


