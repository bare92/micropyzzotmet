Running the downscaling
=======================

This page describes how to run the current CLI workflow.

Run command
-----------

From the repository root::

    micropyzzotmet path/to/config.json

Example::

    micropyzzotmet option_files/micro_config_DEMO_MAIPO.json

What the workflow does
----------------------

The run function performs these main steps:

1. Load the JSON config.
2. Create the folder structure under ``working_directory``.
3. Prepare the DEM:

   - Use ``dem_file`` if provided, or
   - Download the DEM from EarthDataHub when ``dem_file`` is ``null``.

4. Download ERA5-Land when ``era_file`` is ``null``.
5. Compute slope, aspect, and curvature.
6. Downscale all enabled variables month-by-month in parallel.
7. Optionally create S3M forcing files.

Outputs
-------

All outputs are written under ``<working_directory>/outputs``::

    outputs/
    ├── Temperature/
    ├── SW/
    ├── RH/
    ├── P/
    ├── Wind/
    ├── LW/
    └── s3m/   # only when generate_s3m_input = "y"

Input expectations
------------------

- Run from the repository root for reliable access to auxiliary data.
- If ``era_file`` is non-null and automatic download is skipped, ensure climate
  NetCDF files already exist in ``<working_directory>/inputs/climate``.
- Configure EarthDataHub credentials via PAT or ``~/.netrc``.

Parallel settings
-----------------

- ``jobs_parallel_download`` controls download processing parallelism.
- ``jobs_parallel_downscale`` controls per-variable monthly downscaling jobs.

Monitoring progress
-------------------

During execution, the code prints progress messages for:

- DEM creation and terrain derivatives
- ERA5-Land download and monthly processing
- each downscaled variable
- optional S3M file generation

Troubleshooting notes
---------------------

- If you see PROJ database errors, ensure only one active geospatial stack is
  being used in the active environment.
- If zarr or filesystem import errors appear, verify ``fsspec``, ``s3fs``, and
  ``dask`` are installed.
- If automatic download is skipped, verify monthly climate files exist in the
  expected climate input folder before launching the run.


