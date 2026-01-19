Running the downscaling
=======================

This page explains how to execute the full MicroPyzzotMet workflow using
a JSON configuration file. The process includes DEM preparation,
ERA5-Land download or ingestion, and downscaling of all selected
meteorological variables.

Overview of the workflow
------------------------

A typical downscaling run consists of:

1. Preparing or downloading a DEM.
2. Preparing or downloading the ERA5-Land meteorological data.
3. Creating a JSON configuration file describing the domain and settings.
4. Running the main entry point ``main_micromet.py``.
5. Inspecting the generated outputs in the ``working_directory``.

All steps are automated based on the configuration file.

Running MicroPyzzotMet
----------------------

The main entry point is ``main_micromet.py``. To run the full workflow::

    python main_micromet.py path/to/your_config.json

Internally, the script performs the following steps:

1. **Load and validate the configuration**  
   The JSON file is parsed, all paths are checked, ``"y"/"n"`` flags are
   converted to booleans, and optional fields receive defaults.  
   (See :func:`load_and_validate_config`).  
   Typical validations include ensuring:
   - output folders can be created
   - EarthDataHub PAT is provided when required
   - date ranges are valid  
   - DEM and climate inputs exist or can be downloaded

2. **Create the directory structure**  
   ``inputs/`` and ``outputs/`` folders are created automatically in the  
   ``working_directory``.  
   This uses :func:`create_full_micromet_folder_structure`.  
   The structure looks like::

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

3. **DEM preparation**  
   Depending on the configuration:
   - If ``dem_file`` is set, this DEM is copied into ``inputs/dem``.
   - If ``dem_file`` is ``null``, MicroPyzzotMet downloads a DEM using
     Copernicus GLO-30 via EarthDataHub.  
     This is handled by :func:`download_and_save_dem_from_config`,
     including:
     - retrieving DEM tiles
     - cropping to the requested extent
     - reprojecting to ``download_dem_epsg``
     - resampling to the requested resolution

   Once available, the DEM is processed into:
   - slope
   - aspect
   - curvature  
   via :func:`calculate_topographic_parameters`.

4. **Download or load ERA5-Land climate data**  
   If ``era_file`` is ``null``:  
   MicroPyzzotMet automatically downloads ERA5-Land using EarthDataHub,
   month by month, via :func:`get_era5`.  

   ERA5 streams and variables handled:
   - ``t2m`` (temperature)
   - ``u10``, ``v10`` (wind)
   - ``tp`` (precipitation)
   - ``ssrd`` (shortwave radiation)
   - ``strd`` (longwave radiation)
   - ``z`` (geopotential, used for temperature lapse rate calibration)

   The files are saved under::

       working_directory/inputs/climate/era_YYYY_MM.nc

5. **Downscaling of meteorological variables**  
   For each variable with flag ``"y"`` under ``variables_to_downscale``,
   the corresponding downscaling routine is executed:

   - ``t_air`` → temperature  
     :func:`downscale_temperature`
   - ``sw_radiation`` → shortwave  
     :func:`downscale_shortwave`
   - ``relative_humidity`` → RH  
     :func:`downscale_relative_humidity`
   - ``precipitation`` → rainfall or snowfall  
     :func:`downscale_precipitation`
   - ``wind`` → 10m wind field  
     :func:`downscale_wind`
   - ``lw_radiation`` → longwave  
     :func:`downscale_longwave`

   Each variable is processed month by month.  
   Parallelisation is controlled by ``jobs_parallel_downscale``.

6. **(Optional) Generate S3M-ready files**  
   If ``generate_s3m_input`` is set to ``"y"``, the function  
   :func:`convert_micromet_to_s3m_inputs` creates daily  
   ``.nc.gz`` files suitable for the S3M snow model.

Monitoring progress
-------------------

During execution, messages printed on screen (via ``print`` and ``tqdm``)
indicate:

- DEM download and preparation status  
- Climate file downloads and processing  
- Downscaling progress for each variable  
- Monthly loops and parallel execution status  
- S3M file generation (if enabled)

Typical run command
-------------------

After creating a configuration file, a full run looks like::

    conda activate microenv
    python main_micromet.py ./config/my_config.json

Depending on the domain size and variable selection, this may take from
minutes (small DEM, few variables) to hours (large DEM, many years).

Outputs
-------

All results are written in ``working_directory/outputs``. For example::

    outputs/
    ├── Temperature/
    │   ├── t_air_2017_04.nc
    │   ├── t_air_2017_05.nc
    │   └── ...
    ├── SW/
    ├── RH/
    ├── P/
    ├── Wind/
    └── LW/

Each folder contains monthly downscaled fields on the DEM grid.

If S3M export is enabled::

    outputs/s3m/*.nc.gz

These are ready to be used by the Fortran S3M snow model.

Example quick run
-----------------

A minimal end-to-end example::

    git clone https://github.com/bare92/micropyzzotmet
    cd micropyzzotmet

    chmod +x setup_micropyzzotmet_env.sh
    ./setup_micropyzzotmet_env.sh
    conda activate microenv

    python main_micromet.py ./config/micro_config_DEMP_MAIPO.json

This runs the complete workflow on the Maipo domain.


