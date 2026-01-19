---
title: 'MicroPyzzotMet: A Lightweight Python Package for Climate Downscaling'
tags:
  - Python
  - climate
  - downscaling
  - xarray
  - zarr
  - snow

authors:
  - name: Riccardo Barella
    orcid: 0000-0002-2920-2613
    corresponding: true
    equal-contrib: true
    affiliation: 1

  - name: Katharina Theresa Scheidt
    orcid: 0000-0002-1222-5333
    affiliation: 1

  - name: Valentina Premier
    orcid: 0000-0002-4629-2235
    affiliation: 1

  - name: Carlo Marin
    orcid: 0000-0001-6987-9445
    affiliation: 1

affiliations:
 - name: Institute for Earth Observation, Eurac Research, Bolzano
   index: 1

date: 19 January 2026
bibliography: paper.bib
---

# Summary

Modern reanalysis products provide continuous global climate information extending back decades and into future projections, yet their spatial resolution remains too coarse to represent the meteorological variability imposed by mountain terrain [@fanetal2019]. For researchers working on snow, glaciers, permafrost, and alpine hydrology, this mismatch remains a persistent limitation: surface energy-balance and mass-balance models depend on local meteorological fields that capture how terrain modulates atmospheric conditions. Generating such fields does not always require complex dynamical downscaling systems, which—despite their accuracy—often demand substantial computational resources and sophisticated model setups.

`MicroPyzzotMet` addresses this gap with a lightweight downscaling framework focused on practicality and broad usability. Rather than implementing complex physical parameterizations or spatial clustering techniques, it applies a streamlined set of MicroMet-inspired corrections to temperature, radiation, humidity, wind, and precipitation [@liston2006meteorological]. Because it requires only a minimal set of essential climate variables, the tool can operate with virtually any reanalysis or climate dataset. A further strength is its integration with EarthDataHub, enabling rapid access to ERA5-Land and digital terrain models through the Zarr format, which significantly reduces I/O overhead and speeds up preprocessing.

In contrast to more advanced packages such as `TopoPyScale`, which is designed for detailed terrain-driven heterogeneity and fine-scale modelling [@filhol2023topopyscale], `MicroPyzzotMet` prioritizes computational efficiency and conceptual clarity. This makes it ideal for large-domain experiments, operational workflows, or rapid prototyping, while still remaining compatible with higher-resolution approaches when more elaborate topographic corrections are required.

# Statement of need

`MicroPyzzotMet` is an open-source Python package for downscaling meteorological variables from reanalysis climate datasets. It is inspired by the MicroMet methodology [@liston2006meteorological] but reimplemented in a modern Python framework, making the workflow more accessible, flexible, and easier to integrate in contemporary data-processing pipelines.

The increasing availability of atmospheric reanalyses—such as ERA5 and ERA5-Land at 25 km and 9 km spatial resolution—has enabled a wide range of cryospheric and hydrological studies. However, these coarse spatial grids remain inadequate for mountain regions, where elevation, slope, and aspect strongly modulate near-surface climate. Tools such as `TopoPyScale` [@filhol2023topopyscale], based on the TopoSCALE and TopoSUB approaches [@fiddesTopoSCALEDownscalingGridded2014; @fiddesTopoSUBToolEfficient2012], address this limitation by applying sophisticated 3-D interpolation schemes, multi-level atmospheric corrections, and DEM segmentation into terrain clusters. These methods allow detailed reconstruction of fine-scale meteorological patterns, especially over complex alpine terrain.

Yet not all applications require this level of complexity. For coarser target resolutions (e.g., 250–500 m), for large domains processed at high temporal frequency, or for long climatological time series, the computational and data requirements of advanced downscaling frameworks may become limiting. In these cases, the original MicroMet methodology offers an attractive balance between physical robustness and computational simplicity.

`MicroPyzzotMet` builds upon this philosophy. It applies lapse-rate corrections, radiative geometry adjustments, vapor-pressure formulations, and precipitation-elevation relationships following MicroMet, using only the set of climate variables typically available in major reanalysis datasets. This makes the tool broadly applicable, lightweight, and extremely fast, while still delivering spatially coherent meteorological fields suitable for surface energy and mass-balance modelling.

A key feature of `MicroPyzzotMet` is its integration with EarthDataHub [@EarthDataHub2025], which distributes global reanalysis datasets—including ERA5-Land—preconverted into Zarr format. This enables efficient cloud-native data access and processing through Xarray/Dask, greatly accelerating workflows and reducing storage overhead.

\autoref{fig:downscaling_example} illustrates an example for the Maipo region in Chile, comparing native ERA5-Land fields to `MicroPyzzotMet` downscaled products for daily air temperature and incoming shortwave radiation.

![In the figure are presented two examples of downscaled variables—daily air temperature and shortwave incoming radiation—over the Maipo region in Chile on January 1, 2017. Panels a) and c) show the native ERA5-Land fields at 9 km resolution, overlaid with the DEM used for downscaling at 50 m resolution. Panels b) and d) show the corresponding fields produced by MicroPyzzotMet. \label{fig:downscaling_example}](downscaling_examples.png)

# Toolbox methods and structure

`MicroPyzzotMet` is implemented entirely in Python and builds on widely used scientific and geospatial libraries, including [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [xarray](https://docs.xarray.dev/), and its Zarr engine for cloud-native data access. It uses [rasterio](https://rasterio.readthedocs.io/) and [rioxarray](https://corteva.github.io/rioxarray/stable/) for raster handling, and [pyproj](https://pyproj4.github.io/pyproj/stable/) for coordinate transformations. Terrain derivatives such as slope, aspect, and curvature are generated with `rasterio`, `gdaldem`, and custom convolution kernels, while [pvlib](https://pvlib-python.readthedocs.io/) is employed to compute solar geometry required for shortwave radiation corrections. Parallel processing is handled via [joblib](https://joblib.readthedocs.io/) to distribute downscaling tasks across CPU cores.

![Workflow of the MicroPyzzotMet processing pipeline.\label{fig:micropyzzomet_workflow}](micropyzzotmet_workflow.png){ width=50% }

**Figure 2:** Schematic overview of the *MicroPyzzotMet* downscaling workflow. Coarse-resolution reanalysis data (e.g., ERA5-Land) and a Digital Elevation Model (DEM) constitute the primary inputs. Terrain derivatives (slope, aspect, curvature) are computed from the DEM and combined with solar geometry to drive MicroMet-inspired corrections. Each meteorological variable is processed independently through reprojection to the DEM grid, vertical (lapse-rate) adjustment, and terrain-based corrections, producing high-resolution meteorological fields written as NetCDF outputs suitable for cryospheric and hydrological modelling.


The workflow of `MicroPyzzotMet` is controlled by a single JSON configuration file and orchestrated by the main execution function. The pipeline begins by creating a standard folder structure (`inputs/climate`, `inputs/dem`, `outputs`) and by loading or downloading a Digital Elevation Model (DEM). When no DEM is provided, the tool retrieves a Copernicus GLO-30 subset from EarthDataHub as a Zarr dataset, reprojects and resamples it to the user-defined grid, and writes it to GeoTIFF. Slope, aspect, and curvature metrics are then computed and stored for use in the downscaling routines.

Meteorological forcing is obtained either from user-supplied NetCDF files or directly from ERA5-Land via EarthDataHub. When downloaded through EarthDataHub, the Zarr dataset is spatially subsetted to match the DEM extent and written to monthly NetCDF files containing variables such as 2 m air temperature and dewpoint, surface pressure, 10 m wind components, precipitation, and shortwave and longwave radiation. Cumulative fluxes are optionally converted to hourly or daily rates.

Once the DEM and climate inputs are prepared, `MicroPyzzotMet` applies a set of variable-specific downscaling functions. These functions implement MicroMet-style parameterizations:  
- **Temperature** is adjusted using monthly lapse rates or dynamically calibrated rates.  
- **Shortwave radiation** is corrected using topographic metrics and solar geometry.  
- **Relative humidity** is derived from temperature and dewpoint using vapor-pressure relationships.  
- **Precipitation** is scaled with elevation using empirical gradients.  
- **Wind fields** are modified based on terrain metrics.  
- **Longwave radiation** is adjusted using cloudiness estimates derived from humidity and temperature.

Each routine reads a single monthly climate file, reprojects the coarse fields to the DEM grid, applies vertical and topographic corrections, and writes a NetCDF output file.

The selection of variables to downscale is fully configurable, allowing modular development and efficient processing of large datasets.

**Table 1:** Default downscaled output variables of `MicroPyzzotMet` (based on ERA5-Land inputs).

| Name                          | Variable        | Unit     | Downscaling type                                                                 |
|-------------------------------|-----------------|----------|----------------------------------------------------------------------------------|
| 2 m Air temperature           | `t2m`           | K        | Vertical lapse-rate adjustment; reprojection to DEM grid                         |
| Relative humidity             | `RH`            | %        | Lapse-rate corrections; vapor-pressure formulation                                |
| Surface pressure              | `sp`            | Pa       | Reprojection to DEM grid (optional elevation adjustment)                          |
| 10 m Wind speed and direction | `u10`, `v10`    | m s⁻¹    | Reprojection and terrain-based adjustments                                        |
| Precipitation                 | `P`             | mm       | Elevation-dependent scaling using empirical gradients                              |
| Incoming longwave radiation   | `LW`            | W m⁻²    | Atmospheric and cloudiness corrections                                            |
| Incoming shortwave radiation  | `SW`            | W m⁻²    | Topographic and solar-geometry corrections                                        |

# Working examples

A complete working example of `MicroPyzzotMet` is available in the public repository:  
<https://github.com/bare92/micropyzzotmet>.

The included demonstration applies the downscaling workflow to the Maipo basin in central Chile, a region characterized by steep elevation gradients and strong spatial variability in meteorological forcing.

The example is configured through the file `micro_config_DEMO_MAIPO.json` and executed with a simple shell script. In this workflow:

- A DEM covering the Maipo catchment is downloaded from EarthDataHub as a Zarr dataset, reprojected to EPSG:32719 (UTM 19S), and resampled to 50 m resolution.  
- ERA5-Land meteorological inputs for **1 April to 31 May 2017** are fetched via EarthDataHub, enabling fast cloud-native access to reanalysis data.  
- All major variables—air temperature, shortwave and longwave radiation, relative humidity, precipitation, and wind—are downscaled using MicroMet-based parameterizations.  
- Outputs are written as monthly NetCDF files and can be converted into S3M-compatible forcing files.

This demonstration illustrates the typical usage of `MicroPyzzotMet`: a lightweight, configuration-driven workflow capable of producing high-resolution atmospheric forcing fields with minimal user intervention. The Maipo setup can be adapted to other regions by modifying the spatial extent, DEM specifications, and processing period.

# Acknowledgements

This project has received funding from the European Union’s Horizon Research and Innovation Actions programme under Grant Agreement 101180133, and from the Swiss State Secretariat for Education, Research and Innovation (SERI).

# References

