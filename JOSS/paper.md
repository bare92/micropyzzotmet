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

# State of the field

Global climate reanalyses such as ERA5 and ERA5-Land provide long-term, spatially consistent meteorological datasets, but their coarse spatial resolution (9–31 km) limits their direct use in complex terrain, where topography strongly modulates near-surface climate. Downscaling is therefore required to generate meteorological forcing suitable for cryospheric and hydrological applications.

A range of approaches exists, spanning empirical corrections to physically based methods. Tools such as TopoPyScale [@filhol2023topopyscale], based on the TopoSCALE and TopoSUB frameworks [@fiddesTopoSCALEDownscalingGridded2014; @fiddesTopoSUBToolEfficient2012], apply terrain-aware downscaling using vertical atmospheric profiles, 3D interpolation, and topographic clustering. These methods enable detailed reconstruction of fine-scale meteorological variability, particularly in alpine environments.

However, such approaches typically require multiple atmospheric variables, preprocessing steps, and non-trivial computational resources. For large domains, long simulations, or operational workflows, these requirements may become limiting.

In this context, the MicroMet formulation [@liston2006meteorological] provides a widely used alternative based on physically informed empirical corrections (e.g., lapse rates, radiation geometry, and precipitation–elevation relationships) applied directly to near-surface variables. Despite its continued relevance, modern implementations of this approach that integrate with current Python-based, cloud-native data ecosystems remain limited.

# Statement of need

MicroPyzzotMet is an open-source Python package for downscaling meteorological variables from reanalysis datasets. It builds on the MicroMet methodology [@liston2006meteorological] and reimplements it within a modern, modular Python framework, improving accessibility, flexibility, and integration with contemporary data-processing pipelines.

The package is designed for applications where computational efficiency and scalability are critical, such as large spatial domains, high temporal resolution forcing, or multi-decadal simulations. It applies MicroMet-inspired corrections—including lapse-rate adjustments, radiative geometry, vapor-pressure relationships, and precipitation–elevation scaling—using only the set of variables typically available in standard reanalysis products.

A key feature of MicroPyzzotMet is its integration with EarthDataHub [@EarthDataHub2025], which provides global datasets such as ERA5-Land in cloud-native Zarr format. This enables efficient data access and scalable processing through Xarray and Dask, reducing both I/O overhead and storage requirements.

By combining a lightweight methodological approach with modern data infrastructure, MicroPyzzotMet fills the gap between computationally intensive terrain-resolving frameworks and coarse-resolution reanalysis data. It provides a practical solution for generating spatially coherent meteorological forcing suitable for surface energy- and mass-balance modelling.

\autoref{fig:downscaling_example} illustrates an example for the Maipo region in Chile, comparing native ERA5-Land fields to downscaled air temperature and incoming shortwave radiation.

# Toolbox methods and structure

`MicroPyzzotMet` is implemented entirely in Python and builds on widely used scientific and geospatial libraries, including [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/), [xarray](https://docs.xarray.dev/), and its Zarr engine for cloud-native data access. It uses [rasterio](https://rasterio.readthedocs.io/) and [rioxarray](https://corteva.github.io/rioxarray/stable/) for raster handling, and [pyproj](https://pyproj4.github.io/pyproj/stable/) for coordinate transformations. Terrain derivatives such as slope, aspect, and curvature are generated with `rasterio`, `gdaldem`, and custom convolution kernels, while [pvlib](https://pvlib-python.readthedocs.io/) is employed to compute solar geometry required for shortwave radiation corrections. Parallel processing is handled via [joblib](https://joblib.readthedocs.io/) to distribute downscaling tasks across CPU cores.

![Workflow of the MicroPyzzotMet processing pipeline.\label{fig:micropyzzomet_workflow}](micropyzzotmet_workflow.png)

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

# Software Design

`MicroPyzzotMet` follows a modular, configuration-driven architecture designed for simplicity, transparency, and computational scalability. The entire workflow is controlled through a single JSON configuration file, which specifies spatial extent, temporal range, input data sources, resolution, selected variables, and optional lapse-rate calibration. This minimizes hard-coded parameters and ensures reproducibility across experiments and study areas.

Each meteorological variable is processed through an independent downscaling routine. Temperature, shortwave radiation, relative humidity, precipitation, wind, and longwave radiation are implemented as separate modules that follow a consistent structure: coarse-resolution climate fields are read, vertically adjusted using MicroMet-style parameterizations, reprojected to the DEM grid, and optionally corrected for terrain effects. This modular separation improves maintainability, facilitates testing, and allows straightforward extension to additional variables or alternative parameterizations.

Memory-efficient processing is achieved through buffered NetCDF writing and task-level parallelization across monthly climate files. Terrain derivatives (slope, aspect, curvature) are computed once and reused across variables, reducing redundant computation.

**Build vs. contribute justification.** Existing terrain-aware downscaling frameworks such as `TopoPyScale` focus on high-resolution atmospheric interpolation and terrain clustering strategies optimized for fine-scale alpine modelling. These systems prioritize physical detail and complex atmospheric structure reconstruction. `MicroPyzzotMet` instead targets a complementary use case: lightweight, computationally efficient, large-domain processing using MicroMet-inspired corrections integrated with cloud-native ERA5-Land data access. Extending existing high-complexity frameworks to support this simplified, memory-buffered architecture would require substantial structural changes. The development of a purpose-built implementation therefore provides clearer conceptual scope, reduced dependencies, and improved scalability for operational and ensemble applications.

# Research Impact Statement

`MicroPyzzotMet` enables reproducible generation of high-resolution meteorological forcing fields from globally available reanalysis datasets using a computationally lightweight methodology. Its impact lies in lowering both technical and computational barriers to terrain-aware downscaling in cryospheric and hydrological modelling.

The package supports multi-decadal simulations over large spatial domains through buffered NetCDF writing, monthly parallelization, and direct cloud-native access to ERA5-Land via Zarr archives. These design choices make it possible to perform ensemble experiments, sensitivity analyses, and operational workflows on standard multi-core workstations without requiring high-performance computing infrastructure.

The software demonstrates community readiness through:
- Open-source availability under a permissive license  
- Modular variable-specific implementations  
- Configuration-driven reproducibility  
- Automated DEM acquisition and preprocessing  
- Standardized NetCDF outputs with embedded CRS metadata  
- Compatibility with S3M hydrological forcing formats  

By modernizing the MicroMet methodology within a scalable Python framework, `MicroPyzzotMet` expands accessibility to terrain-aware downscaling and supports transparent, reproducible environmental modelling workflows.


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

## AI usage disclosure

Generative AI tools were used during the development of this project. Specifically, AI assistance was employed for portions of software generation, code refactoring, debugging support, and verification of implementation logic, as well as for language editing and drafting of the manuscript. All generated code was reviewed, tested, and validated by the authors. The authors take full responsibility for the correctness, scientific integrity, and functionality of the software and manuscript.

# Acknowledgements

This project has received funding from the European Union’s Horizon Research and Innovation Actions programme under Grant Agreement 101180133, and from the Swiss State Secretariat for Education, Research and Innovation (SERI).

# References

