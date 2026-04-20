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

Global reanalysis products provide long, spatially consistent records of meteorological conditions, but their native resolution is often too coarse for mountain research [@fanetal2019]. Cryospheric and hydrological models instead require meteorological forcing that reflects strong topographic gradients in temperature, radiation, humidity, wind, and precipitation. Dynamical downscaling can address this problem, but it is often too computationally expensive or complex for routine use in long simulations, sensitivity analyses, or operational workflows.

`MicroPyzzotMet` is a lightweight Python package for downscaling reanalysis data to a target digital elevation model using streamlined, MicroMet-inspired corrections [@liston2006meteorological]. It supports air temperature, incoming shortwave and longwave radiation, relative humidity, precipitation, and wind. The package is designed around standard Python geospatial tools and integrates with cloud-native datasets such as ERA5-Land through EarthDataHub and Zarr storage. Its main goal is to provide a practical and reproducible way to generate terrain-aware forcing data for mountain applications over large domains and long time periods.

# Statement of need

Researchers working on snow, glaciers, permafrost, and alpine hydrology frequently need meteorological forcing at resolutions much finer than those of global or regional reanalysis products. In complex terrain, direct use of coarse fields can misrepresent altitudinal gradients, topographic shading, and precipitation patterns, which then propagate into surface energy-balance, mass-balance, and runoff simulations.

`MicroPyzzotMet` addresses this need by providing an open-source Python implementation of meteorological downscaling methods inspired by `MicroMet` [@liston2006meteorological]. The target audience is researchers who need a terrain-aware forcing workflow that is simpler and lighter than full dynamical downscaling or profile-based interpolation systems, but more physically informed than direct reprojection of coarse fields. The package is intended for multi-decadal simulations, large spatial domains, ensemble experiments, and reproducible preprocessing pipelines where computational efficiency matters.

A key practical feature is the ability to work directly with cloud-native reanalysis and terrain data through EarthDataHub [@EarthDataHub2025], Xarray, and Zarr. This reduces storage and I/O overhead while keeping the workflow compatible with standard Python-based modelling pipelines.

# State of the field

Meteorological downscaling in mountains is supported by a range of empirical and physically based approaches. The original `MicroMet` formulation [@liston2006meteorological] remains influential because it applies physically informed empirical corrections directly to near-surface variables, making it useful when only a limited set of meteorological inputs is available. More recent tools such as `TopoPyScale` [@filhol2023topopyscale], together with the `TopoSCALE` and `TopoSUB` frameworks [@fiddesTopoSCALEDownscalingGridded2014; @fiddesTopoSUBToolEfficient2012], reconstruct fine-scale meteorological variability using richer atmospheric information, terrain-aware interpolation, and clustering strategies. These tools are well suited to detailed alpine applications, but they typically require more preprocessing, more input variables, and more computational effort and the usage is restricted to data with multiple pressure levels.

`MicroPyzzotMet` occupies a complementary niche. It prioritizes minimal required inputs, transparent parameterizations, and straightforward integration into Python workflows for long simulations and large mountain domains. Its purpose is not to replace more detailed terrain-resolving frameworks, but to provide a lighter alternative when the scientific question requires scalable production of forcing data rather than the most complex atmospheric reconstruction.

This distinction is also the main build-versus-contribute justification. Extending an existing high-detail framework to support the simpler, near-surface, cloud-native workflow targeted here would require substantial changes to assumptions about inputs, processing strategy, and software scope. A purpose-built implementation is therefore more appropriate for this use case.

# Software design

`MicroPyzzotMet` follows a configuration-driven design in which a single JSON file defines the spatial domain, time period, target resolution, input sources, and variables to process. This keeps workflows reproducible and makes it easy to rerun the same experiment across regions or periods without modifying code.

The package uses a modular variable-by-variable architecture. Temperature, humidity, radiation, precipitation, and wind are processed through separate routines that share a common pattern: ingest coarse forcing, align it to the target grid, apply vertical or terrain-based corrections, and write standardized outputs. This design reflects an explicit trade-off. Instead of reconstructing full atmospheric profiles or relying on computationally expensive terrain clustering, `MicroPyzzotMet` works directly from commonly available near-surface variables. That choice reduces complexity and input requirements, which is essential for multi-decadal and large-domain applications.

The workflow is also designed to keep memory use bounded. Terrain derivatives are computed once and reused across variables, while climate inputs are processed in monthly chunks with parallel execution across files. This favors robust long-period production runs on standard workstations and keeps the package compatible with downstream cryospheric and hydrological modelling workflows. @fig:workflow summarizes this configuration-driven processing chain from input data to downscaled outputs.

![Overview of the configuration-driven `MicroPyzzotMet` workflow, from terrain and reanalysis inputs through variable-specific downscaling to standardized gridded outputs.](micropyzzotmet_workflow.png){#fig:workflow width=60%}

# Research impact statement

`MicroPyzzotMet` has already been used in two ongoing research applications that required the production of high-resolution forcing datasets over large mountain regions. First, it has been used to downscale air temperature and incoming shortwave radiation to 50 m resolution for the extratropical Andes over 2002--2023. Second, it is being used to generate forcing for a snow reanalysis across the entire Alpine region at 500 m resolution over 1950--2024, including air temperature, incoming shortwave radiation, precipitation, and relative humidity.

These applications are not yet published, but they demonstrate concrete and realized use beyond the software paper itself: in both cases, `MicroPyzzotMet` made it feasible to generate terrain-aware forcing datasets spanning decades and large spatial domains within a reproducible Python workflow. This is precisely the class of problem the package was designed to address. Representative examples of the resulting terrain-aware fields are shown in @fig:examples.

![Examples of terrain-aware downscaled outputs generated with `MicroPyzzotMet`, illustrating the added spatial detail produced over complex mountain topography.](downscaling_examples.png){#fig:examples}

The repository also includes a complete example workflow for the Maipo basin in Chile, showing end-to-end acquisition of terrain and ERA5-Land inputs, configuration-driven processing, and production of gridded outputs. Together, the ongoing Andes and Alpine applications and the reproducible example provide specific evidence of near-term scholarly significance for cryospheric and hydrological research.

# AI usage disclosure

Generative AI tools (Claude and ChatGPT) were used in software development for limited code refactoring and debugging assistance, and in the manuscript and documentation for language editing and limited drafting support. All AI-assisted outputs were reviewed, edited, tested, and validated by the authors, who made the scientific and software-design decisions and take full responsibility for the final software and paper.

# Acknowledgements

This project has received funding from the European Union’s Horizon Research and Innovation Actions programme under Grant Agreement 101180133, and from the Swiss State Secretariat for Education, Research and Innovation (SERI).

# References
