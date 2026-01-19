Longwave Radiation Downscaling
==============================

MicroPyzzotMet downscales downwelling longwave radiation (LWR) from 
coarse-resolution climate data by reconstructing key atmospheric 
properties—temperature, humidity, cloudiness, and emissivity—and 
projecting these conditions onto the high-resolution DEM grid.

The approach follows the conceptual framework of MicroMet, but uses a 
lighter and more flexible formulation suitable for operational 
downscaling pipelines.

Conceptual Overview
-------------------

Longwave radiation reaching the surface depends primarily on:

- air temperature,
- atmospheric moisture content,
- cloud fraction, and
- the effective emissivity of the atmosphere.

Because longwave radiation is mostly diffuse and originates from the 
entire sky dome, terrain effects such as slope and aspect are far less 
important than for shortwave radiation.  
MicroPyzzotMet therefore focuses on improving the *atmospheric* 
representation rather than applying terrain-based corrections.

Method Summary
--------------

The downscaling procedure consists of:

1. **Reading temperature and dew point** from ERA5(-Land).  
   These fields provide the basis for estimating atmospheric humidity 
   and emissivity.

2. **Estimating atmospheric profiles.**  
   Temperature and humidity are adjusted with elevation using monthly or 
   user-specified lapse rates. This allows MicroPyzzotMet to represent 
   vertical gradients that are not resolved in coarse ERA5 data.

3. **Diagnosing cloud fraction.**  
   Cloudiness is inferred from mid-tropospheric humidity, providing a 
   smooth transition between clear-sky and overcast conditions.

4. **Computing atmospheric emissivity.**  
   Emissivity is evaluated using empirical relationships that depend on 
   vapour pressure, cloudiness, and elevation. This determines how 
   efficiently the atmosphere emits longwave radiation.

5. **Estimating longwave radiation.**  
   Longwave fluxes are computed using the standard Stefan–Boltzmann 
   formulation based on reconstructed emissivity and temperature.

6. **Reprojecting to the DEM.**  
   The coarse longwave field is interpolated to the DEM grid and masked 
   using DEM nodata values.

7. **Writing monthly output files.**  
   The final product is a time series of longwave radiation maps aligned 
   with the DEM.

What This Method Captures
-------------------------

- Warmer and more humid air emits more longwave radiation.
- Cloudy conditions strongly enhance longwave fluxes.
- Elevation influences atmospheric moisture and emissivity.
- Spatial detail is added through DEM-based reprojection.

What This Method Does *Not* Do
------------------------------

- No terrain shading or slope corrections (appropriate for longwave).  
- No explicit sky-view factor adjustments.  
- No radiative transfer modelling—this is a practical, empirical method.

Output
------

Downscaled longwave radiation fields are written to:

::

    longwave_downscaled_YYYY_MM.nc

Each file contains DEM-aligned longwave radiation (W m⁻²) together with 
time information and metadata documenting the settings used.

Summary
-------

The MicroPyzzotMet longwave method provides a balance between physical 
realism and computational simplicity. By reconstructing key atmospheric 
properties before reprojection, the method produces more realistic 
longwave inputs for snow and hydrological models than a simple spatial 
interpolation of ERA5 values.


