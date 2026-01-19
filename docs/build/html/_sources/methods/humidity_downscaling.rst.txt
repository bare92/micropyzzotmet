Humidity Downscaling
====================

MicroPyzzotMet downscales relative humidity by combining coarse climate 
fields with DEM-based elevation adjustments.  
This provides a spatially detailed humidity field necessary for 
longwave radiation and snowmelt modeling.

Conceptual Overview
-------------------

Relative humidity depends on:
- air temperature,
- atmospheric moisture content,
- elevation and pressure.

Because coarse datasets do not capture fine-scale elevation differences, 
their humidity fields must be adjusted before being applied to a DEM.

Method Summary
--------------

1. **Read dew point temperature from climate data.**  
   Dew point provides a proxy for atmospheric moisture.

2. **Apply an elevation correction.**  
   Dew point is adjusted using a simplified relationship between 
   moisture content and elevation.

3. **Combine downscaled dew point with temperature.**  
   Using the downscaled temperature field, relative humidity is 
   recomputed at each DEM pixel.

4. **Reproject and mask.**  
   Data are interpolated to the DEM and invalid regions are masked.

What This Method Captures
-------------------------

- Humidity decrease with elevation  
- Spatial variability driven by terrain  
- Consistency with downscaled temperature

What This Method Does Not Capture
---------------------------------

- Microclimatic humidity patterns  
- Canopy and vegetation influences  
- Local evaporation effects

Output
------

Downscaled humidity fields are stored in:

::

    RH/humidity_downscaled_YYYY_MM.nc

Summary
-------

This module provides humidity fields consistent with both the DEM and 
downscaled temperature, forming a key component for longwave radiation 
and snowmelt modeling.


