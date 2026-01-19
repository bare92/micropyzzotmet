Wind Downscaling
================

MicroPyzzotMet downscales wind speed and direction by combining coarse 
climate wind fields with terrain characteristics such as slope, aspect, 
and curvature.  
This yields a more realistic wind pattern across mountainous terrain.

Conceptual Overview
-------------------

Wind flow interacts strongly with terrain:
- slopes can accelerate or decelerate winds,
- ridges and valleys channel the flow,
- curvature influences convergence and divergence.

Coarse-resolution wind fields smooth over these effects.

Method Summary
--------------

1. **Extract wind components (u and v).**  
   Horizontal wind fields are read from ERA5(-Land).

2. **Reproject wind fields to the DEM.**  
   Both components are interpolated to the domain.

3. **Compute local terrain metrics.**  
   Slope, aspect, and curvature derived from the DEM serve as proxies 
   for wind exposure and channeling.

4. **Adjust wind speed using terrain weighting.**  
   Wind speed is increased on exposed slopes and decreased in sheltered 
   terrain.

5. **Adjust wind direction.**  
   Flow direction is nudged toward terrain-driven steering influenced by 
   slope and curvature.

6. **Generate time-dependent wind fields.**

What This Method Captures
-------------------------

- Enhanced winds on ridges and exposed slopes  
- Reduced winds in sheltered areas  
- Terrain-driven modification of wind direction

What This Method Does Not Capture
---------------------------------

- Complex flow separation or turbulence  
- Thermal winds or katabatic flows  
- Obstacles such as vegetation or buildings

Output
------

Wind fields are written to:

::

    Wind/wind_speed_direction_YYYY_MM.nc

Summary
-------

This module introduces terrain-dependent refinement to coarse wind 
fields, improving their suitability for snow transport studies, 
evaporation modeling, and energy balance applications.


