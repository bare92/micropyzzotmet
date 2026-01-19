Shortwave Radiation Downscaling
===============================

MicroPyzzotMet downscales incoming shortwave radiation by accounting for 
local terrain orientation (slope and aspect) and solar geometry.  
This introduces realistic spatial contrasts that coarse-resolution 
products cannot resolve.

Conceptual Overview
-------------------

Shortwave radiation is strongly influenced by terrain:

- slopes facing the sun receive more radiation,  
- shaded areas receive less,  
- the sun's position changes daily and seasonally.

Coarse datasets such as ERA5(-Land) provide horizontal irradiance but 
lack the terrain information needed to represent these effects.

Method Summary
--------------

1. **Extract shortwave radiation from input data.**  
   Typically, ERA5 surface solar radiation downwards (SSRD).

2. **Reproject to the DEM grid.**  
   The coarse shortwave field is interpolated to the DEM resolution.

3. **Determine solar position.**  
   Solar zenith and azimuth angles are computed using the domain's 
   geographic location and time.

4. **Apply topographic corrections.**  
   Shortwave radiation is adjusted using:
   - slope and aspect from the DEM,  
   - solar incidence angle on the terrain, and  
   - diffuse vs. direct-beam characteristics.

5. **Generate hourly or daily shortwave fields.**  
   Matching the original temporal resolution.

What This Method Captures
-------------------------

- Terrain orientation effects  
- Spatial variability linked to topography  
- Seasonal/daily solar geometry changes

What This Method Does Not Capture
---------------------------------

- Cloud shading at sub-grid scales  
- Horizon shading from distant mountains (sky-view factor not included)  
- Multiple reflections or albedo feedbacks

Output
------

Shortwave radiation fields are stored in:

::

    SW/shortwave_downscaled_YYYY_MM.nc

Summary
-------

The shortwave module provides terrain-aware solar radiation estimates, 
enhancing the physical realism of energy-balance modeling and snowmelt 
simulations.


