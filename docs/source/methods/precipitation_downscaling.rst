Precipitation Downscaling
=========================

MicroPyzzotMet downscales precipitation by applying simple, flexible 
elevation-based adjustments.  
This adds spatial structure to coarse climate products and allows the 
user to apply regional knowledge or monthly correction factors.

Conceptual Overview
-------------------

Precipitation generally increases with elevation, particularly in 
mountainous regions. Coarse climate datasets cannot represent these 
gradients within complex topography.

Method Summary
--------------

1. **Extract precipitation from climate data.**  
   Typically total accumulated precipitation.

2. **Reproject to the DEM.**  
   Data are interpolated to the resolution of the model domain.

3. **Apply optional elevation adjustments.**  
   Users may specify:
   - monthly scaling factors,  
   - a simple lapse-rate-like increase with elevation, or  
   - no correction at all.

4. **Produce daily or hourly precipitation maps.**  
   Following the temporal resolution of ERA5(-Land).

What This Method Captures
-------------------------

- Broad-scale elevation dependence  
- Spatial consistency with the DEM  
- User knowledge through optional correction factors

What This Method Does Not Capture
---------------------------------

- Orographic lifting dynamics  
- Wind-driven precipitation redistribution  
- Rain–snow partitioning (handled elsewhere if needed)

Output
------

Downscaled precipitation fields are stored in:

::

    P/precipitation_downscaled_YYYY_MM.nc

Summary
-------

This method provides a pragmatic way to refine coarse precipitation 
fields over mountainous terrain, suitable for hydrological and snow 
applications.


