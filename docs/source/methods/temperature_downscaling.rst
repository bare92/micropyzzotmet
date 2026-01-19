Temperature Downscaling
=======================

MicroPyzzotMet downscales 2-meter air temperature by applying 
elevation-dependent corrections and transferring coarse-resolution 
climate fields onto the high-resolution DEM domain.

Conceptual Overview
-------------------

Air temperature decreases with elevation according to a lapse rate, 
and this vertical gradient is a dominant control in mountainous regions.  
Because coarse datasets like ERA5(-Land) cannot resolve local variations 
in elevation, a correction is necessary before the data can be used in 
snow and hydrological models.

Method Summary
--------------

1. **Extract temperature from coarse climate data.**  
   The model reads 2-meter air temperature from ERA5 or other input files.

2. **Apply a lapse-rate correction.**  
   Temperature is adjusted using:
   - a default monthly lapse-rate climatology,  
   - user-defined custom lapse rates, or  
   - a dynamically calibrated lapse rate (optional).

3. **Reconstruct the temperature field at DEM scale.**  
   The coarse temperature field is shifted using the difference between 
   ERA elevation and local DEM elevation, adding spatial detail that 
   reflects real topography.

4. **Reproject to the DEM.**  
   Values are interpolated onto the DEM grid and masked using DEM 
   nodata values.

5. **Produce daily or hourly temperature maps.**  
   Output follows the same temporal resolution as the input data.

What This Method Captures
-------------------------

- Elevation dependence of near-surface air temperature  
- Domain-wide temperature gradients  
- Fine-scale temperature variations linked to terrain

What This Method Does Not Capture
---------------------------------

- Cold-air pooling, inversions, or valley fog  
- Microclimatic effects such as shading or canopy cover  
- Lateral advection or wind-driven variability

Output
------

Downscaled temperature fields are written to:

::

    Temperature/temperature_downscaled_YYYY_MM.nc

Summary
-------

The temperature downscaling module provides a physically transparent, 
computationally efficient method to distribute coarse temperature fields 
over complex terrain, consistent with the Micromet philosophy.



