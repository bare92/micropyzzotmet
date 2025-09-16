#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 28 15:30:12 2025

@author: rbarella
"""
import json
import os
import subprocess

# Aree e bounding box
areas = {
    "Area01":  (375500, 6819500, 427500, 6878500),
    "Area02":  (375500, 6642500, 427500, 6701500),
    "Area03":  (335000, 6560000, 390500, 6624000),
    "Area04":  (340000, 6417500, 395500, 6478500),
    "Area05":  (390000, 6304000, 448500, 6400000),
    "Area06":  (366000, 6205000, 428500, 6342500),
    "Area07":  (342000, 6084000, 398500, 6158500),
    "Area08":  (323500, 5993500, 375500, 6052500),
    "Area09":  (288000, 5970500, 328500, 6011000),
    "Area10": (271500, 5875500, 323500, 5934500),
}

# Zone -> lapse rates key
zone_map = {
    "north": "Andes_Extratropical_North",
    "center": "Andes_Extratropical_Central",
    "south": "Andes_Extratropical_South"
}
north_areas = {"Area01", "Area02", "Area03"}
center_areas = {"Area04", "Area05", "Area06"}
south_areas = {"Area07", "Area08", "Area09", "Area10"}

# Base paths (da adattare al tuo cluster)
container = "/mnt/CEPH_PROJECTS/SNOWCOP/Paloma"
base_config = "/mnt/CEPH_PROJECTS/SNOWCOP/Riccardo/micropyzzotmet/micro_config_SNOWCOP_DOMAIN.json"
lapse_rates_file = "/mnt/CEPH_PROJECTS/SNOWCOP/Riccardo/micropyzzotmet/auxiliary_data/lapse_rates_doc.json"
run_script = "/mnt/CEPH_PROJECTS/SNOWCOP/Riccardo/micropyzzotmet/run_micromet_SNOWCOP_DOMAIN.sh"

# Carica lapse rates
with open(lapse_rates_file) as f:
    lapse_rates = json.load(f)

# Carica config base
with open(base_config) as f:
    base_cfg = json.load(f)

for area, (xmin, ymin, xmax, ymax) in areas.items():
    if area in north_areas:
        zone = "north"
    elif area in center_areas:
        zone = "center"
    else:
        zone = "south"

    cfg = base_cfg.copy()

    # Working dir
    workdir = os.path.join(container, area, 'Micromet')
    cfg["working_directory"] = workdir

    # Extent
    cfg["download_dem_extent"] = {
        "lat_min": ymin,
        "lat_max": ymax,
        "lon_min": xmin,
        "lon_max": xmax
    }

    # Lapse rates
    cfg["custom_lapse_rates"] = lapse_rates[zone_map[zone]]

    os.makedirs(workdir, exist_ok=True)

    # Config file per area
    config_path = os.path.join(workdir, f"micro_config_{area}.json")
    with open(config_path, "w") as f:
        json.dump(cfg, f, indent=2)

    # Lancia lo .sh passando il config
    print(f"Running Micromet for {area}...")
    subprocess.run([run_script, config_path], check=True)

print("All Micromet runs completed.")
