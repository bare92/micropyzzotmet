#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

unset PROJ_DATA
unset PROJ_LIB
unset GDAL_DATA
unset GDAL_DRIVER_PATH
unset PYTHONPATH

PYTHON="$PROJECT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: Virtual environment is missing. Run: uv sync"
    exit 1
fi

echo "Checking Rasterio CRS support..."
"$PYTHON" - <<'PY'
import rasterio
from rasterio.crs import CRS
print("Rasterio:", rasterio.__version__)
print("GDAL:", rasterio.__gdal_version__)
print("PROJ:", rasterio.__proj_version__)
print("CRS:", CRS.from_epsg(4326))
PY

CONFIG_PATH="$PROJECT_DIR/option_files/micro_config_DEMO_MAIPO.json"

echo "Running MicroMet downscaling..."
"$PYTHON" -m micropyzzotmet.main_micromet "$CONFIG_PATH"

echo "Done."


