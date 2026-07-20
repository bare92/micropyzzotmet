#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Clear all PROJ/GDAL env vars that conda activation may have injected.
# The venv's Python handles its own data-path initialization via os.environ.pop
# at startup (see main_micromet.py), but unsetting here keeps the process
# environment clean from the start and avoids interference with GDAL CLI tools.
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

CONFIG_PATH="$PROJECT_DIR/option_files/micro_config_DAO.json"

echo "Running MicroMet downscaling..."
"$PYTHON" -m micropyzzotmet.main_micromet "$CONFIG_PATH"

echo "Done."


