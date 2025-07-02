#!/bin/bash

# Exit if any command fails
set -e

# Activate conda environment
echo "Activating conda environment 'swe3'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate swe3

# Path to your Python script and config
SCRIPT_PATH="./main_micromet.py"
CONFIG_PATH="./micro_config_MENDOZA.json"

echo "Running MicroMet downscaling..."
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done DK."


