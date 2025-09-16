#!/bin/bash
set -e

echo "Activating conda environment 'microenv'..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate microenv

SCRIPT_PATH="./main_micromet.py"
CONFIG_PATH="$1"   # <- prende il config passato da Python

echo "Running MicroMet downscaling with config: $CONFIG_PATH"
python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done DK."



