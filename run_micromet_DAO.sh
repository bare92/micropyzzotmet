#!/bin/bash

# Exit if any command fails
set -e

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Path to your Python script and config
SCRIPT_PATH="./src/micropyzzotmet/main_micromet.py"
CONFIG_PATH="./option_files/micro_config_DAO.json"

echo "Running MicroMet downscaling..."
./.venv/bin/python -m micropyzzotmet.main_micromet "$CONFIG_PATH"

echo "Done."


