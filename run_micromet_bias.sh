#!/bin/bash

#!/bin/bash

set -e

echo "Running MicroMet with micromamba..."

SCRIPT_PATH="./main_micromet.py"
CONFIG_PATH="$1"   # <- prende il config passato da Python

micromamba run -n microenv python "$SCRIPT_PATH" "$CONFIG_PATH"

echo "Done DK."


