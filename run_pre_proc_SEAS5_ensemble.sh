#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_SCRIPT="${SCRIPT_DIR}/split_seas5_members.py"
CONFIG_FILE="${SCRIPT_DIR}/split_seas5_members.json"

echo "============================================================"
echo "SEAS5 ensemble member split"
echo "============================================================"
echo "Python script : ${PYTHON_SCRIPT}"
echo "Configuration : ${CONFIG_FILE}"
echo "============================================================"

python3 "${PYTHON_SCRIPT}" "${CONFIG_FILE}"

echo "============================================================"
echo "Completed"
echo "============================================================"