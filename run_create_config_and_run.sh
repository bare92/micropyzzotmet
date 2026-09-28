#!/bin/bash

source /root/miniconda3/etc/profile.d/conda.sh
conda activate seasonal_giulia

CONFIG="/root/SEAS5_BC_DS_1KM_ITALY/DAO_template.json"
LOG="/root/SEAS5_BC_DS_1KM_ITALY/run_1km.log"

python /root/SEAS5_BC_DS_1KM_ITALY/create_config_and_run_1Km.py "$CONFIG" 2>&1 | tee "$LOG"