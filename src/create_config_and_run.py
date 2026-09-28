import os
import json
import re
import copy
import traceback
import time
import threading
from datetime import datetime

from micropyzzotmet.main_micromet import run_micropezzomet


json_template = "/root/SEAS5_BC_DS_1KM_ITALY/DAO_template.json"
config_output_dir = "/root/SEAS5_BC_DS_1KM_ITALY/configs"
root_directory = "/share/SEAS5_1KM/italy"

completed_log = "/root/SEAS5_BC_DS_1KM_ITALY/completed.log"


os.makedirs(config_output_dir, exist_ok=True)


with open(json_template, "r") as f:
    config = json.load(f)


pattern = re.compile(r"^EM\d{6}_FM\d{6}$")


folders = sorted([
    folder for folder in os.listdir(root_directory)
    if pattern.match(folder)
    and os.path.isdir(os.path.join(root_directory, folder))
])


print(f"Found {len(folders)} forecast folders")


# ==========================================================
# STEP 1 - CREATE ALL CONFIG FILES
# ==========================================================

config_files = []


for folder in folders:

    forecast_dir = os.path.join(root_directory, folder)

    for n in range(51):

        member_dir = os.path.join(
            forecast_dir,
            f"member_{n:02d}"
        )


        era_file = os.path.join(
            member_dir,
            "inputs",
            "climate",
            f"SEAS5_9km_{folder}_{n}.nc"
        )


        if not os.path.isfile(era_file):
            print(f"[SKIP] Missing {era_file}")
            continue


        run_config = copy.deepcopy(config)

        run_config["working_directory"] = member_dir
        run_config["era_file"] = era_file


        json_name = f"{folder}_m{n:02d}.json"

        config_file = os.path.join(
            config_output_dir,
            json_name
        )


        with open(config_file, "w") as f:
            json.dump(
                run_config,
                f,
                indent=2
            )


        config_files.append(config_file)


print()
print("="*80)
print(f"Created {len(config_files)} configuration files")
print("="*80)



# ==========================================================
# LOAD COMPLETED RUNS
# ==========================================================

completed = set()

if os.path.isfile(completed_log):

    with open(completed_log, "r") as f:
        completed = set(
            line.strip()
            for line in f.readlines()
        )


print(f"Already completed: {len(completed)} runs")



# ==========================================================
# HEARTBEAT
# ==========================================================

running = False


def heartbeat():

    while running:

        print(
            f"[RUNNING] {datetime.now().strftime('%H:%M:%S')} - Micromet still running..."
        )

        time.sleep(60)



# ==========================================================
# STEP 2 - RUN MICROMET
# ==========================================================


total = len(config_files)

print()
print("="*80)
print(f"START MICROMET: {total} configurations")
print("="*80)



for i, config_file in enumerate(config_files, start=1):


    name = os.path.basename(config_file)


    if name in completed:

        print(f"[{i}/{total}] SKIP completed {name}")
        continue



    print()
    print("-"*80)
    print(f"[{i}/{total}] START {name}")
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print("-"*80)


    start = time.time()


    try:

        running = True

        thread = threading.Thread(
            target=heartbeat,
            daemon=True
        )

        thread.start()


        run_micropezzomet(config_file)


        running = False


        elapsed = (time.time()-start)/60


        print()
        print(f"[OK] {name}")
        print(f"Elapsed time: {elapsed:.1f} min")


        with open(completed_log, "a") as f:
            f.write(name+"\n")


        completed.add(name)



    except Exception as e:


        running = False


        elapsed = (time.time()-start)/60


        print()
        print(f"[ERROR] {name}")
        print(f"Elapsed time: {elapsed:.1f} min")
        print(e)


        traceback.print_exc()


        continue



print()
print("="*80)
print("ALL RUNS COMPLETED")
print("="*80)