#!/usr/bin/env python3

"""
-------------------------------------------------------------------------------
This script `preprocessed.py` retrieves data from the latest CSV file created 
in the 'data/raw/' directory.

1. It applies preprocessing to the data.
   
2. The results of the preprocessing are saved in a new CSV file 
   in the 'data/processed/' directory, with a name formatted as 
   'sales_processed_YYYYMMDD_HHMM.csv'.
   
3. All preprocessing steps are logged in the 
   'logs/preprocessed.logs' file to ensure detailed tracking of the process.

Any errors or anomalies are also logged to ensure traceability.
-------------------------------------------------------------------------------
"""
import os
from datetime import datetime, timezone


# get the current utc time
current_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
this_dir = os.path.dirname(os.path.abspath(__file__))
# find out the latest modified file (except sales_data.csv) in data/raw/
raw_dir = os.path.join(this_dir, "../data/raw/")
files = os.listdir(raw_dir)
files = [f for f in files if f.endswith(".csv") and f != "sales_data.csv"]
latest_file = max(files, key=lambda x: os.path.getmtime(os.path.join(raw_dir, x)))

# 1. It applies preprocessing to the data.
# read the model and sales value, make them a dictionary
sales = {}
with open(os.path.join(raw_dir, latest_file), "r") as f:
    next(f)  # skip header line
    for line in f:
        line = line.strip()
        if line:
            timestamp, model, sales_value = line.split(",")
            sales_value = int(sales_value)
            if sales_value < 0:
               sales_value = 0
            sales[model] = sales_value

# 2. Save the processed data to a new CSV file in data/processed/
processed_dir = os.path.join(this_dir, "../data/processed/")
# create this folder if not exist
os.makedirs(processed_dir, exist_ok=True)
# create the processed file name
processed_file = "sales_processed_" + latest_file.split(".")[0][6:] + ".csv"
processed_file = os.path.join(processed_dir, processed_file)
# write the processed data to the file
with open(processed_file, "w") as f:
    f.write(f"rtx3060,rtx3070,rtx3080,rtx3090,rx6700\n")
    f.write(f"{sales['rtx3060']},{sales['rtx3070']},{sales['rtx3080']},{sales['rtx3090']},{sales['rx6700']}\n")

# 3. Log the preprocessing activity in the logs/preprocessed.logs file.
log_file = os.path.join(this_dir, "../logs/preprocessed.logs")
with open(log_file, "a") as f:
    f.write(f"[{current_utc}] Preprocessed {latest_file} to {processed_file.split("/")[-1]}\n")

