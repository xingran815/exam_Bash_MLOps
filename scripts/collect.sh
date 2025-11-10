# ==============================================================================
# Script: collect.sh
# Description:
#   This script queries an API to retrieve sales data for the following graphics card models:
#     - rtx3060
#     - rtx3070
#     - rtx3080
#     - rtx3090
#     - rx6700
#
#   The collected data is appended to a copy of the file:
#     data/raw/sales_data.csv
#
#   The output file is saved in the format:
#     data/raw/sales_YYYYMMDD_HHMM.csv
#   with the following columns:
#     timestamp, model, sales
#
#   Collection activity (requests, queried models, results, errors)
#   is recorded in a log file:
#     logs/collect.logs
#
#   The log should be human-readable and must include:
#     - The date and time of each request
#     - The queried models
#     - The retrieved sales data
#     - Any possible errors
# ==============================================================================

## TODO: record any error!
set -eu
this_dir=$(dirname "$0")

# 1. Set the timestamp for the output file
timestamp=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
output_timestamp=$(date -u +"%Y%m%d_%H%M")

# 2. Query the API for each model and append the results to the output file
declare -A sales
for model in rtx3060 rtx3070 rtx3080 rtx3090 rx6700; do
  response=$(curl -s "http://0.0.0.0:5000/${model}")
  sales[$model]=$response
  echo "${timestamp},${model},${response}" >> "${this_dir}/../data/raw/sales_data.csv"
  echo "${timestamp},${model},${response}" >> "${this_dir}/../data/raw/sales_${output_timestamp}.csv"
done

# 3. Record the collection activity in the log file
echo "[${timestamp}] rtx3060: ${sales[rtx3060]}, rtx3070: ${sales[rtx3070]}, "\
     "rtx3080: ${sales[rtx3080]}, rtx3090: ${sales[rtx3090]}, "\
     "rx6700: ${sales[rx6700]}" >> "${this_dir}/../logs/collect.logs"
