#!/bin/bash
# -----------------------------------------------------------------------------
# This script train.sh runs the Python program src/train.py.
# This program trains a prediction model and saves the final model
# in the model/ directory. The script also logs all execution details
# in the file logs/train.logs.
# -----------------------------------------------------------------------------

this_dir=$(dirname "$0")
script_dir="${this_dir}/../src"

python3 "${script_dir}/train.py"
