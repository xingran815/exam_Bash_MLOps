#!/bin/bash

# =============================================================================
# This script preprocessed.sh runs the program src/preprocessed.py
# and logs the execution details in the log file
# logs/preprocessed.logs.
# =============================================================================

this_dir=$(dirname "$0")
script_dir="${this_dir}/../src"

python3 "${script_dir}/preprocessed.py"
