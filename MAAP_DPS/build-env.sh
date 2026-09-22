#!/bin/bash
# build-env.sh
# Sets up the custom conda environment for VIIRS batch preprocessing.
# Called by DPS before the run script is executed.

set -e  # exit immediately on error

basedir=$(dirname "$(readlink -f "$0")")

echo "[build-env] $(date -u '+%Y-%m-%d %H:%M:%S UTC') Starting environment setup"
echo "[build-env] Script directory: ${basedir}"
echo "[build-env] Conda version: $(conda --version)"

# ---------------------------------------------------------------------------
# CREATE OR UPDATE THE ENVIRONMENT
# ---------------------------------------------------------------------------

if conda env list | grep -q "^viirs_cubed "; then
    echo "[build-env] Environment 'viirs_cubed' exists — updating..."
    conda env update --name viirs_cubed \
                     --file "${basedir}/environment.yml" \
                     --prune
else
    echo "[build-env] Creating environment 'viirs_cubed'..."
    conda env create --name viirs_cubed \
                     --file "${basedir}/environment.yml"
fi

echo "[build-env] $(date -u '+%Y-%m-%d %H:%M:%S UTC') Environment ready."