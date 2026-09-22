#!/bin/bash
# run_viirs_batch.sh
# DPS run script for VIIRS batch swath preprocessing.
#
# Lives in: VIIRS-cubed/MAAP_DPS/
# Imports in run_batch_preprocessing.py resolve via PYTHONPATH
# set to the repo root (VIIRS-cubed/) where utils.py,
# swath_preprocessing.py and standardizations.py live.
#
# Positional arguments (order must match DPS registration):
#   $1  output_dir      S3 URI or local path for outputs
#   $2  sensors         JSON list e.g. '["SNPP","NOAA20","NOAA21"]'
#   $3  overwrite       "true" | "false"
#   $4  n_timesteps     integer (-1 = all)
#   $5  make_plots      "true" | "false"
#   $6  no_spatial_test "true" | "false"

set -e

# ---------------------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------------------

basedir=$(dirname "$(readlink -f "$0")")

# Repo root is one level up from MAAP_DPS/ — this is where utils.py,
# swath_preprocessing.py and standardizations.py live
repodir=$(dirname "${basedir}")

# ---------------------------------------------------------------------------
# PARSE ARGUMENTS (with defaults)
# ---------------------------------------------------------------------------

OUTPUT_DIR="${1:-s3://maap-ops-workspace/shared/gsfc_landslides/L1_VIIRS_Swath_Outputs}"
SENSORS="${2:-[\"SNPP\", \"NOAA20\", \"NOAA21\"]}"
OVERWRITE="${3:-false}"
N_TIMESTEPS="${4:--1}"
MAKE_PLOTS="${5:-false}"
NO_SPATIAL_TEST="${6:-false}"

# ---------------------------------------------------------------------------
# LOG INPUTS
# ---------------------------------------------------------------------------

echo "========================================================"
echo "VIIRS Batch Preprocessing — DPS Run"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "========================================================"
echo "  output_dir      : ${OUTPUT_DIR}"
echo "  sensors         : ${SENSORS}"
echo "  overwrite       : ${OVERWRITE}"
echo "  n_timesteps     : ${N_TIMESTEPS}"
echo "  make_plots      : ${MAKE_PLOTS}"
echo "  no_spatial_test : ${NO_SPATIAL_TEST}"
echo "  basedir         : ${basedir}"
echo "  repodir         : ${repodir}"
echo "========================================================"

# ---------------------------------------------------------------------------
# BUILD OPTIONAL FLAGS
# ---------------------------------------------------------------------------

EXTRA_FLAGS=""
if [ "${OVERWRITE}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --overwrite"
fi
if [ "${MAKE_PLOTS}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --make_plots"
fi
if [ "${NO_SPATIAL_TEST}" = "true" ]; then
    EXTRA_FLAGS="${EXTRA_FLAGS} --no_spatial_test"
fi

# ---------------------------------------------------------------------------
# OUTPUT DIR (DPS preserves anything written here)
# ---------------------------------------------------------------------------

mkdir -p output

# ---------------------------------------------------------------------------
# PYTHONPATH — expose repo root so local module imports resolve correctly
# ---------------------------------------------------------------------------

export PYTHONPATH="${repodir}:${PYTHONPATH}"

# ---------------------------------------------------------------------------
# RUN
# ---------------------------------------------------------------------------

conda run --live-stream --name viirs_cubed \
    python "${basedir}/run_batch_preprocessing.py" \
        --output_dir "${OUTPUT_DIR}" \
        --sensors "${SENSORS}" \
        --n_timesteps "${N_TIMESTEPS}" \
        ${EXTRA_FLAGS}

echo "========================================================"
echo "$(date -u '+%Y-%m-%d %H:%M:%S UTC') Run script complete."
echo "========================================================"