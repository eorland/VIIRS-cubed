#!/usr/bin/env python
# coding: utf-8
"""
run_batch_preprocessing.py

Batch runner for VIIRS swath preprocessing (Step 1) over a set of fire
perimeters stored in a parquet file. For each fire, process_swaths() is
called with a date window of +/- 14 days around the discovery/containment
times and the fire's bounding box.

State tracking columns are written back to the parquet after each fire:
    has_run               : bool  — completed successfully
    has_error             : bool  — threw an exception
    output_dir_path       : str   — path to Step1_Compiled_Swaths/ output
    last_run_timestamp    : str   — ISO timestamp of last attempt

Usage
-----
# Dry run (skip already-run fires):
python run_batch_preprocessing.py --output_dir VIIRS-cubed-outputs

# Force rerun of all fires (including those with has_run=True):
python run_batch_preprocessing.py --output_dir VIIRS-cubed-outputs --overwrite

# With optional passthrough arguments:
python ~/VIIRS-cubed/run_batch_preprocessing.py \
    --output_dir 's3://maap-ops-workspace/shared/gsfc_landslides/L1_VIIRS_Swath_Outputs' \
    --sensors '["SNPP", "NOAA20", "NOAA21"]'     
"""

import argparse
import ast
import datetime as dt
import os
import traceback

import pandas as pd
import geopandas as gpd
import s3fs
import sys

# Ensure repo root is on path so sibling modules resolve correctly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from swath_preprocessing import process_swaths



# ===========================================================================
# CONSTANTS
# ===========================================================================

# PARQUET_PATH = os.path.join(os.path.dirname(__file__), 'nifc_perimeters_to_run.parquet')
PARQUET_PATH = 's3://maap-ops-workspace/shared/gsfc_landslides/nifc_perimeters_to_run.parquet'
WINDOW_DAYS = 14

STATE_COLS = {
    'has_run': False,
    'has_error': False,
    'output_dir_path': None,
    'last_run_timestamp': None,
}


# ===========================================================================
# HELPERS
# ===========================================================================

def _s3_path(path: str) -> str:
    """Strip the s3:// prefix for use with s3fs directly."""
    return path[len("s3://"):]

def load_parquet(path: str) -> gpd.GeoDataFrame:
    """Load the perimeters parquet and initialise any missing state columns."""
    if path.startswith("s3://"):
        fs = s3fs.S3FileSystem(anon=False)
        with fs.open(_s3_path(path), 'rb') as f:
            df = gpd.read_parquet(f)
    else:
        df = gpd.read_parquet(path)

    for col, default in STATE_COLS.items():
        if col not in df.columns:
            df[col] = default
    df['has_run'] = df['has_run'].fillna(False).astype(bool)
    df['has_error'] = df['has_error'].fillna(False).astype(bool)
    return df


def save_parquet(df: gpd.GeoDataFrame, path: str) -> None:
    """Write the updated GeoDataFrame back to parquet.
    
    For S3 paths: write to a local temp file first, then upload to S3.
    This ensures the live S3 object is never partially overwritten.
    """
    if path.startswith("s3://"):
        import tempfile
        fs = s3fs.S3FileSystem(anon=False)
        with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            df.to_parquet(tmp_path, index=False)          # safe local write
            fs.put(tmp_path, _s3_path(path))              # upload only if write succeeded
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)                       # always clean up
    else:
        tmp_path = path + ".tmp"
        try:
            df.to_parquet(tmp_path, index=False)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise


def make_fire_name(incident_name: str) -> str:
    """Convert poly_IncidentName to a filesystem-safe fire_name."""
    return str(incident_name).strip().replace(' ', '_')


# def build_output_dir_path(output_dir: str, fire_name: str) -> str:
#     """Return the expected Step 1 output data path for a given fire."""
#     return os.path.join(
#         os.path.abspath(output_dir),
#         f"{fire_name}_Gridded_VIIRS",
#         "Data",
#         "Step1_Compiled_Swaths",
#     )

def build_output_dir_path(output_dir: str, fire_name: str) -> str:
    """Return the expected Step 1 output data path for a given fire."""
    base = output_dir if output_dir.startswith("s3://") else os.path.abspath(output_dir)
    return "/".join([base.rstrip("/"), f"{fire_name}_Gridded_VIIRS", "Data", "Step1_Compiled_Swaths"])


def parse_bbox(bbox_str: str) -> list:
    """Parse a bbox string 'minx,miny,maxx,maxy' into a list of floats."""
    return [float(x) for x in bbox_str.split(',')]


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch VIIRS swath preprocessing over NIFC fire perimeters."
    )

    # --- Output / infra ---
    parser.add_argument(
        '--output_dir', type=str, default='VIIRS-cubed-outputs',
        help='Base output directory (local path or S3 URI). Default: VIIRS-cubed-outputs'
    )
    parser.add_argument(
        '--overwrite', action='store_true', default=False,
        help='Rerun fires where has_run=True and overwrite existing Step 1 files.'
    )

    # --- Passthrough to process_swaths ---
    parser.add_argument(
        '--sensors', type=str, default='["SNPP", "NOAA20", "NOAA21"]',
        help='Sensors to include. E.g. \'["SNPP", "NOAA20"]\''
    )
    parser.add_argument(
        '--pix_lut_path', type=str, default=None,
        help='Path to pixel size lookup table CSV.'
    )
    parser.add_argument(
        '--make_plots', action='store_true', default=False,
        help='Generate and save overview plots for each swath.'
    )
    parser.add_argument(
        '--no_spatial_test', action='store_true', default=False,
        help='Skip spatial alignment verification.'
    )
    parser.add_argument(
        '--n_timesteps', type=int, default=-1,
        help='Max timesteps to process per fire. -1 = all. Default: -1'
    )

    args = parser.parse_args()
    sensors = ast.literal_eval(args.sensors)

    # -----------------------------------------------------------------------
    # LOAD PARQUET
    # -----------------------------------------------------------------------
    print(f"Loading perimeters from: {PARQUET_PATH}")
    df = load_parquet(PARQUET_PATH)
    n_total = len(df)
    print(f"  {n_total} fires loaded.\n")

    # -----------------------------------------------------------------------
    # BATCH LOOP
    # -----------------------------------------------------------------------
    n_run = 0
    n_skipped = 0
    n_error = 0

    for idx, row in df.iterrows():

        fire_name = make_fire_name(row['poly_IncidentName'])

        # --- Skip logic ---
        if row['has_run'] and not args.overwrite:
            print(f"[{idx}] SKIP (has_run=True): {fire_name}")
            n_skipped += 1
            continue

        # --- Derive date window ---
        start = (row['parsed_discovery_time'] - pd.Timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d')
        end = (row['parsed_containment_time'] + pd.Timedelta(days=WINDOW_DAYS)).strftime('%Y-%m-%d')

        # --- Parse bbox ---
        bbox = parse_bbox(row['bbox'])

        # --- Expected output path (written regardless of success/failure) ---
        step1_data_path = build_output_dir_path(args.output_dir, fire_name)

        print(f"[{idx}] RUNNING: {fire_name}")
        print(f"       Period : {start}  →  {end}")
        print(f"       BBox   : {bbox}")
        print(f"       Output : {step1_data_path}")

        run_timestamp = dt.datetime.now().isoformat(timespec='seconds')

        try:
            process_swaths(
                fire_name=fire_name,
                start=start,
                end=end,
                bbox=bbox,
                n_timesteps=args.n_timesteps,
                pix_lut_path=args.pix_lut_path,
                sensors=sensors,
                make_plots=args.make_plots,
                save_data=True,
                overwrite=args.overwrite,
                run_spatial_test=not args.no_spatial_test,
                output_dir=args.output_dir,
            )

            # SUCCESS — update state
            df.at[idx, 'has_run'] = True
            df.at[idx, 'has_error'] = False
            df.at[idx, 'output_dir_path'] = step1_data_path
            df.at[idx, 'last_run_timestamp'] = run_timestamp
            n_run += 1
            print(f"       Completed {fire_name}\n")

        except Exception as e:
            # FAILURE — flag but do not halt the batch
            df.at[idx, 'has_run'] = False
            df.at[idx, 'has_error'] = True
            df.at[idx, 'last_run_timestamp'] = run_timestamp
            n_error += 1
            print(f"       ✗ ERROR: {fire_name}")
            print(f"         {type(e).__name__}: {e}")
            traceback.print_exc()
            print()

        finally:
            # Write parquet after every fire so progress survives a crash
            save_parquet(df, PARQUET_PATH)

    # -----------------------------------------------------------------------
    # SUMMARY
    # -----------------------------------------------------------------------
    print("=" * 60)
    print(f"Batch complete.")
    print(f"  Fires run successfully : {n_run}")
    print(f"  Fires skipped          : {n_skipped}")
    print(f"  Fires with errors      : {n_error}")
    print(f"  Total                  : {n_total}")
    print("=" * 60)


if __name__ == '__main__':
    main()