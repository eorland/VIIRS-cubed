import ast
import datetime as dt
import os

from swath_preprocessing import process_swaths
from standardizations import standardize_swaths
from utils import log_message


def process_single_fire(
    # --- Shared required parameters ---
    fire_name,
    start,
    end,
    bbox,
    n_timesteps=-1,

    # --- Step 1 parameters (process_swaths) ---
    pix_lut_path=None,
    sensors=['SNPP', 'NOAA20', 'NOAA21'],
    make_plots=False,
    save_data=True,
    overwrite=False,
    run_spatial_test=True,

    # --- Step 2 parameters (standardize_swaths) ---
    grid_region='conus',
    grid_resolution=375,
    batch_size=50,
    grid_pad=10,
    remove_bowtie=False,
    deduplicate_scans=False,

    # --- Shared output/infra parameters ---
    copy_to_s3=False,
    s3_prefix=None,
    output_dir='VIIRS-cubed-outputs',
    remove_local=False,

    # --- Persistence parameters (compute_fire_persistence_baseline via standardize_swaths) ---
    add_persistence=False,
    persistence_fire_mask_col=None,
    persistence_suffix=None,
    persistence_start_threshold=6,
    persistence_end_threshold=6,

    # --- Workflow-level logging ---
    save_workflow_log=True,
):
    '''Preprocess and standardize VIIRS swath data for a single fire/region.

    Runs Step 1 (swath preprocessing) followed by Step 2 (gridding into a
    Zarr datacube). Optionally computes fire persistence metrics at the end
    of Step 2 before any S3 upload or local removal.

    Parameters
    ----------
    fire_name : str
        Name of the fire or region; used for output directory naming.
    start : str
        Start date in YYYY-MM-DD format.
    end : str
        End date in YYYY-MM-DD format.
    bbox : list
        Bounding box as [lon_min, lat_min, lon_max, lat_max] in EPSG:4326.
    n_timesteps : int, optional
        Number of overpasses to process. -1 processes all (default).
    pix_lut_path : str or None, optional
        Local path to the VIIRS pixel-size lookup table CSV.
    sensors : list of str, optional
        Satellites to include. Default: ['SNPP', 'NOAA20', 'NOAA21'].
    make_plots : bool, optional
        Generate diagnostic plots at both steps. Default: False.
    save_data : bool, optional
        Save Step 1 NetCDF output files. Default: True.
    overwrite : bool, optional
        Reprocess files that already exist. Default: False.
    run_spatial_test : bool, optional
        Run spatial alignment test in Step 1. Default: True.
    grid_region : str, optional
        Reference grid region: 'conus', 'global', or 'custom'. Default: 'conus'.
    grid_resolution : int, optional
        Reference grid cell size in meters. Default: 375.
    batch_size : int, optional
        Swaths to accumulate before flushing to the Zarr store. Default: 50.
    grid_pad : int, optional
        Extra grid cells of padding around the bounding box. Default: 10.
    remove_bowtie : bool, optional
        Remove bowtie-affected pixels before aggregation. Default: False.
    deduplicate_scans : bool, optional
        Deduplicate overlapping scan lines. Default: False.
    copy_to_s3 : bool, optional
        Upload outputs to S3 after processing. Default: False.
    s3_prefix : str or None, optional
        S3 destination prefix. Required when copy_to_s3=True.
    output_dir : str, optional
        Local base output directory. Default: 'VIIRS-cubed-outputs'.
    remove_local : bool, optional
        Delete local files after S3 upload. Requires copy_to_s3=True.
    add_persistence : bool, optional
        Compute fire persistence metrics after gridding. Default: False.
    persistence_fire_mask_col : str or None, optional
        Fire mask column for persistence. If None (default), runs both
        'fire_mask_max' (suffix 'max') and 'fire_mask_area_weighted_majority'
        (suffix 'aw').
    persistence_suffix : str or None, optional
        Output variable suffix. Required when persistence_fire_mask_col is set.
    persistence_start_threshold : int, optional
        Fire mask value threshold for ignition detection. Default: 6.
    persistence_end_threshold : int, optional
        Fire mask value threshold for sustained detection. Default: 6.
    save_workflow_log : bool, optional
        If True, a single log file spanning both steps is written to
        ``{output_dir}/{fire_name}_Gridded_VIIRS/Logs/`` alongside the
        per-step logs. Filename: ``{fire_name}_workflow_log_{timestamp}.txt``.
        Default: True.
    '''
    if copy_to_s3 and s3_prefix is None:
        raise ValueError("s3_prefix is required when copy_to_s3=True")
    if remove_local and not copy_to_s3:
        raise ValueError("remove_local=True requires copy_to_s3=True")
    if persistence_fire_mask_col is not None and persistence_suffix is None:
        raise ValueError("persistence_suffix is required when persistence_fire_mask_col is set")

    # Open workflow-level log if requested
    wf_log = None
    if save_workflow_log:
        logs_dir = os.path.join(
            os.path.abspath(output_dir), f"{fire_name}_Gridded_VIIRS", "Logs"
        )
        os.makedirs(logs_dir, exist_ok=True)
        run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        workflow_log_path = os.path.join(
            logs_dir, f"{fire_name}_workflow_log_{run_timestamp}.txt"
        )
        wf_log = open(workflow_log_path, 'a')
        log_message("=" * 70, wf_log, include_timestamp=False)
        log_message("VIIRS WORKFLOW LOG", wf_log, include_timestamp=False)
        log_message(f"Run started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                    wf_log, include_timestamp=False)
        log_message("=" * 70, wf_log, include_timestamp=False)
        log_message(f"Fire name: {fire_name}", wf_log)
        log_message(f"Date range: {start} to {end}", wf_log)
        log_message(f"BBOX: {bbox}", wf_log)
        log_message("", wf_log, include_timestamp=False)

    try:
        # ===================================================================
        # STEP 1: SWATH PREPROCESSING
        # ===================================================================

        if wf_log:
            log_message("--- STEP 1: Swath Preprocessing ---", wf_log)
        print(f"\n{'='*60}")
        print(f"STEP 1: Preprocessing swaths for {fire_name}")
        print(f"{'='*60}\n")

        process_swaths(
            fire_name=fire_name,
            start=start,
            end=end,
            bbox=bbox,
            n_timesteps=n_timesteps,
            pix_lut_path=pix_lut_path,
            sensors=sensors,
            make_plots=make_plots,
            save_data=save_data,
            overwrite=overwrite,
            run_spatial_test=run_spatial_test,
            copy_to_s3=copy_to_s3,
            s3_prefix=s3_prefix,
            output_dir=output_dir,
            remove_local=remove_local,
            log_file=wf_log,
        )

        # ===================================================================
        # STEP 2: STANDARDIZATION (+ optional persistence)
        # ===================================================================

        if wf_log:
            log_message("", wf_log, include_timestamp=False)
            log_message("--- STEP 2: Standardization ---", wf_log)
        print(f"\n{'='*60}")
        print(f"STEP 2: Standardizing swaths for {fire_name}")
        print(f"{'='*60}\n")

        standardize_swaths(
            fire_name=fire_name,
            bbox=bbox,
            start=start,
            end=end,
            n_timesteps=n_timesteps,
            grid_region=grid_region,
            grid_resolution=grid_resolution,
            overwrite=overwrite,
            make_plots=make_plots,
            copy_to_s3=copy_to_s3,
            s3_prefix=s3_prefix,
            batch_size=batch_size,
            grid_pad=grid_pad,
            remove_bowtie=remove_bowtie,
            deduplicate_scans=deduplicate_scans,
            output_dir=output_dir,
            remove_local=remove_local,
            add_persistence=add_persistence,
            persistence_fire_mask_col=persistence_fire_mask_col,
            persistence_suffix=persistence_suffix,
            persistence_start_threshold=persistence_start_threshold,
            persistence_end_threshold=persistence_end_threshold,
            log_file=wf_log,
        )

    finally:
        if wf_log is not None:
            log_message("", wf_log, include_timestamp=False)
            log_message(f"Workflow completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                        wf_log, include_timestamp=False)
            wf_log.close()
            print(f"\n{'='*60}")
            print(f"Workflow log saved: {workflow_log_path}")
            print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Full VIIRS pipeline: preprocess swaths and standardize onto a reference grid."
    )

    # ===================================================================
    # REQUIRED ARGUMENTS
    # ===================================================================

    parser.add_argument(
        "--fire_name",
        type=str,
        required=True,
        help="Name of the fire/region (used for output directory naming). E.g. 'Stanford_Flaring'"
    )
    parser.add_argument(
        "--start",
        type=str,
        required=True,
        help="Start date in YYYY-MM-DD format. E.g. '2026-05-10'"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date in YYYY-MM-DD format. E.g. '2026-08-30'"
    )
    parser.add_argument(
        "--bbox",
        type=str,
        required=True,
        help="Bounding box as '[xmin, ymin, xmax, ymax]'. E.g. --bbox '[-112.25, 32.25, -111.25, 33.25]'"
    )

    # ===================================================================
    # OPTIONAL ARGUMENTS
    # ===================================================================

    parser.add_argument(
        "--n_timesteps",
        type=int,
        default=-1,
        help="Number of overpasses to process. Use -1 (default) to process all."
    )
    parser.add_argument(
        "--pix_lut_path",
        type=str,
        default=None,
        help="Local path to VIIRS pixel-size lookup table CSV."
    )
    parser.add_argument(
        "--sensors",
        type=str,
        default='["SNPP", "NOAA20", "NOAA21"]',
        help="JSON list of satellites to include. Default: '[\"SNPP\", \"NOAA20\", \"NOAA21\"]'."
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        default=False,
        help="If set, generate diagnostic plots at both pipeline steps."
    )
    parser.add_argument(
        "--no_save_data",
        action="store_true",
        default=False,
        help="If set, skip saving Step 1 NetCDF output files."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, reprocess and overwrite existing output files."
    )
    parser.add_argument(
        "--no_spatial_test",
        action="store_true",
        default=False,
        help="If set, skip the Step 1 spatial alignment test."
    )
    parser.add_argument(
        "--grid_region",
        type=str,
        default='conus',
        help="Reference grid region: 'conus', 'global', or 'custom'. Default: 'conus'."
    )
    parser.add_argument(
        "--grid_resolution",
        type=int,
        default=375,
        help="Reference grid cell size in meters. Default: 375."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Swaths to accumulate before flushing to the Zarr store. Default: 50."
    )
    parser.add_argument(
        "--grid_pad",
        type=int,
        default=10,
        help="Extra grid cells of padding around the bounding box. Default: 10."
    )
    parser.add_argument(
        "--remove_bowtie",
        action="store_true",
        default=False,
        help="If set, remove bowtie-affected pixels before aggregation."
    )
    parser.add_argument(
        "--deduplicate_scans",
        action="store_true",
        default=False,
        help="If set, deduplicate overlapping scan lines before aggregation."
    )
    parser.add_argument(
        "--copy_to_s3",
        action="store_true",
        default=False,
        help="If set, copy outputs to S3 after processing."
    )
    parser.add_argument(
        "--s3_prefix",
        type=str,
        default=None,
        help="S3 destination prefix. Required if --copy_to_s3 is set. E.g. 's3://my-bucket/outputs/'"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default='VIIRS-cubed-outputs',
        help="Local base directory for all outputs. Default: 'VIIRS-cubed-outputs'."
    )
    parser.add_argument(
        "--remove_local",
        action="store_true",
        default=False,
        help="If set, remove local output files after a successful S3 upload. Requires --copy_to_s3."
    )
    parser.add_argument(
        "--add_persistence",
        action="store_true",
        default=False,
        help="If set, compute fire persistence metrics after gridding and write them to the Zarr store."
    )
    parser.add_argument(
        "--persistence_fire_mask_col",
        type=str,
        default=None,
        help="Fire mask column for persistence. If not set, runs both 'fire_mask_max' and 'fire_mask_area_weighted_majority'."
    )
    parser.add_argument(
        "--persistence_suffix",
        type=str,
        default=None,
        help="Suffix for persistence output variables. Required when --persistence_fire_mask_col is set."
    )
    parser.add_argument(
        "--persistence_start_threshold",
        type=int,
        default=6,
        help="Fire mask threshold for ignition detection. Default: 6."
    )
    parser.add_argument(
        "--persistence_end_threshold",
        type=int,
        default=6,
        help="Fire mask threshold for sustained detection. Default: 6."
    )
    parser.add_argument(
        "--save_workflow_log",
        action="store_true",
        default=True,
        help="If set, write a single log spanning both pipeline steps to the fire's Logs/ directory."
    )

    args = parser.parse_args()

    # ===================================================================
    # VALIDATE ARGUMENT COMBINATIONS
    # ===================================================================

    if args.copy_to_s3 and args.s3_prefix is None:
        parser.error("--s3_prefix is required when --copy_to_s3 is set.")
    if args.remove_local and not args.copy_to_s3:
        parser.error("--remove_local requires --copy_to_s3.")
    if args.persistence_fire_mask_col is not None and args.persistence_suffix is None:
        parser.error("--persistence_suffix is required when --persistence_fire_mask_col is set.")

    # ===================================================================
    # CALL process_single_fire
    # ===================================================================

    bbox = ast.literal_eval(args.bbox)
    sensors = ast.literal_eval(args.sensors)

    process_single_fire(
        fire_name=args.fire_name,
        start=args.start,
        end=args.end,
        bbox=bbox,
        n_timesteps=args.n_timesteps,
        pix_lut_path=args.pix_lut_path,
        sensors=sensors,
        make_plots=args.make_plots,
        save_data=not args.no_save_data,
        overwrite=args.overwrite,
        run_spatial_test=not args.no_spatial_test,
        grid_region=args.grid_region,
        grid_resolution=args.grid_resolution,
        batch_size=args.batch_size,
        grid_pad=args.grid_pad,
        remove_bowtie=args.remove_bowtie,
        deduplicate_scans=args.deduplicate_scans,
        copy_to_s3=args.copy_to_s3,
        s3_prefix=args.s3_prefix,
        output_dir=args.output_dir,
        remove_local=args.remove_local,
        add_persistence=args.add_persistence,
        persistence_fire_mask_col=args.persistence_fire_mask_col,
        persistence_suffix=args.persistence_suffix,
        persistence_start_threshold=args.persistence_start_threshold,
        persistence_end_threshold=args.persistence_end_threshold,
        save_workflow_log=args.save_workflow_log,
    )

    # ===================================================================
    # USAGE EXAMPLE
    # ===================================================================
    #
    # python workflows.py \
    #     --fire_name 'Dragon_Bravo_TEST' \
    #     --start '2025-07-01' \
    #     --end '2025-07-10' \
    #     --bbox '[-112.309113, 36.112467, -111.800995, 36.748712]' \
    #     --n_timesteps -1 \
    #     --sensors '["SNPP", "NOAA20", "NOAA21"]' \
    #     --grid_region 'conus' \
    #     --grid_resolution 375 \
    #     --batch_size 50 \
    #     --grid_pad 10 \
    #     --add_persistence \
    #     --copy_to_s3 \
    #     --s3_prefix 's3://maap-ops-workspace/shared/gsfc_landslides/FireSense/' \
    #     --output_dir 'VIIRS-cubed-outputs' \
    #     --overwrite \
    #     --save_workflow_log
