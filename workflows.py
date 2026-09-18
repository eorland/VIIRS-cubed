import ast
import datetime as dt
import os

from swath_preprocessing import process_swaths
from standardizations import standardize_swaths
from utils import log_message, upload_log_to_s3, is_s3_path


def process_single_fire(
    # --- Shared required parameters ---
    fire_name,
    start,
    end,
    bbox,
    n_timesteps=-1,

    # --- Step 1 parameters ---
    pix_lut_path=None,
    sensors=['SNPP', 'NOAA20', 'NOAA21'],
    make_plots=False,
    save_data=True,
    overwrite=False,
    run_spatial_test=True,

    # --- Step 2 parameters ---
    grid_region='conus',
    grid_resolution=375,
    batch_size=50,
    grid_pad=10,
    remove_bowtie=False,
    deduplicate_scans=False,

    # --- Shared output/infra parameters ---
    output_dir='VIIRS-cubed-outputs',

    # --- Persistence parameters ---
    add_persistence=False,
    persistence_threshold_col=None,
    persistence_suffix=None,
    persistence_start_threshold=0,
    persistence_end_threshold=0,
    area_fraction_col='candidate_area_fraction',
    area_fraction_threshold=0.5,

    # --- Workflow-level logging ---
    save_workflow_log=True,
):
    '''
    Preprocess and standardize VIIRS swath data for a single fire/region.

    Runs Step 1 (swath preprocessing) followed by Step 2 (gridding into a
    Zarr datacube). Outputs are written to output_dir, which may be a local
    path or an S3 URI (s3://bucket/prefix/). Both steps route transparently
    based on the output_dir prefix.

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
    output_dir : str, optional
        Base output directory. Accepts a local path or an S3 URI
        (s3://bucket/prefix/). Default: 'VIIRS-cubed-outputs'.
    add_persistence : bool, optional
        Compute fire persistence metrics after gridding. Default: False.
    persistence_threshold_col : str or None, optional
        Column for persistence classifications. If None, runs both
        'candidate_confidence_max' and 'candidate_confidence_area_weighted_majority'.
    persistence_suffix : str or None, optional
        Output variable suffix. Required when persistence_threshold_col is set.
    persistence_start_threshold : int, optional
        Fire mask value threshold for ignition detection. Default: 0.
    persistence_end_threshold : int, optional
        Fire mask value threshold for sustained detection. Default: 0.
    area_fraction_col : str, optional
        Variable in all_data giving the fraction of contributing pixel area
        occupied by candidates at each (time, y, x). Default
        'candidate_area_fraction'.
    area_fraction_threshold : float, optional
        Minimum area fraction required for a timestep to count as a
        detection. Default 0.5 (candidates must cover >= 50% of the cell's
        contributing area).
    save_workflow_log : bool, optional
        Write a single log spanning both steps to the fire's Logs/ directory.
        Default: True.
    '''

    if persistence_threshold_col is not None and persistence_suffix is None:
        raise ValueError(
            "persistence_suffix is required when persistence_threshold_col is set"
        )

    # ===================================================================
    # WORKFLOW LOG SETUP
    # ===================================================================

    wf_log = None
    workflow_log_path = None

    if save_workflow_log:
        logs_dir = os.path.join(
            os.path.abspath('.'), f"{fire_name}_Gridded_VIIRS", "Logs"
        )
        os.makedirs(logs_dir, exist_ok=True)
        run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
        workflow_log_path = os.path.join(
            logs_dir, f"{fire_name}_workflow_log_{run_timestamp}.txt"
        )
        wf_log = open(workflow_log_path, 'a')
        log_message("=" * 70, wf_log, include_timestamp=False)
        log_message("VIIRS WORKFLOW LOG",wf_log, include_timestamp=False)
        log_message(
            f"Run started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            wf_log, include_timestamp=False
        )
        log_message("=" * 70, wf_log, include_timestamp=False)
        log_message(f"Fire name: {fire_name}", wf_log)
        log_message(f"Date range: {start} to {end}", wf_log)
        log_message(f"BBOX: {bbox}", wf_log)
        log_message(f"Output dir: {output_dir}", wf_log)
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
            output_dir=output_dir,
            log_file=wf_log,
        )

        # ===================================================================
        # STEP 2: STANDARDIZATION
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
            batch_size=batch_size,
            grid_pad=grid_pad,
            remove_bowtie=remove_bowtie,
            deduplicate_scans=deduplicate_scans,
            output_dir=output_dir,
            add_persistence=add_persistence,
            persistence_threshold_col=persistence_threshold_col,
            persistence_suffix=persistence_suffix,
            persistence_start_threshold=persistence_start_threshold,
            persistence_end_threshold=persistence_end_threshold,
            area_fraction_col=area_fraction_col,
            area_fraction_threshold=area_fraction_threshold,
            log_file=wf_log,
        )

    finally:
        if wf_log is not None:
            log_message("", wf_log, include_timestamp=False)
            log_message(
                f"Workflow completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                wf_log, include_timestamp=False
            )
            wf_log.close()
            wf_log = None
            if is_s3_path(output_dir):
                upload_log_to_s3(workflow_log_path, output_dir, fire_name)
            print(f"\n{'='*60}")
            print(f"Workflow log saved: {workflow_log_path}")
            print(f"{'='*60}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Full VIIRS pipeline: preprocess swaths and grid into a Zarr datacube."
    )

    # Required
    parser.add_argument("--fire_name", type=str, required=True)
    parser.add_argument("--start", type=str, required=True)
    parser.add_argument("--end", type=str, required=True)
    parser.add_argument("--bbox", type=str, required=True,
                        help="'[xmin, ymin, xmax, ymax]'")

    # Optional
    parser.add_argument("--n_timesteps", type=int,  default=-1)
    parser.add_argument("--pix_lut_path", type=str,  default=None)
    parser.add_argument("--sensors", type=str,
                        default='["SNPP", "NOAA20", "NOAA21"]')
    parser.add_argument("--make_plots", action="store_true", default=False)
    parser.add_argument("--no_save_data", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--no_spatial_test", action="store_true", default=False)
    parser.add_argument("--grid_region", type=str, default='conus')
    parser.add_argument("--grid_resolution", type=int, default=375)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--grid_pad", type=int, default=10)
    parser.add_argument("--remove_bowtie", action="store_true", default=False)
    parser.add_argument("--deduplicate_scans",action="store_true", default=False)
    parser.add_argument("--output_dir", type=str,
                        default='VIIRS-cubed-outputs',
                        help=("Local path or S3 URI (s3://bucket/prefix/). "
                              "Both steps route transparently based on prefix."))
    parser.add_argument("--add_persistence", action="store_true", default=False)
    parser.add_argument("--persistence_threshold_col", type=str, default=None)
    parser.add_argument("--persistence_suffix", type=str, default=None)
    parser.add_argument("--persistence_start_threshold", type=int, default=0)
    parser.add_argument("--persistence_end_threshold", type=int, default=0)
    parser.add_argument("--area_fraction_col", type=str, default='candidate_area_fraction',
                        help="Zarr variable used as the area gate for persistence detection.")
    parser.add_argument("--area_fraction_threshold", type=float, default=0.5,
                        help="Minimum area fraction for a timestep to count as a detection (default: 0.5).")
    parser.add_argument("--no_save_workflow_log", action="store_true", default=False)

    args = parser.parse_args()

    if args.persistence_threshold_col is not None and args.persistence_suffix is None:
        parser.error(
            "--persistence_suffix is required when --persistence_threshold_col is set."
        )

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
        output_dir=args.output_dir,
        add_persistence=args.add_persistence,
        persistence_threshold_col=args.persistence_threshold_col,
        persistence_suffix=args.persistence_suffix,
        persistence_start_threshold=args.persistence_start_threshold,
        persistence_end_threshold=args.persistence_end_threshold,
        area_fraction_col=args.area_fraction_col,
        area_fraction_threshold=args.area_fraction_threshold,
        save_workflow_log=not args.no_save_workflow_log,
    )

    # ===================================================================
    # USAGE EXAMPLE
    # ===================================================================
    #
    # Local output:
    # python workflows.py \
    #     --fire_name 'Dragon_Bravo_TEST' \
    #     --start '2025-07-01' --end '2025-07-10' \
    #     --bbox '[-112.309113, 36.112467, -111.800995, 36.748712]' \
    #     --output_dir 'VIIRS-cubed-outputs' --overwrite
    #
    # S3 output with persistence:
    # python workflows.py \
    #     --fire_name 'Dragon_Bravo_TEST' \
    #     --start '2025-07-01' --end '2025-07-10' \
    #     --bbox '[-112.309113, 36.112467, -111.800995, 36.748712]' \
    #     --output_dir 's3://maap-ops-workspace/shared/gsfc_landslides/FireSense/' \
    #     --overwrite --add_persistence \
    #     --area_fraction_col 'candidate_area_fraction' \
    #     --area_fraction_threshold 0.5
    # ===================================================================

