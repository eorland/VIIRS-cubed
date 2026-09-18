import datetime as dt
import numpy as np
import pandas as pd
import xarray as xr
import subprocess
import tempfile
import os
import s3fs
import netCDF4 as nc4


def log_message(message, log_file, print_to_console=True, include_timestamp=True):
    if include_timestamp:
        timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
    else:
        formatted_message = message
    log_file.write(formatted_message + '\n')
    log_file.flush()
    if print_to_console:
        print(formatted_message)


def upload_log_to_s3(log_path: str, output_dir: str, fire_name: str) -> None:
    """
    Upload a local log file to the fire's Logs/ directory on S3.

    Parameters
    ----------
    log_path : str
        Local path to the log file.
    output_dir : str
        Base output directory — must be an S3 URI (s3://bucket/prefix/).
    fire_name : str
        Fire/region name, used to construct the S3 destination path.
    """
    fs = s3fs.S3FileSystem()
    s3_log_dest = (
        f"{output_dir.rstrip('/')}/{fire_name}_Gridded_VIIRS"
        f"/Logs/{os.path.basename(log_path)}"
    )
    fs.put(log_path, s3_log_dest)


# ===================================================================
# S3 / LOCAL PATH ROUTING HELPERS
# ===================================================================

def is_s3_path(path: str) -> bool:
    """Return True if path is an S3 URI (starts with s3://)."""
    return path.startswith('s3://')


def file_exists(path: str) -> bool:
    """
    Check whether a file exists at a local or S3 path.

    Parameters
    ----------
    path : str
        Local filesystem path or S3 URI (s3://bucket/key).

    Returns
    -------
    bool
    """
    if is_s3_path(path):
        fs = s3fs.S3FileSystem()
        return fs.exists(path)
    return os.path.exists(path)


def save_dataset(ds: xr.Dataset, output_path: str) -> None:
    """
    Save an xarray Dataset to a local or S3 path.

    For S3 paths, the dataset is written to a local temp file first,
    then uploaded and the temp file is deleted.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset to save.
    output_path : str
        Destination path — local filesystem path or S3 URI.
    """
    if is_s3_path(output_path):
        fs = s3fs.S3FileSystem()
        # Write to a temp file, upload, then clean up
        with tempfile.NamedTemporaryFile(suffix='.nc', delete=False) as tmp:
            tmp_path = tmp.name
        try:
            ds.to_netcdf(tmp_path)
            fs.put(tmp_path, output_path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        ds.to_netcdf(output_path)


def list_swath_files(data_dir: str) -> list:
    """
    List all *_swath.nc files in a local or S3 directory.

    Returns full paths (local) or S3 URIs (s3://...) sorted alphabetically.

    Parameters
    ----------
    data_dir : str
        Local directory path or S3 URI prefix.

    Returns
    -------
    list of str
    """
    if is_s3_path(data_dir):
        fs = s3fs.S3FileSystem()
        # Strip s3:// prefix for fs.glob, then restore it on results
        prefix = data_dir[5:]  # everything after 's3://'
        matches = fs.glob(f"{prefix}/*_swath.nc")
        return sorted([f's3://{p}' for p in matches])
    return sorted([
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.endswith('_swath.nc')
    ])


# def open_netcdf_metadata(path: str):
#     """
#     Open a NetCDF file for metadata reads only (attributes + dimension
#     sizes, no data arrays), from either a local or S3 path.

#     NetCDF metadata reads are sequential rather than random-access, so
#     streaming via fsspec works reliably here without loading the full
#     file into memory.

#     Usage:
#         with open_netcdf_metadata(path) as ds:
#             attrs = ds.ncattrs()

#     Parameters
#     ----------
#     path : str
#         Local filesystem path or S3 URI.

#     Returns
#     -------
#     nc4.Dataset
#     """
#     if is_s3_path(path):
#         fs = s3fs.S3FileSystem()
#         return nc4.Dataset(fs.open(path), 'r')
#     return nc4.Dataset(path, 'r')

def compute_fire_persistence_baseline(
    all_data,
    persistence_threshold_col,
    suffix,
    start_value=0,
    end_value=0,
    area_fraction_col='candidate_area_fraction',
    area_fraction_threshold=0.5,
):
    """
    Compute fire persistence and detection metrics per pixel.

    Definitions
    -----------
    t_start : first time persistence_threshold_col >= start_value.
    t_end   : last time persistence_threshold_col >= end_value.
    Once a pixel ignites (reaches t_start), all subsequent metrics use
    the end threshold and are restricted to the window [t_start, t_end].

    Metrics
    -------
    persistence_hours : (t_end - t_start) in hours, floored at 12,
        NaN where the pixel never ignites.
    n_day / n_night : count of in-window timesteps with persistence_threshold_col >= end
        threshold, split by day/night overpass period. t_start is included.
    n_total : n_day + n_night (total in-window detections).
    dp_ratio : n_total / (number of valid, non-NaN in-window observations).
        Bounded (0, 1] for any ignited pixel; NaN where no ignition.
    n_cloud_detection_windows : count of in-window timesteps with original fire_mask == 4 (cloud)

    Parameters
    ----------
    all_data : xr.Dataset
        Input datacube with 'time', 'daynight', and persistence_threshold_col.
    persistence_threshold_col : str
        Name of the candidate confidence variable
        ('candidate_confidence_max' or
        'candidate_confidence_area_weighted_majority').
    suffix : str
        Suffix appended to all output variable names (e.g. 'aw', 'max').
    start_value : int
        persistence_threshold_col threshold for ignition (>=). Default 0 (candidate).
    end_value : int
        persistence_threshold_col threshold for continued detection (>=). Default 0 (candidate).
    area_fraction_col : str, optional
        Variable in all_data giving the fraction of contributing pixel area
        occupied by candidates at each (time, y, x). Default
        'candidate_area_fraction'.
    area_fraction_threshold : float, optional
        Minimum area fraction required for a timestep to count as a
        detection. Default 0.5 (candidates must cover >= 50% of the cell's
        contributing area).

    Returns
    -------
    all_data : xr.Dataset
        Copy of input with new variables added:
          persistence_hours_baseline_{suffix}, t_fire_start_{suffix},
          t_fire_end_baseline_{suffix}, n_day_detection_windows_{suffix},
          n_night_detection_windows_{suffix}, n_total_detection_windows_{suffix},
          dp_ratio_{suffix}, n_cloud_detection_windows_{suffix}
    """

    assert start_value >= end_value, (
        f"start_value ({start_value}) must be >= end_value ({end_value})"
    )

    # filter to just candidate confidence or higher
    cm = all_data[persistence_threshold_col].where(all_data[persistence_threshold_col] >= 0)
    # similar filtering for candidate area fractions
    area_threshold = all_data[area_fraction_col] >= area_fraction_threshold

    # define both start and end conditions. these are boolean masks of shape (time, y, x)
    start_condition = (cm >= start_value) & area_threshold
    end_condition = (cm >= end_value) & area_threshold
    
    # time is 1D, so we need to convert to 3D to match cm shape
    time_bcast = all_data['time'].broadcast_like(cm)

    # find the first and last times the start and end conditions are met, respectively
    t_start_ns = time_bcast.where(start_condition).min('time', skipna=True)
    t_end_ns = time_bcast.where(end_condition).max('time', skipna=True)

    has_start = t_start_ns.notnull() # filter to pixels with valid start time

    # convert ns to hrs
    persistence_hours = (t_end_ns - t_start_ns) / 3.6e12 
    # set min of 12 hours for any pixel that ignites, and set to NaN for pixels that never ignite
    persistence_hours = persistence_hours.where(persistence_hours >= 12, 12).where(has_start)

    # define each pixel's "burning" window
    in_window = (time_bcast >= t_start_ns) & (time_bcast <= t_end_ns)

    # pull out the overpass period and determine which timesteps are day vs night
    daynight = xr.DataArray(
        np.array([str(p).strip() for p in all_data['daynight'].values]),
        dims='time',
    )
    is_day = daynight == 'Day'

    # how many pixels are above the fire mask threshold and within the defined window of burning?
    # NOTE: end_condition is used here for simplicity and because it is the more relevant threshold for continued detection.
    #       this also assumes that start threshold >= end threshold, 
    #       which the assert statement covers
    detected_in_window = end_condition & in_window

    # break out by overpass period category
    # .sum() method works here, because True/False values are treated as 1/0, respectively
    n_day = (detected_in_window & is_day).sum('time').astype('float32').where(has_start)
    n_night = (detected_in_window & ~is_day).sum('time').astype('float32').where(has_start)
    n_total = (n_day + n_night)

    # BONUS: return count of 'cloud' pixels - these could also be smoke
    fire_mask_col = persistence_threshold_col.replace('candidate_confidence', 'fire_mask')
    if fire_mask_col not in all_data:
        raise KeyError(
            f"'{fire_mask_col}' is not found."
            f"Available variables: {list(all_data.data_vars)}"
        )
    n_cloud = ((all_data[fire_mask_col] == 4) & in_window).sum('time')

    # final step - compute the ratio between final detection counts and the number of valid observations in the defined window.
    valid_observation = all_data[fire_mask_col] >= 0
    n_valid = (valid_observation & in_window).sum('time')
    dp_ratio = (n_total / n_valid).astype('float32').where(has_start) # this can't be greater than 1



    # --- Store results ---
    out = {
        f'persistence_hours_baseline_{suffix}': persistence_hours,
        f't_fire_start_{suffix}': t_start_ns.astype('datetime64[ns]'),
        f't_fire_end_baseline_{suffix}': t_end_ns.astype('datetime64[ns]'),
        f'n_day_detection_windows_{suffix}': n_day,
        f'n_night_detection_windows_{suffix}': n_night,
        f'n_total_detection_windows_{suffix}': n_total,
        f'dp_ratio_{suffix}': dp_ratio,
        f'n_cloud_detection_windows_{suffix}': n_cloud,
    }

    all_data = all_data.drop_vars(list(out.keys()), errors='ignore')
    for name, da in out.items():
        all_data[name] = da

    return all_data


SWATH_METADATA_FILENAME = '_swath_metadata.csv'

_SWATH_METADATA_COLUMNS = [
    'filepath', 'filename', 'satellite', 'timestamp',
    'date', 'time', 'avg_scan_angle', 'daynight',
    'overpass_period', 'n_scans', 'n_pixels',
    'original_scans', 'original_pixels', 'scan_min', 'scan_max',
]


def _swath_metadata_path(data_dir: str) -> str:
    """Return the swath metadata CSV path for a Step 1 data directory."""
    if is_s3_path(data_dir):
        return f"{data_dir.rstrip('/')}/{SWATH_METADATA_FILENAME}"
    return os.path.join(data_dir, SWATH_METADATA_FILENAME)


def load_swath_metadata(data_dir: str) -> pd.DataFrame:
    """
    Load the swath metadata table for a Step 1 data directory.

    Deduplicates on 'filepath' keeping the last entry — handles the
    case where overwrite=True caused a filepath to be appended twice.
    Returns an empty DataFrame with the correct schema if not found.

    Parameters
    ----------
    data_dir : str
        Local path or S3 URI of the Step 1 data directory.

    Returns
    -------
    pd.DataFrame
    """
    path = _swath_metadata_path(data_dir)
    if not file_exists(path):
        return pd.DataFrame(columns=_SWATH_METADATA_COLUMNS)
    try:
        if is_s3_path(path):
            fs = s3fs.S3FileSystem()
            with fs.open(path, 'r') as f:
                df = pd.read_csv(f, parse_dates=['timestamp'])
        else:
            df = pd.read_csv(path, parse_dates=['timestamp'])
        # Deduplicate: if overwrite=True was used, a filepath may appear
        # more than once — keep the most recently written row
        df = (df.drop_duplicates(subset='filepath', keep='last')
                .sort_values('timestamp')
                .reset_index(drop=True))
        return df
    except Exception as e:
        print(f"Warning: could not load swath metadata at {path}: {e}. "
              f"Starting fresh.")
        return pd.DataFrame(columns=_SWATH_METADATA_COLUMNS)


def append_swath_metadata_row(row: dict, data_dir: str) -> None:
    """
    Append a single metadata row to the swath metadata CSV for a Step 1
    data directory. Creates the file with a header if it does not yet
    exist; otherwise appends without repeating the header.

    For S3 destinations, the existing CSV is downloaded, the row is
    appended locally, and the updated file is re-uploaded. This keeps
    the file in a single, human-readable location on S3 rather than
    scattering per-run fragments.

    Parameters
    ----------
    row : dict
        Single row of metadata. Keys must match _SWATH_METADATA_COLUMNS.
    data_dir : str
        Local path or S3 URI of the Step 1 data directory.
    """
    path = _swath_metadata_path(data_dir)
    row_df = pd.DataFrame([row])

    if is_s3_path(path):
        fs = s3fs.S3FileSystem()
        with tempfile.NamedTemporaryFile(
            suffix='.csv', mode='w', delete=False
        ) as tmp:
            tmp_path = tmp.name
        try:
            # Download existing file if present, append row, re-upload
            if fs.exists(path):
                with fs.open(path, 'r') as f:
                    existing = pd.read_csv(f, parse_dates=['timestamp'])
                updated = pd.concat([existing, row_df], ignore_index=True)
            else:
                updated = row_df
            updated.to_csv(tmp_path, index=False)
            fs.put(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    else:
        # Pure append — no load/rewrite needed for local files
        write_header = not os.path.exists(path)
        row_df.to_csv(path, mode='a', header=write_header, index=False)