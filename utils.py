import datetime as dt

import numpy as np
import xarray as xr


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


def compute_fire_persistence_baseline(
    all_data,
    fire_mask_col,
    suffix,
    start_fire_mask_value=6,
    end_fire_mask_value=6,
):
    """
    Compute fire persistence and detection metrics per pixel.

    Definitions
    -----------
    t_start : first time fire_mask >= start_fire_mask_value.
    t_end   : last time fire_mask >= end_fire_mask_value.
    Once a pixel ignites (reaches t_start), all subsequent metrics use
    the end threshold and are restricted to the window [t_start, t_end].

    Metrics
    -------
    persistence_hours : (t_end - t_start) in hours, floored at 12,
        NaN where the pixel never ignites.
    n_am / n_pm : count of in-window timesteps with fire_mask >= end
        threshold, split by AM/PM overpass period. t_start is included.
    n_total : n_am + n_pm (total in-window detections).
    dp_ratio : n_total / (number of valid, non-NaN in-window observations).
        Bounded (0, 1] for any ignited pixel; NaN where no ignition.
    n_cloud_detection_windows : count of in-window timesteps with fire_mask == 4 (cloud)

    Parameters
    ----------
    all_data : xr.Dataset
        Input datacube with 'time', 'overpass_period', and fire_mask_col.
    fire_mask_col : str
        Name of the fire mask variable ('fire_mask_max' or
        'fire_mask_area_weighted_majority').
    suffix : str
        Suffix appended to all output variable names (e.g. 'aw', 'max').
    start_fire_mask_value : int
        Fire mask threshold for ignition (>=). Default 6.
    end_fire_mask_value : int
        Fire mask threshold for continued detection (>=). Default 6.

    Returns
    -------
    all_data : xr.Dataset
        Copy of input with new variables added:
          persistence_hours_baseline_{suffix}, t_fire_start_{suffix},
          t_fire_end_baseline_{suffix}, n_am_detection_windows_{suffix},
          n_pm_detection_windows_{suffix}, n_total_detection_windows_{suffix},
          dp_ratio_{suffix}, n_cloud_detection_windows_{suffix}
    """

    # set all -1 values in to NaN for proper handling of min/max and counts.
    fm = all_data[fire_mask_col].where(all_data[fire_mask_col] >= 0)

    # define both start and end conditions. these are boolean masks of shape (time, y, x)
    start_condition = fm >= start_fire_mask_value
    end_condition = fm >= end_fire_mask_value

     # time is 1D, so we need to convert to 3D to match fm shape
    time_bcast = all_data['time'].broadcast_like(fm)

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

    # pull out the overpass period and determine which timesteps are AM vs PM
    period = xr.DataArray(
        np.array([str(p).strip() for p in all_data['overpass_period'].values]),
        dims='time',
    )
    is_am = period == 'AM'

    # how many pixels are above the fire mask threshold and within the defined window of burning?
    # NOTE: end_condition is used here for simplicity and because it is the more relevant threshold for continued detection.
    #       this also assumes that start threshold >= end threshold, 
    #       which is true for the default values.
    detected_in_window = end_condition & in_window

    # break out by overpass period category
    # .sum() method works here, because True/False values are treated as 1/0, respectively
    n_am = (detected_in_window & is_am).sum('time').astype('float32').where(has_start)
    n_pm = (detected_in_window & ~is_am).sum('time').astype('float32').where(has_start)
    n_total = (n_am + n_pm)

    # final step - compute the ratio between final detection counts and the number of valid observations in the defined window.
    n_valid = (fm.notnull() & in_window).sum('time')
    dp_ratio = (n_total / n_valid).astype('float32').where(has_start) # this can't be greater than 1

    # BONUS: return count of 'cloud' pixels - these could also be smoke
    n_cloud = ((fm == 4) & in_window).sum('time')

    # --- Store results ---
    out = {
        f'persistence_hours_baseline_{suffix}': persistence_hours,
        f't_fire_start_{suffix}': t_start_ns.astype('datetime64[ns]'),
        f't_fire_end_baseline_{suffix}': t_end_ns.astype('datetime64[ns]'),
        f'n_am_detection_windows_{suffix}': n_am,
        f'n_pm_detection_windows_{suffix}': n_pm,
        f'n_total_detection_windows_{suffix}': n_total,
        f'dp_ratio_{suffix}': dp_ratio,
        f'n_cloud_detection_windows_{suffix}': n_cloud,
    }

    all_data = all_data.drop_vars(list(out.keys()), errors='ignore')
    for name, da in out.items():
        all_data[name] = da

    return all_data