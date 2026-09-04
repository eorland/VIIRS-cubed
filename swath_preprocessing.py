# Core data processing
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray as rio
import rasterio
import datetime as dt
import os
from tqdm import tqdm
import fsspec
import io
import s3fs
import subprocess
import shutil

# Satellite data access
import earthaccess

# Optional plotting and visualization
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner


def gather_filepaths(bbox, start, end, sensors):
    '''
    Compiles all available VIIRS datasets for a chosen set of sensors
    based on user-specified spatiotemporal properties. 

    Returns:
    Dictionary of all available files organized by timestep of swath's overpass.
    
    '''

    # start with earthdata login
    earthaccess.login(strategy='netrc')
    assert earthaccess.__auth__.authenticated, "Earthdata login failed! Check your .netrc file."

    all_products = {'SNPP': ['VNP03IMG','VNP02IMG','VNP14IMG'], # order: L1 geolocation, L1 science, l2 fire mask
                'NOAA20': ['VJ103IMG','VJ102IMG','VJ114IMG'],
                'NOAA21': ['VJ203IMG','VJ202IMG','VJ214IMG']}

    selected_products = {s: all_products[s] for s in sensors}
    
    #################################
    ##### COLLECT ALL FILEPATHS #####
    #################################
    
    files = {}
    for sat in list(selected_products.keys()):
        print(f"Querying {sat} data...")
        
        # science data 02IMG
        results = earthaccess.search_data(
            short_name=selected_products[sat][1],
            bounding_box=(bbox[0],bbox[1],bbox[2],bbox[3]),
            temporal=(start, end),
            count=-1) # -1 returns all matches!
        files[selected_products[sat][1]] = earthaccess.open(results) # add files under the keyword referenced in the dict 'selected_products'
        
        # geolocation 03IMG
        results = earthaccess.search_data(
            short_name=selected_products[sat][0],
            bounding_box=(bbox[0],bbox[1],bbox[2],bbox[3]),
            temporal=(start, end),
            count=-1)
        files[selected_products[sat][0]] = earthaccess.open(results)
    
        # Level 2 data
        results = earthaccess.search_data(
            short_name=selected_products[sat][2],
            bounding_box=(bbox[0],bbox[1],bbox[2],bbox[3]),
            temporal=(start, end),
            count=-1)  
        
        earthaccess.login(strategy='netrc')
        files[selected_products[sat][2]] = earthaccess.open(results)

    ###########################################
    ##### ORGANIZE FILES AFTER COLLECTION #####
    ###########################################

    # For each swath timestep, collect all relevent products for that overpass
    print("Collecting all timesteps from all satellites...")
    all_timesteps = []
    
    for sat in list(selected_products.keys()): # loop through each sat
        print(f"Collecting timesteps from {sat}...")
        
        # Get all files for this satellite
        geo_files = files[selected_products[sat][0]]
        science_files = files[selected_products[sat][1]]
        fire_files = files[selected_products[sat][2]]
        
        # Create dictionaries mapping timestamps to files
        def get_timestamp_from_file(file):
            """Extract timestamp string from file path"""
            if isinstance(file, str):  # Local file path
                timestamp = '.'.join(file.split('.')[-5:-3])
            else:  # earthaccess file object
                timestamp = '.'.join(file.path.split('.')[-5:-3])
            return timestamp
        
        # Build lookup dictionaries - each key is a timestamp! Will be used below.
        science_dict = {get_timestamp_from_file(f): f for f in science_files}
        fire_dict = {get_timestamp_from_file(f): f for f in fire_files}
        
        # Iterate through geo files and match by timestamp
        for geo_file in geo_files:
            # Parse timestamp
            timestamp_str = get_timestamp_from_file(geo_file)
            timestamp = timestamp_str.split('.')
            year = timestamp[0][1:5]
            day = timestamp[0][5:8] 
            time = timestamp[1]
            acq_datetime = dt.datetime.strptime(year+day+time[:2]+time[2:], '%Y%j%H%M')
            timestamp_pd = pd.Timestamp(acq_datetime)
            
            # Find matching files by timestamp
            science_file = science_dict.get(timestamp_str)
            fire_file = fire_dict.get(timestamp_str)
            
            # Only include if we have all three files
            if science_file is None or fire_file is None:
                # Format as MMDDYYYY HHMMSS
                readable_time = acq_datetime.strftime('%m-%d-%Y %H:%M:%S')
                print(f"  Skipping {timestamp_str} ({readable_time}) - missing matching files")
                continue
            
            # Store all file info together
            timestep_info = {
                'satellite': sat,
                'timestamp': timestamp_pd,
                'geo_file': geo_file,
                'science_file': science_file,
                'fire_file': fire_file,
                'timestamp_str': timestamp_str
            }
            
            all_timesteps.append(timestep_info)
    
    # Sort all timesteps chronologically
    all_timesteps.sort(key=lambda x: x['timestamp'])
    
    print(f"\nFound {len(all_timesteps)} complete timesteps across all satellites")
    if all_timesteps:
        print(f"Time range: {all_timesteps[0]['timestamp']} to {all_timesteps[-1]['timestamp']}")

    return all_timesteps


# DEFINE MAIN FUNCTION
def combine_viirs_swaths(timestep_info, bbox, pix_lut_path=None, verbose=False, run_tests=False):
    """
    Load and organize VIIRS swath data, rasterize all fire point variables, and crop to bounding box.
    
    Parameters
    ----------
    timestep_info : dict
        Dictionary containing satellite, timestamp, and file handles
    bbox : list
        Bounding box [lon_min, lat_min, lon_max, lat_max]
    pix_lut : string
        Filepath of Lookup table for scan angle and pixel area (indexed by sample number)
    verbose : bool
        Print progress messages
    run_tests : bool
        Optionally run spatial test to ensure no data were altered upon handling
    
    Returns
    -------
    swath_ds : xr.Dataset
        xarray Dataset with all variables as 2D rasters, cropped to bbox
    """

    if pix_lut_path:
        pix_lut = pd.read_csv(pix_lut_path)
    
    else: 
        pix_lut = pd.read_csv('s3://maap-ops-workspace/shared/coffield/pix_size_lut.csv', index_col='sample') 
    
    # Unpack timestep info from dictionary
    sat = timestep_info['satellite']
    timestamp_pd = timestep_info['timestamp']
    geo_filepath = timestep_info['geo_file']
    science_filepath = timestep_info['science_file']
    fire_filepath = timestep_info['fire_file']
    
    # ===================================================================
    # LOAD GEOLOCATION DATA (03IMG)
    # ===================================================================
    if verbose: 
        print("  [1/6] Loading geolocation data...")

    geolocation_ds = xr.open_dataset(geo_filepath, engine='h5netcdf', group='geolocation_data')

    longitude = geolocation_ds['longitude'].values
    latitude = geolocation_ds['latitude'].values
    sza = geolocation_ds['solar_zenith'].values
    
    n_lines, n_pixels = longitude.shape
    
    _, j = np.indices(longitude.shape)
    
    scan_angle = pix_lut.loc[j.flatten(), 'scan_angle'].values.reshape(j.shape)
    pixel_area = pix_lut.loc[j.flatten(), 'pix_area'].values.reshape(j.shape)
    
    geolocation_ds.close()
    
    # ===================================================================
    # LOAD L1 SCIENCE DATA (02IMG)
    # ===================================================================
    
    if verbose: 
        print("  [2/6] Loading L1 brightness temperature data...")

    science_ds = xr.open_dataset(science_filepath, engine='h5netcdf',
                                  group='observation_data', mask_and_scale=False) # note mask_and_scale == False
    
    # --- Grab I4 (3.75 µm) data ---
    i4_raw = science_ds['I04'].values
    i4_lut = science_ds['I04_brightness_temperature_lut'].values
    i4_bt = i4_lut[i4_raw]
    
    # --- Grab I5 (11 µm) Data ---
    i5_raw = science_ds['I05'].values
    i5_lut = science_ds['I05_brightness_temperature_lut'].values
    i5_bt = i5_lut[i5_raw]

    # Replace -999 fill values with NaN
    i4_n_fill = (i4_bt < 0).sum()
    i5_n_fill = (i5_bt < 0).sum()
    i4_bt[i4_bt < 0] = np.nan
    i5_bt[i5_bt < 0] = np.nan

    # Compute band difference
    delta_i4_i5 = i4_bt - i5_bt
    
    if verbose:
        n_total = i4_bt.size
        print(f"    I4 fill pixels: {i4_n_fill} ({i4_n_fill/n_total*100:.1f}%)")
        print(f"    I5 fill pixels: {i5_n_fill} ({i5_n_fill/n_total*100:.1f}%)")
    
    science_ds.close()
    
    # ===================================================================
    # LOAD L2 FIRE DATA (14IMG)
    # ===================================================================
    
    if verbose: 
        print("  [3/6] Loading L2 fire mask and detection data...")
    
    fire_ds = xr.open_dataset(fire_filepath, phony_dims='sort')
    
    # --- Pull swath categorical data (2D rasters) ---
    fire_mask = fire_ds['fire mask'].values
    algorithm_qa = fire_ds['algorithm QA'].values
    
    daynight = str(fire_ds.DayNightFlag)

    # ===================================================================
    # DECODE ALGORITHM QA FLAGS
    # ===================================================================
    
    if verbose:
        print("  [3b/6] Decoding algorithm QA flags...")
    
    # Per-band radiometric quality (0 = nominal, 1 = non-nominal)
    qa_I4_quality = ((algorithm_qa >> 3) & 1).astype(np.uint8)
    qa_I5_quality = ((algorithm_qa >> 4) & 1).astype(np.uint8)
    
    # Detection algorithm flags used for candidate identification
    qa_background_pixel = ((algorithm_qa >> 8) & 1).astype(np.uint8)
    qa_candidate_pixel = ((algorithm_qa >> 10) & 1).astype(np.uint8)
    
    # Saturation/folding flag
    # Set when: BT5 >= 325K OR BT4 = 367K OR delta_BT45 < 0
    qa_saturation = ((algorithm_qa >> 16) & 1).astype(np.uint8)

    # Bowtie flag - helpful for filtering later
    qa_bowtie = ((algorithm_qa >> 22) & 1).astype(np.uint8)

    if verbose:
        print(f"    I4 non-nominal quality: {qa_I4_quality.sum()} pixels")
        print(f"    I5 non-nominal quality: {qa_I5_quality.sum()} pixels")
        print(f"    Background pixels: {qa_background_pixel.sum()}")
        print(f"    Candidate pixels: {qa_candidate_pixel.sum()}")
        print(f"    Saturation/folding: {qa_saturation.sum()}")
        print(f"    Residual bowtie: {qa_bowtie.sum()}")

    # ===================================================================
    # PROCESS 1D DETECTION DATA
    # ===================================================================
    
    if verbose:
        print("  [4/6] Rasterizing fire point detections...")

    # get line and sample vals for all L2 point detections
    fp_line = fire_ds['FP_line'].values
    fp_sample = fire_ds['FP_sample'].values
    
    # Check for duplicate values 
    assert len(np.unique(np.stack([fp_line, fp_sample], axis=1), axis=0)) == len(fp_line), \
    f"Duplicate FP_line/FP_sample pairs in {fire_filepath}"

    fp_rasters = {}
    for var_name in fire_ds.data_vars:
        var_dims = fire_ds[var_name].dims # all point detection data are 1D, so we check the dims here
        if len(var_dims) == 1:
            if var_name in ['FP_line', 'FP_sample']: # skip because we already have / don't need these
                continue
                
            clean_var_name = var_name.replace(' ', '_').replace('-', '_')
            fp_raster = np.full((n_lines, n_pixels), np.nan, dtype=np.float32) # empty array same size as all others
            fp_raster[fp_line, fp_sample] = fire_ds[var_name].values # populate with values in correct locations
            fp_rasters[clean_var_name] = fp_raster # add to dict containing all FP arrays
    
    # ===================================================================
    # CREATE CANDIDATE PIXEL RASTERS
    # ===================================================================
    
    if verbose:
        print("  [5/6] Creating candidate pixel rasters...")
    
    # Use decoded QA flags for candidate identification
    has_qa_flag = (qa_background_pixel == 1) | (qa_candidate_pixel == 1) # note: this is isolating candidates ONLY
    is_confirmed_fire = fire_mask >= 7 # this is isolating already confirmed fires
    
    # NaN = not a candidate, 0 = QA candidate, 1 = low, 2 = nominal, 3 = high
    confidence_array = np.full_like(fire_mask, fill_value=np.nan, dtype=np.float32)
    confidence_array[(fire_mask < 7) & has_qa_flag] = 0 # i.e., not confirmed fire BUT candidate
    confidence_array[fire_mask == 7] = 1
    confidence_array[fire_mask == 8] = 2
    confidence_array[fire_mask == 9] = 3

    if verbose:
        n_total_candidates = np.isfinite(confidence_array).sum()
        n_confirmed = (fire_mask >= 7).sum()
        n_qa_only = ((fire_mask < 7) & has_qa_flag).sum()
        print(f"    Total candidates: {n_total_candidates}")
        print(f"    Confirmed fires: {n_confirmed}")
        print(f"    QA-flagged only: {n_qa_only}")
    
    fire_ds.close()
    
    # ===================================================================
    # BUILD FULL XARRAY DATASET
    # ===================================================================
    
    data_vars = {
        # Geolocation (2D)
        'longitude': (['scan', 'pixel'], longitude),
        'latitude': (['scan', 'pixel'], latitude),
        'sza': (['scan', 'pixel'], sza),
        'scan_angle': (['scan', 'pixel'], scan_angle),
        'pixel_area': (['scan', 'pixel'], pixel_area),
        'sample_index': (['scan', 'pixel'], j),
        
        # L1 Brightness Temperature (2D)
        'I4_bt': (['scan', 'pixel'], i4_bt),
        'I5_bt': (['scan', 'pixel'], i5_bt),
        'delta_I4_I5': (['scan', 'pixel'], delta_i4_i5),
        
        # L2 Fire Mask (2D)
        'fire_mask': (['scan', 'pixel'], fire_mask),
        
        # Algorithm QA — raw integer + decoded flags
        'algorithm_qa': (['scan', 'pixel'], algorithm_qa),
        'qa_I4_quality': (['scan', 'pixel'], qa_I4_quality),
        'qa_I5_quality': (['scan', 'pixel'], qa_I5_quality),
        'qa_background_pixel': (['scan', 'pixel'], qa_background_pixel),
        'qa_candidate_pixel': (['scan', 'pixel'], qa_candidate_pixel),
        'qa_saturation': (['scan', 'pixel'], qa_saturation),
        'qa_bowtie': (['scan', 'pixel'], qa_bowtie),
        
        # Candidate pixel information (2D)
        'candidate_confidence': (['scan', 'pixel'], confidence_array),
    }
    
    # Add rasterized FP variables
    for fp_name, fp_raster in fp_rasters.items():
        data_vars[fp_name] = (['scan', 'pixel'], fp_raster)
    
    swath_ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            'scan': np.arange(n_lines),
            'pixel': np.arange(n_pixels),
        },
    )
    
    # ===================================================================
    # CROP TO BOUNDING BOX
    # ===================================================================
    if verbose: 
        print("  [6/6] Cropping to bounding box...")
    
    bbox_mask = (
        (swath_ds['longitude'] >= bbox[0]) & 
        (swath_ds['longitude'] <= bbox[2]) & 
        (swath_ds['latitude'] >= bbox[1]) & 
        (swath_ds['latitude'] <= bbox[3])
    )
    
    rows_with_data = np.where(bbox_mask.values.any(axis=1))[0]
    cols_with_data = np.where(bbox_mask.values.any(axis=0))[0]
    
    if len(rows_with_data) == 0 or len(cols_with_data) == 0:
        if verbose:
            print("      WARNING: No data intersects with bounding box!")
        return None
    
    row_slice = slice(rows_with_data[0], rows_with_data[-1] + 1)
    col_slice = slice(cols_with_data[0], cols_with_data[-1] + 1)
    
    swath_ds_cropped = swath_ds.isel(scan=row_slice, pixel=col_slice)

    if verbose and 'candidate_confidence' in swath_ds_cropped:
        cand_conf = swath_ds_cropped['candidate_confidence'].values
        n_total = np.isfinite(cand_conf).sum()
        n_confirmed = (cand_conf >= 1).sum()
        n_qa_only = (cand_conf == 0).sum()
        print(f"    Candidates in cropped region:")
        print(f"      Total: {n_total}")
        print(f"      Confirmed fires: {n_confirmed}")
        print(f"      QA-flagged only: {n_qa_only}")
    
    # ===================================================================
    # ADD PER-TIMESTEP METADATA AS DATA VARIABLES
    # ===================================================================
    
    avg_scan_angle = float(swath_ds_cropped['scan_angle'].mean().values)
    overpass_period = 'PM' if timestamp_pd.hour >= 12 else 'AM'
    
    swath_ds_cropped['satellite'] = sat
    swath_ds_cropped['timestamp_str'] = str(timestamp_pd)
    swath_ds_cropped['daynight'] = daynight
    swath_ds_cropped['overpass_period'] = overpass_period
    swath_ds_cropped['avg_scan_angle_scene'] = avg_scan_angle
    
    # Structural metadata stays as attributes
    swath_ds_cropped.attrs = {
        'bbox': bbox,
        'original_shape': (n_lines, n_pixels),
        'cropped_shape': (
            swath_ds_cropped.sizes['scan'],
            swath_ds_cropped.sizes['pixel']),
        'avg_scan_angle_scene': avg_scan_angle,
        'daynight': daynight,
        'overpass_period': overpass_period
    }

    # ===================================================================
    # OPTIONAL: VERIFY CROPPED DATA MATCHES SOURCE ARRAYS
    # ===================================================================
    
    if run_tests:
        if verbose:
            print("  [TEST] Verifying cropped data against source arrays...")
        
        crop_scans = swath_ds_cropped['scan'].values
        crop_pixels = swath_ds_cropped['pixel'].values
        scan_idx = np.ix_(crop_scans, crop_pixels)
        
        tests = {
            'longitude': (longitude[scan_idx],  swath_ds_cropped['longitude'].values),
            'latitude':  (latitude[scan_idx],   swath_ds_cropped['latitude'].values),
            'I4_bt':     (i4_bt[scan_idx],      swath_ds_cropped['I4_bt'].values),
            'I5_bt':     (i5_bt[scan_idx],      swath_ds_cropped['I5_bt'].values),
            'fire_mask': (fire_mask[scan_idx],   swath_ds_cropped['fire_mask'].values),
        }
        
        failures = []
        for name, (source, compiled) in tests.items():
            if np.issubdtype(source.dtype, np.floating):
                match = np.allclose(source, compiled, equal_nan=True)
            else:
                match = np.array_equal(source, compiled)
            
            if not match:
                if np.issubdtype(source.dtype, np.floating):
                    n_diff = (~np.isclose(source, compiled, equal_nan=True)).sum()
                else:
                    n_diff = (source != compiled).sum()
                failures.append(f"{name}: {n_diff} mismatched values")
        
        if failures:
            failure_msg = "\n    ".join(failures)
            raise ValueError(
                f"Fidelity test failed for {sat} {timestamp_pd}:\n    {failure_msg}"
            )
        
        if verbose:
            print("    All fidelity tests passed")
            
    return swath_ds_cropped

def plot_swath_overview(swath_data, bbox, output_path=None, zoom_padding=0.01):
    """
    Create a 9-panel visualization of VIIRS swath data.
    
    Top row: I4 BT, Candidate Detections (on I4 BT), I5 BT
    Middle row: Fire Mask, Scan Angle, FP_power
    Bottom row: I4 BT (zoomed), Candidate Detections (zoomed), FP_power (zoomed)
    
    Parameters
    ----------
    swath_data : xr.Dataset or dict
        xarray Dataset or dictionary containing swath data
    bbox : list
        Bounding box [lon_min, lat_min, lon_max, lat_max]
    output_path : str, optional
        Path to save the figure. If None, displays interactively.
    zoom_padding : float, optional
        Padding (in degrees) to add around detection extent for zoom panels (default: 0.01)
    """
        
    # Handle both xarray Dataset and dict inputs
    if isinstance(swath_data, xr.Dataset):
        lon = swath_data['longitude'].values
        lat = swath_data['latitude'].values
        i4_bt = swath_data['I4_bt'].values
        i5_bt = swath_data['I5_bt'].values
        fire_mask = swath_data['fire_mask'].values
        scan_angle = swath_data['scan_angle'].values
        pixel_area = swath_data['pixel_area'].values
        fp_power = swath_data['FP_power'].values if 'FP_power' in swath_data else None
        candidate_confidence = swath_data['candidate_confidence'].values if 'candidate_confidence' in swath_data else None
        sat = swath_data['satellite'].item()
        timestamp = pd.Timestamp(swath_data['timestamp_str'].item())
    else:
        # Original dict format
        lon = swath_data['longitude']
        lat = swath_data['latitude']
        i4_bt = swath_data['I4_bt']
        i5_bt = swath_data['I5_bt']
        fire_mask = swath_data['fire_mask']
        scan_angle = swath_data['scan_angle']
        pixel_area = swath_data['pixel_area']
        fp_power = swath_data.get('FP_power', None)
        candidate_confidence = swath_data.get('candidate_confidence', None)
        sat = swath_data['satellite']
        timestamp = swath_data['timestamp']
    
    # Create custom colormaps (preserve NaNs as white)
    cmap_plasma = plt.cm.plasma.copy()
    cmap_plasma.set_bad(color='white', alpha=1)
    
    cmap_viridis = plt.cm.viridis.copy()
    cmap_viridis.set_bad(color='white', alpha=1)
    
    cmap_hot = plt.cm.hot.copy()
    cmap_hot.set_bad(color='white', alpha=1)
    
    # Fire mask colormap
    mask_colors = [mpl.colormaps['tab10'](c) for c in [4, 6, 5, 0, 9, 2, 7, 8, 1, 3]]
    cmap_fire = ListedColormap(mask_colors)
    
    # ===================================================================
    # CALCULATE ZOOM EXTENT BASED ON DETECTIONS
    # ===================================================================
    zoom_extent = None
    has_detections = False
    
    if candidate_confidence is not None:
        candidate_locs = np.isfinite(candidate_confidence)  # Any non-NaN value
        if candidate_locs.any():
            has_detections = True
            # Get lon/lat of all detections
            det_lons = lon[candidate_locs]
            det_lats = lat[candidate_locs]
            
            # Calculate extent with padding
            # NOTE: set_extent expects [lon_min, lon_max, lat_min, lat_max]
            zoom_extent = [
                det_lons.min() - zoom_padding,  # lon_min
                det_lons.max() + zoom_padding,  # lon_max
                det_lats.min() - zoom_padding,  # lat_min
                det_lats.max() + zoom_padding   # lat_max
            ]
    
    # Create figure with 9 panels (3 rows × 3 columns)
    fig = plt.figure(figsize=(20, 18))
    
    # Top row
    ax1 = plt.subplot(3, 3, 1, projection=ccrs.PlateCarree())
    ax2 = plt.subplot(3, 3, 2, projection=ccrs.PlateCarree())
    ax3 = plt.subplot(3, 3, 3, projection=ccrs.PlateCarree())
    
    # Middle row
    ax4 = plt.subplot(3, 3, 4, projection=ccrs.PlateCarree())
    ax5 = plt.subplot(3, 3, 5, projection=ccrs.PlateCarree())
    ax6 = plt.subplot(3, 3, 6, projection=ccrs.PlateCarree())
    
    # Bottom row (zoomed)
    ax7 = plt.subplot(3, 3, 7, projection=ccrs.PlateCarree())
    ax8 = plt.subplot(3, 3, 8, projection=ccrs.PlateCarree())
    ax9 = plt.subplot(3, 3, 9, projection=ccrs.PlateCarree())
    
    plot_extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
    
    # ===================================================================
    # TOP ROW
    # ===================================================================
    
    # Panel 1: I4 Brightness Temperature
    ax1.set_extent(plot_extent)
    plot1 = ax1.pcolormesh(lon, lat, i4_bt, 
                           vmin=250, vmax=360, 
                           cmap=cmap_plasma, 
                           transform=ccrs.PlateCarree())
    ax1.set_title(f"I4 BT (3.75 µm)\n{sat} {timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                  fontsize=12)
    
    # Panel 2: Candidate Detection Overlay (on I4 BT background)
    ax2.set_extent(plot_extent)
    
    # Start with I4 BT background
    plot2_bg = ax2.pcolormesh(lon, lat, i4_bt, 
                              vmin=250, vmax=360, 
                              cmap=cmap_plasma, 
                              transform=ccrs.PlateCarree())
    
    # Add candidate overlay if available
    n_confirmed = 0
    n_qa_candidates = 0
    if candidate_confidence is not None:
        # Get candidate locations
        candidate_locs = np.isfinite(candidate_confidence)  # Any non-NaN value
        
        if candidate_locs.any():
            # Separate into white (QA candidates) and black (confirmed fires)
            white_candidates = candidate_locs & (candidate_confidence == 0)  # QA only
            black_fires = candidate_locs & (candidate_confidence >= 1)  # Confirmed (1, 2, or 3)
            
            # Plot white candidates first (background)
            if white_candidates.any():
                ax2.scatter(lon[white_candidates], lat[white_candidates], 
                           c='white', s=1.5, alpha=0.8, edgecolors='black', linewidths=0.2,
                           transform=ccrs.PlateCarree(), label='QA candidates')
            
            # Plot black fires on top
            if black_fires.any():
                ax2.scatter(lon[black_fires], lat[black_fires], 
                           c='black', s=1.5, alpha=0.9,
                           transform=ccrs.PlateCarree(), label='Confirmed fires')
            
            # Count detections
            n_confirmed = black_fires.sum()
            n_qa_candidates = white_candidates.sum()
            
            # Add text annotations
            ax2.text(0.02, 0.98, f'{n_confirmed} confirmed fire pixels', 
                    c='black', transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
            ax2.text(0.02, 0.92, f'{n_qa_candidates} QA candidate pixels', 
                    c='white', transform=ax2.transAxes, fontsize=10,
                    verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, pad=0.3))
            
            ax2.set_title(f"Candidate Detections\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                         fontsize=12)
        else:
            ax2.set_title(f"Candidate Detections (None)\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                         fontsize=12)
    else:
        ax2.set_title(f"Candidate Detections\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                     fontsize=12)
    
    # Panel 3: I5 Brightness Temperature
    ax3.set_extent(plot_extent)
    plot3 = ax3.pcolormesh(lon, lat, i5_bt, 
                           vmin=250, vmax=360, 
                           cmap=cmap_plasma, 
                           transform=ccrs.PlateCarree())
    ax3.set_title(f"I5 BT (11 µm)\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                  fontsize=12)
    
    # ===================================================================
    # MIDDLE ROW
    # ===================================================================
    
    # Panel 4: Fire Mask
    ax4.set_extent(plot_extent)
    plot4 = ax4.pcolormesh(lon, lat, fire_mask, 
                           vmin=0, vmax=10, 
                           cmap=cmap_fire, 
                           transform=ccrs.PlateCarree())
    ax4.set_title(f"Fire Mask\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                  fontsize=12)
    
    # Panel 5: Scan Angle (with pixel area range in title)
    ax5.set_extent(plot_extent)
    plot5 = ax5.pcolormesh(lon, lat, scan_angle, 
                           cmap=cmap_viridis, 
                           transform=ccrs.PlateCarree())
    ax5.set_title(f"Scan Angle: {np.nanmin(scan_angle):.1f}° - {np.nanmax(scan_angle):.1f}°\n"
                  f"Pixel Area: {np.nanmin(pixel_area):.1f} - {np.nanmax(pixel_area):.1f} km²", 
                  fontsize=12)
    
    # Panel 6: FP_power (if available)
    ax6.set_extent(plot_extent)
    plot6 = None
    if fp_power is not None:
        # Only plot non-NaN values for better color scaling
        fp_power_valid = np.ma.masked_invalid(fp_power)
        if fp_power_valid.count() > 0:  # Check if there are any valid values
            plot6 = ax6.pcolormesh(lon, lat, fp_power_valid, 
                                   cmap=cmap_hot, 
                                   transform=ccrs.PlateCarree(),
                                   vmin=0, vmax=100)
            n_detections = fp_power_valid.count()
            ax6.set_title(f"Fire Radiative Power\n{n_detections} detections", 
                         fontsize=12)
        else:
            ax6.text(0.5, 0.5, 'No FRP detections', 
                    transform=ax6.transAxes, ha='center', va='center', fontsize=14)
            ax6.set_title("Fire Radiative Power", fontsize=12)
    else:
        ax6.text(0.5, 0.5, 'FP_power not available', 
                transform=ax6.transAxes, ha='center', va='center', fontsize=14)
        ax6.set_title("Fire Radiative Power", fontsize=12)
    
    # ===================================================================
    # BOTTOM ROW (ZOOMED TO DETECTIONS)
    # ===================================================================
    
    plot7 = None
    plot8_bg = None
    plot9 = None
    
    if has_detections and zoom_extent is not None:
        
        # Panel 7: I4 BT (Zoomed)
        ax7.set_extent(zoom_extent, crs=ccrs.PlateCarree())
        plot7 = ax7.pcolormesh(lon, lat, i4_bt, 
                               vmin=250, vmax=360, 
                               cmap=cmap_plasma, 
                               transform=ccrs.PlateCarree(),
                               shading='auto')
        ax7.set_title(f"I4 BT (Zoomed)\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                      fontsize=12)
        
        # Panel 8: Candidate Detections (Zoomed)
        ax8.set_extent(zoom_extent, crs=ccrs.PlateCarree())
        
        # I4 BT background
        plot8_bg = ax8.pcolormesh(lon, lat, i4_bt, 
                                  vmin=250, vmax=360, 
                                  cmap=cmap_plasma, 
                                  transform=ccrs.PlateCarree(),
                                  shading='auto')
        
        # Add candidate overlay- same logic as above
        if candidate_confidence is not None:
            candidate_locs = np.isfinite(candidate_confidence)
            
            if candidate_locs.any():
                white_candidates = candidate_locs & (candidate_confidence == 0)
                black_fires = candidate_locs & (candidate_confidence >= 1)
                
                # Use larger markers for zoomed view
                if white_candidates.any():
                    ax8.scatter(lon[white_candidates], lat[white_candidates], 
                               c='white', s=8, alpha=0.8, edgecolors='black', linewidths=0.5,
                               transform=ccrs.PlateCarree())
                
                if black_fires.any():
                    ax8.scatter(lon[black_fires], lat[black_fires], 
                               c='black', s=8, alpha=0.9,
                               transform=ccrs.PlateCarree())
                
                # Add text annotations
                ax8.text(0.02, 0.98, f'{n_confirmed} confirmed', 
                        c='black', transform=ax8.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, pad=0.3))
                ax8.text(0.02, 0.92, f'{n_qa_candidates} QA candidates', 
                        c='white', transform=ax8.transAxes, fontsize=10,
                        verticalalignment='top',
                        bbox=dict(boxstyle='round', facecolor='black', alpha=0.8, pad=0.3))
        
        ax8.set_title(f"Candidate Detections (Zoomed)\n{timestamp.strftime('%Y-%m-%d %H:%M')} UTC", 
                     fontsize=12)
        
        # Panel 9: FP_power (Zoomed)
        ax9.set_extent(zoom_extent, crs=ccrs.PlateCarree())
        if fp_power is not None:
            fp_power_valid = np.ma.masked_invalid(fp_power)
            
            if fp_power_valid.count() > 0:
                plot9 = ax9.pcolormesh(lon, lat, fp_power_valid, 
                                       cmap=cmap_hot, 
                                       transform=ccrs.PlateCarree(),
                                       vmin=0, vmax=100,
                                       shading='auto')
                
                # Add confirmed fire points overlay
                if candidate_confidence is not None:
                    candidate_locs = np.isfinite(candidate_confidence)
                    if candidate_locs.any():
                        # Only plot confirmed fires (confidence >= 1)
                        black_fires = candidate_locs & (candidate_confidence >= 1)
                        
                        if black_fires.any():
                            ax9.scatter(lon[black_fires], lat[black_fires], 
                                       c='black', s=8, alpha=0.9,
                                       transform=ccrs.PlateCarree(),
                                       label='Confirmed fires')
                
                ax9.set_title(f"FRP (Zoomed)\n{fp_power_valid.count()} detections", 
                             fontsize=12)
            else:
                ax9.text(0.5, 0.5, 'No FRP in zoom area', 
                        transform=ax9.transAxes, ha='center', va='center', fontsize=14)
                ax9.set_title("FRP (Zoomed)", fontsize=12)
        else:
            ax9.text(0.5, 0.5, 'FP_power not available', 
                    transform=ax9.transAxes, ha='center', va='center', fontsize=14)
            ax9.set_title("FRP (Zoomed)", fontsize=12)
        
    else:
        # No detections - show message in all three zoomed panels
        for ax in [ax7, ax8, ax9]:
            ax.text(0.5, 0.5, 'No detections to zoom', 
                   transform=ax.transAxes, ha='center', va='center', fontsize=14)
            ax.set_title("(No detections)", fontsize=12)
    
    # Add axis labels without gridlines
    for ax in [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]:
        gl = ax.gridlines(draw_labels=True, linewidth=0)
        gl.top_labels = False
        gl.right_labels = False
        gl.xlabel_style = {'size': 8}
        gl.ylabel_style = {'size': 8}
    
    # Adjust layout for colorbars
    plt.subplots_adjust(bottom=0.05, top=0.98, hspace=0.25, wspace=0.3, left=0.05, right=0.95)
    
    # Add colorbars under each row of panels
    # Bottom row colorbars (under panels 7, 8, 9)
    if has_detections and zoom_extent is not None:
        if plot7 is not None:
            cbar_ax7 = fig.add_axes([0.08, 0.03, 0.22, 0.01])
            cbar7 = plt.colorbar(plot7, cax=cbar_ax7, orientation='horizontal')
            cbar7.set_label('I4 Temperature (K)', fontsize=8)
            cbar7.ax.tick_params(labelsize=7)
        
        if plot8_bg is not None:
            cbar_ax8 = fig.add_axes([0.39, 0.03, 0.22, 0.01])
            cbar8 = plt.colorbar(plot8_bg, cax=cbar_ax8, orientation='horizontal')
            cbar8.set_label('I4 Temperature (K)', fontsize=8)
            cbar8.ax.tick_params(labelsize=7)
        
        if plot9 is not None:
            cbar_ax9 = fig.add_axes([0.70, 0.03, 0.22, 0.01])
            cbar9 = plt.colorbar(plot9, cax=cbar_ax9, orientation='horizontal')
            cbar9.set_label('FRP (MW)', fontsize=8)
            cbar9.ax.tick_params(labelsize=7)
    
    # Middle row colorbars (under panels 4, 5, 6)
    cbar_ax4 = fig.add_axes([0.08, 0.37, 0.22, 0.01])
    cbar4 = plt.colorbar(plot4, cax=cbar_ax4, orientation='horizontal')
    fire_labels = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    cbar4.ax.set_xticks(np.arange(len(fire_labels)))
    cbar4.ax.set_xticklabels(fire_labels, fontsize=7)
    cbar4.set_label('Fire Mask Categories', fontsize=8)
    
    cbar_ax5 = fig.add_axes([0.39, 0.37, 0.22, 0.01])
    cbar5 = plt.colorbar(plot5, cax=cbar_ax5, orientation='horizontal')
    cbar5.set_label('Scan Angle (degrees)', fontsize=8)
    cbar5.ax.tick_params(labelsize=7)
    
    if plot6 is not None:
        cbar_ax6 = fig.add_axes([0.70, 0.37, 0.22, 0.01])
        cbar6 = plt.colorbar(plot6, cax=cbar_ax6, orientation='horizontal')
        cbar6.set_label('FRP (MW)', fontsize=8)
        cbar6.ax.tick_params(labelsize=7)
    
    # Top row colorbars (under panels 1, 2, 3)
    cbar_ax1 = fig.add_axes([0.08, 0.69, 0.22, 0.01])
    cbar1 = plt.colorbar(plot1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_label('I4 Temperature (K)', fontsize=8)
    cbar1.ax.tick_params(labelsize=7)
    
    cbar_ax2 = fig.add_axes([0.39, 0.69, 0.22, 0.01])
    cbar2 = plt.colorbar(plot2_bg, cax=cbar_ax2, orientation='horizontal')
    cbar2.set_label('I4 Temperature (K)', fontsize=8)
    cbar2.ax.tick_params(labelsize=7)
    
    cbar_ax3 = fig.add_axes([0.70, 0.69, 0.22, 0.01])
    cbar3 = plt.colorbar(plot3, cax=cbar_ax3, orientation='horizontal')
    cbar3.set_label('I5 Temperature (K)', fontsize=8)
    cbar3.ax.tick_params(labelsize=7)
    
    # Save or show
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  Figure saved to: {output_path}")
        plt.close()
    else:
        plt.show()

def test_spatial_alignment(swath, tolerance_degrees=0.0001, tolerance_temp=0.1, verbose=True):
    """
    Test spatial alignment by comparing FP_* coordinates and temperatures with grid values.
    
    This verifies that rasterized FP values are placed at the correct spatial locations
    by checking:
    1. FP_latitude matches the main latitude grid at detection pixels
    2. FP_longitude matches the main longitude grid at detection pixels
    3. FP_T5 matches I5_bt at detection pixels (I5 doesn't saturate like I4)
    
    Parameters
    ----------
    swath : xr.Dataset
        Processed swath with FP_latitude and FP_longitude rasterized
    tolerance_degrees : float
        Maximum acceptable difference in lat/lon (degrees). Default 0.0001° ≈ 10m
    tolerance_temp : float
        Maximum acceptable difference in temperature (K). Default 0.1K
    verbose : bool
        If True, prints all results. If False, only prints when mismatches found.
    
    Returns
    -------
    test_results : dict
        Spatial alignment test results
    """
    
    if verbose:
        print("="*70)
        print("SPATIAL ALIGNMENT VERIFICATION TEST")
        print("="*70)
        
    # Extract data
    longitude_grid = swath['longitude'].values
    latitude_grid = swath['latitude'].values
    fp_longitude = swath['FP_longitude'].values
    fp_latitude = swath['FP_latitude'].values
    i5_bt = swath['I5_bt'].values
    fp_t5 = swath['FP_T5'].values
    
    if verbose:
        print(f"\nSwath info:")
        print(f"  Satellite: {swath['satellite'].item()}")
        print(f"  Timestamp: {swath['timestamp_str'].item()}")
        print(f"  Shape: {longitude_grid.shape}")
    
    # Find detection locations
    detection_mask = ~np.isnan(fp_longitude)
    detection_rows, detection_cols = np.where(detection_mask)
    n_detections = len(detection_rows)
    
    if verbose:
        print(f"\nDetections found: {n_detections}")
    
    if n_detections == 0:
        if verbose:
            print("No detections - nothing to test")
        return {'passed': True, 'n_detections': 0, 'message': 'No detections'}
    
    # ===================================================================
    # Calculate all differences (always compute, print conditionally)
    # ===================================================================
    
    # Extract values at detection locations
    fp_lon_at_detections = fp_longitude[detection_rows, detection_cols]
    grid_lon_at_detections = longitude_grid[detection_rows, detection_cols]
    fp_lat_at_detections = fp_latitude[detection_rows, detection_cols]
    grid_lat_at_detections = latitude_grid[detection_rows, detection_cols]
    fp_t5_at_detections = fp_t5[detection_rows, detection_cols]
    i5_bt_at_detections = i5_bt[detection_rows, detection_cols]
    
    # Calculate differences
    lon_diff = np.abs(fp_lon_at_detections - grid_lon_at_detections)
    lat_diff = np.abs(fp_lat_at_detections - grid_lat_at_detections)
    t5_diff = np.abs(fp_t5_at_detections - i5_bt_at_detections)
    
    # Statistics
    lon_mean_diff = np.mean(lon_diff)
    lon_max_diff = np.max(lon_diff)
    lon_perfect_matches = (lon_diff == 0).sum()
    
    lat_mean_diff = np.mean(lat_diff)
    lat_max_diff = np.max(lat_diff)
    lat_perfect_matches = (lat_diff == 0).sum()
    
    t5_mean_diff = np.mean(t5_diff)
    t5_max_diff = np.max(t5_diff)
    t5_perfect_matches = (t5_diff == 0).sum()
    
    # Determine pass/fail
    lon_test_passed = lon_max_diff < tolerance_degrees
    lat_test_passed = lat_max_diff < tolerance_degrees
    t5_test_passed = t5_max_diff < tolerance_temp
    all_passed = lon_test_passed and lat_test_passed and t5_test_passed
    
    # ===================================================================
    # PRINT RESULTS (based on verbose setting)
    # ===================================================================
    
    # If verbose=True, print everything
    # If verbose=False, only print if there are mismatches
    should_print = verbose or not all_passed
    
    if should_print:
        
        # TEST 1: Longitude
        print("\n" + "-"*70)
        print("TEST 1: Longitude Alignment")
        print("-"*70)
        print(f"  Perfect matches: {lon_perfect_matches}/{n_detections} ({100*lon_perfect_matches/n_detections:.1f}%)")
        print(f"  Mean difference: {lon_mean_diff:.6f}°")
        print(f"  Max difference: {lon_max_diff:.6f}°")
        
        print(f"\n  Sample comparisons (first 5 detections):")
        for i in range(min(5, n_detections)):
            print(f"    Detection {i}: FP_lon={fp_lon_at_detections[i]:.6f}°, Grid_lon={grid_lon_at_detections[i]:.6f}°, diff={lon_diff[i]:.8f}°")
        
        print(f"\n  TEST 1: {'PASSED' if lon_test_passed else 'FAILED'} (tolerance: {tolerance_degrees}°)")
        
        # TEST 2: Latitude
        print("\n" + "-"*70)
        print("TEST 2: Latitude Alignment")
        print("-"*70)
        print(f"  Perfect matches: {lat_perfect_matches}/{n_detections} ({100*lat_perfect_matches/n_detections:.1f}%)")
        print(f"  Mean difference: {lat_mean_diff:.6f}°")
        print(f"  Max difference: {lat_max_diff:.6f}°")
        
        print(f"\n  Sample comparisons (first 5 detections):")
        for i in range(min(5, n_detections)):
            print(f"    Detection {i}: FP_lat={fp_lat_at_detections[i]:.6f}°, Grid_lat={grid_lat_at_detections[i]:.6f}°, diff={lat_diff[i]:.8f}°")
        
        print(f"\n  TEST 2: {'PASSED' if lat_test_passed else 'FAILED'} (tolerance: {tolerance_degrees}°)")
        
        # TEST 3: I5 Temperature
        print("\n" + "-"*70)
        print("TEST 3: I5 Temperature Alignment")
        print("-"*70)
        print(f"  Perfect matches: {t5_perfect_matches}/{n_detections} ({100*t5_perfect_matches/n_detections:.1f}%)")
        print(f"  Mean difference: {t5_mean_diff:.4f} K")
        print(f"  Max difference: {t5_max_diff:.4f} K")
        
        print(f"\n  Sample comparisons (first 5 detections):")
        for i in range(min(5, n_detections)):
            print(f"    Detection {i}: FP_T5={fp_t5_at_detections[i]:.2f} K, I5_bt={i5_bt_at_detections[i]:.2f} K, diff={t5_diff[i]:.4f} K")
        
        print(f"\n  TEST 3: {'PASSED' if t5_test_passed else 'FAILED'} (tolerance: {tolerance_temp} K)")
        
        # Summary
        print("\n" + "="*70)
        print("SPATIAL ALIGNMENT TEST SUMMARY")
        print("="*70)
        print(f"  TEST 1 (Longitude): {'PASSED' if lon_test_passed else 'FAILED'}")
        print(f"  TEST 2 (Latitude):  {'PASSED' if lat_test_passed else 'FAILED'}")
        print(f"  TEST 3 (I5 Temp):   {'PASSED' if t5_test_passed else 'FAILED'}")
        
        if all_passed:
            print(f"\n  SPATIAL ALIGNMENT VERIFIED")
        else:
            print(f"\n  SPATIAL ALIGNMENT ISSUES DETECTED")
    
    # Always return results dict
    return {
        'passed': all_passed,
        'n_detections': n_detections,
        'longitude_test': {
            'passed': lon_test_passed,
            'mean_diff': lon_mean_diff,
            'max_diff': lon_max_diff,
            'perfect_matches': lon_perfect_matches
        },
        'latitude_test': {
            'passed': lat_test_passed,
            'mean_diff': lat_mean_diff,
            'max_diff': lat_max_diff,
            'perfect_matches': lat_perfect_matches
        },
        'i5_temp_test': {
            'passed': t5_test_passed,
            'mean_diff': t5_mean_diff,
            'max_diff': t5_max_diff,
            'perfect_matches': t5_perfect_matches
        }
    }

def log_message(message, log_file, print_to_console=True, include_timestamp=True):
    """Write message to both log file and optionally console.
    
    Parameters
    ----------
    message : str
        The message to log
    print_to_console : bool, default=True
        Whether to also print to console
    include_timestamp : bool, default=True
        Whether to prepend timestamp to message
    """
    if include_timestamp:
        timestamp = dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_message = f"[{timestamp}] {message}"
    else:
        formatted_message = message
    
    log_file.write(formatted_message + '\n')
    log_file.flush()  # Ensure it's written immediately
    if print_to_console:
        print(formatted_message)

def process_swaths(fire_name, start, end, bbox, n_timesteps, pix_lut_path=None,
                   sensors=['SNPP', 'NOAA20', 'NOAA21'], make_plots=False,
                   save_data=True, overwrite=False, run_spatial_test=True,
                   copy_to_s3=False, s3_prefix=None,
                   output_dir='VIIRS-cubed-outputs', remove_local=False):

    '''
    Main orchestration function which processes an arbitrary number
    of VIIRS swaths for a region of interest.
    
    '''

    if copy_to_s3 and s3_prefix is None:
        raise ValueError("s3_prefix is required when copy_to_s3=True")
    if remove_local and not copy_to_s3:
        raise ValueError("remove_local=True requires copy_to_s3=True")

    # ===================================================================
    # CREATE ORGANIZED DIRECTORY STRUCTURE
    # ===================================================================

    base_output_dir = os.path.join(os.path.abspath(output_dir), f"{fire_name}_Gridded_VIIRS")
    plots_dir = os.path.join(base_output_dir, "Plots", "Step1_Compiled_Swaths")
    data_dir = os.path.join(base_output_dir, "Data", "Step1_Compiled_Swaths")
    logs_dir = os.path.join(base_output_dir, "Logs")
    
    # Make all directories if not already created
    for directory in [base_output_dir, plots_dir, data_dir, logs_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")

    print(f"\nOutput structure:")
    print(f"  Base: {base_output_dir}")
    print(f"  Data: {data_dir}")
    print(f"  Plots: {plots_dir}")
    print(f"  Logs: {logs_dir}")

    run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{fire_name}_processing_log_{run_timestamp}.txt"
    log_path = os.path.join(logs_dir, log_filename)

    # Open log file
    log_file = open(log_path, 'w')

    # Log header (without timestamps for clean formatting)
    log_message("="*70, log_file, include_timestamp=False)
    log_message(f"VIIRS SWATH PROCESSING LOG", log_file, include_timestamp=False)
    log_message(f"Run started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file, include_timestamp=False)
    log_message("="*70, log_file, include_timestamp=False)
    log_message(f"Fire name: {fire_name}", log_file,)
    log_message(f"Bounding box: {bbox}", log_file,)
    log_message(f"Date range: {start} to {end}", log_file,)
    log_message(f"Sensors: {sensors}", log_file,)
    log_message(f"Max timesteps: {n_timesteps}", log_file,)
    log_message(f"Overwrite: {overwrite}", log_file,)
    log_message(f"Make plots: {make_plots}", log_file,)
    log_message(f"Run spatial test: {run_spatial_test}", log_file,)
    log_message(f"Copy to S3: {copy_to_s3}", log_file,)
    if copy_to_s3:
        log_message(f"S3 prefix: {s3_prefix}", log_file,)
    log_message("", log_file,)

    # ===================================================================
    # PROCESS ALL TIMESTEPS
    # ===================================================================
    
    log_message(f"{'='*70}", log_file, include_timestamp=False)
    log_message(f"PROCESSING {n_timesteps} TIMESTEPS", log_file,)
    log_message(f"{'='*70}", log_file, include_timestamp=False)
    if not overwrite:
        log_message(f"OVERWRITE = False: Skipping existing files", log_file,)
    log_message("", log_file,)
    
    processed_count = 0
    skipped_count = 0
    already_exists_count = 0
    error_count = 0
    spatial_test_failures = []

    all_timesteps = gather_filepaths(bbox, start, end, sensors)

    if n_timesteps == -1:
        timesteps_to_process = all_timesteps
    else:
        timesteps_to_process = all_timesteps[:n_timesteps]
    
    # Create progress bar
    pbar = tqdm(timesteps_to_process, desc="Processing swaths", unit="swath")
    
    for timestep_count, timestep_info in enumerate(pbar):
    
        # Update progress bar description with current satellite/time
        sat = timestep_info['satellite']
        timestamp = timestep_info['timestamp']
        file_timestamp = timestamp.strftime('%Y%m%d_%H%M')
        
        pbar.set_description(f"Processing {sat} {timestamp.strftime('%Y-%m-%d %H:%M')}")
        
        # Define output filenames
        data_filename = f"{sat}_{file_timestamp}_swath.nc"
        data_output_path = os.path.join(data_dir, data_filename)
        
        plot_filename = f"{sat}_{file_timestamp}_swath.png"
        plot_output_path = os.path.join(plots_dir, plot_filename)
        
        # Check if files already exist
        data_exists = os.path.exists(data_output_path) if save_data else False
        plot_exists = os.path.exists(plot_output_path) if make_plots else False
        
        # Skip if both files exist and OVERWRITE is False
        if not overwrite:
            skip_data = save_data and data_exists
            skip_plot = make_plots and plot_exists
            
            if (not save_data or skip_data) and (not make_plots or skip_plot):
                already_exists_count += 1
                log_message(f"Skipping {data_filename} - already exists", log_file, print_to_console=False)
                pbar.set_postfix({
                    'processed': processed_count, 
                    'exists': already_exists_count,
                    'skipped': skipped_count, 
                    'errors': error_count
                })
                continue
        
        # Load and process swath
        try:
            swath = combine_viirs_swaths(timestep_info, bbox, pix_lut_path, run_tests=True)
            
            if swath is None:
                skipped_count += 1
                log_message(f"Skipping {data_filename} - no bbox intersection", log_file, print_to_console=False)
                pbar.set_postfix({
                    'processed': processed_count, 
                    'exists': already_exists_count,
                    'skipped': skipped_count, 
                    'errors': error_count
                })
                continue
            
            # Run spatial alignment test (if enabled)
            if run_spatial_test:
                spatial_results = test_spatial_alignment(swath, verbose=False)
                
                if spatial_results and not spatial_results['passed']:
                    fail_msg = f"\nSPATIAL TEST FAILED: {data_filename}"
                    log_message(fail_msg, log_file, print_to_console=True)
                    log_message(f"  Longitude test: {'PASSED' if spatial_results['longitude_test']['passed'] else 'FAILED'}", log_file,)
                    log_message(f"  Latitude test:  {'PASSED' if spatial_results['latitude_test']['passed'] else 'FAILED'}", log_file,)
                    log_message(f"  I5 temp test:   {'PASSED' if spatial_results['i5_temp_test']['passed'] else 'FAILED'}", log_file,)
                    
                    spatial_test_failures.append({
                        'filename': data_filename,
                        'timestamp': str(timestamp),
                        'results': spatial_results
                    })
            
            # Extract metadata for filenames
            sat = swath['satellite'].item()
            timestamp = pd.Timestamp(swath['timestamp_str'].item())
            file_timestamp = timestamp.strftime('%Y%m%d_%H%M')
            
            # Save data (if needed)
            if save_data and (overwrite or not data_exists):
                swath.to_netcdf(data_output_path)
                log_message(f"Saved: {data_filename}", log_file, print_to_console=False)
            
            # Create plot (if needed)
            if make_plots and (overwrite or not plot_exists):
                plot_swath_overview(
                    swath_data=swath,
                    bbox=bbox,
                    output_path=plot_output_path
                )
                log_message(f"Plotted: {plot_filename}", log_file, print_to_console=False)
            
            processed_count += 1
            pbar.set_postfix({
                'processed': processed_count, 
                'exists': already_exists_count,
                'skipped': skipped_count, 
                'errors': error_count
            })
            
        except Exception as e:
            error_count += 1
            error_msg = f"ERROR processing {data_filename}: {str(e)}"
            log_message(error_msg, log_file, print_to_console=True)
            pbar.set_postfix({
                'processed': processed_count, 
                'exists': already_exists_count,
                'skipped': skipped_count, 
                'errors': error_count
            })
            continue
    
    pbar.close()

    if copy_to_s3:
        s3_dest = f"{s3_prefix.rstrip('/')}/{fire_name}_Gridded_VIIRS"
        log_message(f"\n{'='*70}", log_file, include_timestamp=False)
        log_message(f"COPYING TO S3", log_file,)
        log_message(f"  Source:      {base_output_dir}/", log_file,)
        log_message(f"  Destination: {s3_dest}/", log_file,)
        log_message(f"{'='*70}", log_file, include_timestamp=False)
    
        if overwrite:
            cmd = ["aws", "s3", "cp", base_output_dir, s3_dest, "--recursive"]
        else:
            cmd = ["aws", "s3", "sync", base_output_dir, s3_dest]
    
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
    
        if result.returncode == 0:
            log_message(f"S3 upload complete!", log_file,)
            log_message(f"  S3 location: {s3_dest}/", log_file,)
            if remove_local:
                shutil.rmtree(base_output_dir)
                log_message(f"Local files removed: {base_output_dir}", log_file,)
        else:
            log_message(f"S3 upload failed!", log_file,)
            log_message(f"  Error: {result.stderr}", log_file,)
    
    log_message(f"\n{'='*70}", log_file, include_timestamp=False)
    log_message(f"Log file saved: {log_path}", log_file,)
    log_message(f"{'='*70}", log_file, include_timestamp=False)

    # ===================================================================
    # SUMMARY
    # ===================================================================
    
    log_message(f"\n{'='*70}", log_file, include_timestamp=False)
    log_message(f"PROCESSING COMPLETE", log_file,)
    log_message(f"{'='*70}", log_file, include_timestamp=False)
    log_message(f"  Successfully processed: {processed_count} swaths", log_file,)
    log_message(f"  Already existed (skipped): {already_exists_count} swaths", log_file,)
    log_message(f"  Skipped (no bbox intersection): {skipped_count} swaths", log_file,)
    log_message(f"  Errors: {error_count} swaths", log_file,)
    log_message(f"  Total: {processed_count + already_exists_count + skipped_count + error_count} timesteps", log_file,)
    
    if run_spatial_test:
        log_message(f"\n  Spatial alignment tests run: {processed_count}", log_file,)
        log_message(f"  Spatial test failures: {len(spatial_test_failures)}", log_file,)
        
        if len(spatial_test_failures) > 0:
            log_message(f"\n  Files with spatial alignment issues:", log_file,)
            for failure in spatial_test_failures:
                log_message(f"    - {failure['filename']}", log_file,)
    
    log_message(f"\nOutputs saved to: {base_output_dir}", log_file,)
    if save_data:
        log_message(f"  Data files: {data_dir}", log_file,)
    if make_plots:
        log_message(f"  Plot files: {plots_dir}", log_file,)
    
    log_message(f"\nLog saved to: {log_path}", log_file,)
    log_message(f"Run completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file,)
    
    # Close log file before S3 upload so it's complete
    log_file.close()


    return

if __name__ == "__main__":

    import argparse
    import ast

    parser = argparse.ArgumentParser(
        description="Process VIIRS swath data for a given fire/region of interest."
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
        help="Start date for data query in YYYY-MM-DD format. E.g. '2026-05-10'"
    )
    parser.add_argument(
        "--end",
        type=str,
        required=True,
        help="End date for data query in YYYY-MM-DD format. E.g. '2026-08-30'"
    )
    parser.add_argument(
    "--bbox",
    type=str,
    required=True,
    help="Bounding box as '[xmin, ymin, xmax, ymax]'. E.g. --bbox '[-112.25, 32.25, -111.25, 33.25]'"
    )
    parser.add_argument(
    "--n_timesteps",
    type=int,
    default=-1,
    help="Number of timesteps to process. Use -1 (default) to process all."
    )

    # ===================================================================
    # OPTIONAL ARGUMENTS
    # ===================================================================

    parser.add_argument(
        "--pix_lut_path",
        type=str,
        default=None,
        help="Path to pixel size lookup table CSV. Defaults to S3 path if not provided."
    )
    parser.add_argument(
    "--sensors",
    type=str,
    default='["SNPP", "NOAA20", "NOAA21"]',
    help='Sensors to include as a list of strings. E.g. --sensors \'["SNPP", "NOAA20"]\''
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        default=False,
        help="If set, generate and save overview plots for each swath."
    )
    parser.add_argument(
        "--no_save_data",
        action="store_true",
        default=False,
        help="If set, skip saving processed swaths as NetCDF files."
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
        help="If set, skip spatial alignment verification test on each swath."
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
        help="Local base directory for all outputs. Defaults to 'VIIRS-cubed-outputs'."
    )
    parser.add_argument(
        "--remove_local",
        action="store_true",
        default=False,
        help="If set, remove local output files after a successful S3 upload. Requires --copy_to_s3."
    )

    args = parser.parse_args()

    # ===================================================================
    # VALIDATE ARGUMENT COMBINATIONS
    # ===================================================================

    if args.copy_to_s3 and args.s3_prefix is None:
        parser.error("--s3_prefix is required when --copy_to_s3 is set.")
    if args.remove_local and not args.copy_to_s3:
        parser.error("--remove_local requires --copy_to_s3.")

    # ===================================================================
    # CALL process_swaths
    # ===================================================================

    bbox = ast.literal_eval(args.bbox)
    sensors = ast.literal_eval(args.sensors)
    
    process_swaths(
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
        copy_to_s3=args.copy_to_s3,
        s3_prefix=args.s3_prefix,
        output_dir=args.output_dir,
        remove_local=args.remove_local
    )
    
    # ===================================================================
    # USAGE EXAMPLE
    # ===================================================================    
    #
    # python swath_preprocessing.py \
    #     --fire_name 'Dragon_Bravo_TEST' \
    #     --start '2025-07-01' \
    #     --end '2025-07-10' \
    #     --bbox '[-112.309113, 36.112467, -111.800995, 36.748712]' \
    #     --n_timesteps -1 \
    #     --sensors '["SNPP", "NOAA20", "NOAA21"]' \
    #     --copy_to_s3 \
    #     --s3_prefix 's3://maap-ops-workspace/shared/gsfc_landslides/FireSense/' \
    #     --output_dir 'VIIRS-cubed-outputs' \    
    #     --overwrite \