# Core data processing
import numpy as np
import random
import time
import pandas as pd
import glob
import s3fs
import geopandas as gpd
import shapely
import netCDF4 as nc4
import xarray as xr
import rioxarray as rio
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from rasterio.windows import transform as window_transform
import datetime as dt
import os
from tqdm import tqdm
import pyproj
from shapely.geometry import box, Polygon
import zarr
from zarr.codecs import BloscCodec
import shutil
import gc
import fsspec

# from preprocessing script
from swath_preprocessing import log_message
from utils import compute_fire_persistence_baseline

# Plotting and visualization
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
import matplotlib.lines as mlines
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.gridliner


def query_available_swath_data(fire_name, output_dir='VIIRS-cubed-outputs'):

    # ===================================================================
    # LOCATE STEP 1 OUTPUT FILES
    # ===================================================================

    # Define paths based on Step 1 naming convention
    base_output_dir = os.path.join(os.path.abspath(output_dir), f"{fire_name}_Gridded_VIIRS")
    data_dir = os.path.join(base_output_dir, "Data", "Step1_Compiled_Swaths")
    
    # Check if directory exists
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Step 1 data directory not found: {data_dir}")
    
    # Get all NetCDF files in a list
    swath_files = sorted([
        os.path.join(data_dir, f) 
        for f in os.listdir(data_dir) 
        if f.endswith('_swath.nc')
    ])

    print(f"Found {len(swath_files)} swath files in {data_dir}")

    # Loop through each file and parse metadata to compile into single table
    swath_metadata = []
    bad_files = []
    
    for filepath in swath_files:
        filename = os.path.basename(filepath)
        # Expected format: SATELLITE_YYYYMMDD_HHMM_swath.nc
        parts = filename.replace('_swath.nc', '').split('_')
        
        if len(parts) >= 3:
            satellite = parts[0]
            date_str = parts[1]
            time_str = parts[2]
            
            # Parse timestamp
            timestamp = pd.to_datetime(f"{date_str}_{time_str}", format='%Y%m%d_%H%M')
            
            try:
                # Fast metadata extraction using netCDF4 directly (no data loading)
                with nc4.Dataset(filepath, 'r') as ds:
                    # Get attributes
                    avg_scan_angle = ds.getncattr('avg_scan_angle_scene') if 'avg_scan_angle_scene' in ds.ncattrs() else np.nan
                    daynight = ds.getncattr('daynight') if 'daynight' in ds.ncattrs() else 'Unknown'
                    overpass_period = ds.getncattr('overpass_period') if 'overpass_period' in ds.ncattrs() else 'Unknown'
                    
                    # Original and cropped shapes (stored as attributes)
                    original_shape = ds.getncattr('original_shape') if 'original_shape' in ds.ncattrs() else [0, 0]
                    cropped_shape = ds.getncattr('cropped_shape') if 'cropped_shape' in ds.ncattrs() else [0, 0]
                    
                    # Get dimension sizes
                    n_scans = len(ds.dimensions['scan']) if 'scan' in ds.dimensions else 0
                    n_pixels = len(ds.dimensions['pixel']) if 'pixel' in ds.dimensions else 0
                    
                    # Get scan coordinate range for boundary analysis
                    scan_var = ds.variables['scan'][:]
                    scan_min = int(scan_var[0])
                    scan_max = int(scan_var[-1])
    
                  
                swath_metadata.append({
                    'filepath': filepath,
                    'filename': filename,
                    'satellite': satellite,
                    'timestamp': timestamp,
                    'date': timestamp.date(),
                    'time': timestamp.time(),
                    'avg_scan_angle': avg_scan_angle,
                    'daynight': daynight,
                    'overpass_period': overpass_period,
                    'n_scans': n_scans,
                    'n_pixels': n_pixels,
                    'original_scans': int(original_shape[0]),
                    'original_pixels': int(original_shape[1]),
                    'scan_min': scan_min,
                    'scan_max': scan_max,
                })
                
            except Exception as e:
                # Log the problematic file
                bad_files.append({
                    'filepath': filepath,
                    'filename': filename,
                    'satellite': satellite,
                    'timestamp': timestamp,
                    'error_type': type(e).__name__,
                    'error_message': str(e)
                })
                print(f"  ERROR reading {filename}: {type(e).__name__} - {str(e)}")
    
    # Convert to DataFrame for easy filtering/analysis
    swath_df = pd.DataFrame(swath_metadata)
    
    print(f"\nSuccessfully read {len(swath_df)} swath files")
    
    if len(bad_files) > 0:
        print(f"\n{'='*70}")
        print(f"WARNING: {len(bad_files)} files could not be read!")
        print(f"{'='*70}")
        bad_files_df = pd.DataFrame(bad_files)
        print(bad_files_df[['filename', 'error_type']])
        print("\nYou may want to delete or re-process these files:")
        for bf in bad_files:
            print(f"  {bf['filepath']}")

    return swath_df


# define first functon

def create_reference_grid(region='conus', resolution=500, 
                          epsg=None, geo_bbox=None, region_name=None):
    """
    CHECKED
    
    Create a single reference grid for all VIIRS fire analysis.
    This grid is created once and reused for all fires.
    
    The grid is defined by an affine transform anchored at the top-left
    corner of the top-left cell, following the GeoTIFF/GDAL convention.
    Cell-center coordinate arrays (for xarray/CF) are derived from this
    transform to ensure consistency.

    NOTE: It's recommended to use equal area projections, which produce
    the most precise aggregation results. 
    
    Parameters
    ----------
    region : str
        Grid region. Options with resulting epsg and bbox parameters:
            - 'conus': EPSG:5070 (Albers Equal Area), [-130, 20, -60, 52]
            - 'global': EPSG:6933 (EASE-Grid 2.0 CEA), [-180, -60, 180, 75]
            - 'custom': User-defined. Requires epsg, geo_bbox, and region_name.
    resolution : int
        Grid cell size in meters
    epsg : int, optional
        EPSG code for the projected CRS. Required when region='custom'.
        Must be a projected CRS with meter units.
    geo_bbox : list, optional
        Geographic bounding box [west, south, east, north] in decimal
        degrees (EPSG:4326). Required when region='custom'.
    region_name : str, optional
        Short descriptive name for the custom region (e.g., 'new_mexico',
        'australia').
        Required when region='custom'.
    
    Returns
    -------
    grid_meta : dict
        Grid metadata including:
            - 'transform': rasterio Affine transform
            - 'crs': pyproj.CRS object
            - 'x_coords', 'y_coords': 1D cell-center arrays (meters)
            - 'width', 'height': grid dimensions
            - 'resolution_m': cell size
    """
    
    # ================================================================
    # STEP 1: Set defaults based on region
    # ================================================================
    
    region_defaults = {
        'conus': {
            'epsg': 5070,
            'geo_bbox': [-130, 20, -60, 52],
            'region_name': 'conus',
        },
        'global': {
            'epsg': 6933,
            'geo_bbox': [-180, -60, 180, 75],
            'region_name': 'global',
        },
    }
    
    if region in region_defaults:
        # Named region — use defaults, ignore custom params
        defaults = region_defaults[region]
        epsg_code = defaults['epsg']
        bbox = defaults['geo_bbox']
        name = defaults['region_name']
        
    elif region == 'custom':
        # Validate that all required params are provided
        missing = []
        if epsg is None:
            missing.append('epsg')
        if geo_bbox is None:
            missing.append('geo_bbox')
        if region_name is None:
            missing.append('region_name')
        
        if missing:
            raise ValueError(
                f"region='custom' requires: {', '.join(missing)}. "
                f"Example: create_reference_grid(region='custom', "
                f"epsg=32613, geo_bbox=[-107, 35, -105, 37], "
                f"region_name='new_mexico')"
            )
        
        epsg_code = epsg
        bbox = geo_bbox
        name = region_name
        
    else:
        raise ValueError(
            f"region must be 'conus', 'global', or 'custom'. Got: '{region}'"
        )
    
    # ================================================================
    # STEP 2: Build and validate CRS
    # ================================================================

    # The crs needs to be both projected and in meters
    # The below lines check for that
    
    proj_crs = pyproj.CRS.from_epsg(epsg_code)
    
    if proj_crs.is_geographic:
        raise ValueError(
            f"EPSG:{epsg_code} is a geographic CRS. "
            f"A projected CRS with meter units is required."
        )
    
    axis_info = proj_crs.axis_info
    units = {ax.unit_name for ax in axis_info}
    if 'metre' not in units:
        raise ValueError(
            f"EPSG:{epsg_code} uses units {units}. "
            f"A projected CRS with metre units is required "
            f"(resolution is specified in meters)."
        )
    
    # ================================================================
    # STEP 3: Transform bbox to projected crs
    # ================================================================

    # define pyproj transformer
    transformer = pyproj.Transformer.from_crs(
        4326, proj_crs, always_xy=True
    )

    # grab bbox corners and transform
    west, south, east, north = bbox
    proj_west, proj_south, proj_east, proj_north = transformer.transform_bounds(
        west, south, east, north,
        densify_pts=21
    )
    
    # Find the closest multiple of our specified resolution either above or below the projected bounds
    # aka "snapping" to the future grid
    aligned_west  = np.floor(proj_west / resolution) * resolution
    aligned_east  = np.ceil(proj_east / resolution) * resolution
    aligned_south = np.floor(proj_south / resolution) * resolution
    aligned_north = np.ceil(proj_north / resolution) * resolution
    
    width = int(round((aligned_east - aligned_west) / resolution))
    height = int(round((aligned_north - aligned_south) / resolution))

    # ================================================================
    # STEP 4: Derive cell-center coordinates from the transform
    # ================================================================
    
    # CF/xarray convention: coordinates represent cell centers.
    # Center of pixel (col, row) = edge + (index + 0.5) * pixel_size
    x_coords = aligned_west + (np.arange(width) + 0.5) * resolution
    y_coords = aligned_north - (np.arange(height) + 0.5) * resolution
    
    # ================================================================
    # STEP 5: Define the authoritative affine transform
    # ================================================================

    grid_transform = rasterio.transform.from_origin(
        aligned_west,    # west edge
        aligned_north,   # north edge
        resolution,      # pixel width  (positive, increasing eastward)
        resolution       # pixel height (positive, from_origin handles the sign)
    )
    
    # ================================================================
    # STEP 6: Build metadata
    # ================================================================
    
    grid_meta = {
        # Spatial registration (authoritative)
        'transform': grid_transform,
        'crs': proj_crs,
        'crs_epsg': epsg_code,
        'crs_wkt': proj_crs.to_wkt(),
        
        # Grid dimensions
        'width': width,
        'height': height,
        'resolution_m': resolution,
        'total_cells': width * height,
        
        # Coordinate arrays (cell centers, derived from transform)
        'x_coords': x_coords,
        'y_coords': y_coords,
        
        # Extents (cell edges, projected and geographic)
        'extent_proj_m': [aligned_west, aligned_south, aligned_east, aligned_north],
        'extent_geo_deg': list(bbox),
        
        # Region info
        'region': region,
        'region_name': name,
    }
    
    return grid_meta
    
# define the function for pulling out subset of larger reference grid
def create_fire_grid_extent(bbox, grid_meta, pad=10, verbose=False):
    """
    CHECKED
    
    Define a fixed spatial subset of the reference grid for a fire event.
    
    All swaths for this fire will be mapped onto these exact grid cells,
    ensuring they can be stacked along a time dimension into a datacube.
    Computed once and reused for all swaths.
    
    Parameters
    ----------
    bbox : list
        Fire bounding box [lon_min, lat_min, lon_max, lat_max] in EPSG:4326
    grid_meta : dict
        Grid metadata from create_reference_grid. Must include 
        'transform', 'crs', 'resolution_m', 'width', 'height'.
    pad : int
        Extra grid cells around the bbox. Accounts for swath pixels
        that extend slightly beyond the nominal fire perimeter.
    verbose: bool
        Optional printing.
    
    Returns
    -------
    fire_extent : dict
        Fixed grid extent with:
            transform : rasterio.Affine
                Subset transform (authoritative, from parent grid)
            x_coords, y_coords : np.ndarray
                1D cell-center coordinate arrays (projected, meters)
            lon, lat : np.ndarray
                2D geographic coordinate arrays (for xarray/plotting)
            row_min, row_max, col_min, col_max : int
                Index bounds on the full reference grid (inclusive)
            n_rows, n_cols : int
                Spatial dimensions
    grid_gdf : gpd.GeoDataFrame
        Reference grid cell polygons for the fire extent.
        Contains grid_row, grid_col, grid_id, and geometry.
        CRS matches grid_meta['crs'].
    """
    
    resolution = grid_meta['resolution_m']
    grid_transform = grid_meta['transform']
    
    # Transform fire bbox to grid projection
    transformer = pyproj.Transformer.from_crs(
        'EPSG:4326', grid_meta['crs'], always_xy=True
    )
    proj_west, proj_south, proj_east, proj_north = transformer.transform_bounds(
        bbox[0], bbox[1], bbox[2], bbox[3],
        densify_pts=21
    )
    
    # Convert projected bounds to grid row/col indices
    col_min = int(np.floor((proj_west - grid_transform.c) / grid_transform.a)) - pad
    col_max = int(np.ceil((proj_east - grid_transform.c) / grid_transform.a)) + pad
    row_min = int(np.floor((grid_transform.f - proj_north) / (-grid_transform.e))) - pad # NOTE: -grid_transform.e (it's naturally negative)
    row_max = int(np.ceil((grid_transform.f - proj_south) / (-grid_transform.e))) + pad
    
    # Clamp to grid bounds (inclusive)
    col_min = max(0, col_min)
    col_max = min(grid_meta['width'] - 1, col_max)
    row_min = max(0, row_min)
    row_max = min(grid_meta['height'] - 1, row_max)
    
    n_rows = row_max - row_min + 1
    n_cols = col_max - col_min + 1
    
    # Derive subset transform from parent via windowing
    window = Window(
        col_off=col_min,
        row_off=row_min,
        width=n_cols,
        height=n_rows
    )
    subset_transform = rasterio.windows.transform(window, grid_transform)
    
    # Derive cell-center coordinates from the subset transform
    x_coords = subset_transform.c + (np.arange(n_cols) + 0.5) * resolution
    y_coords = subset_transform.f - (np.arange(n_rows) + 0.5) * resolution
    
    # Pre-compute 2D lat/lon arrays (done once, reused for all swaths)
    transformer_to_geo = pyproj.Transformer.from_crs(
        grid_meta['crs'], 'EPSG:4326', always_xy=True
    )
    xx, yy = np.meshgrid(x_coords, y_coords)
    lons, lats = transformer_to_geo.transform(xx, yy)

    fire_extent = {
        'transform': subset_transform,
        'x_coords': x_coords,
        'y_coords': y_coords,
        'lon': lons,
        'lat': lats,
        'row_min': row_min,
        'row_max': row_max,
        'col_min': col_min,
        'col_max': col_max,
        'n_rows': n_rows,
        'n_cols': n_cols,
    }
    
    
    # ================================================================
    # BUILD GRID CELL POLYGONS
    # ================================================================
    
    grid_rows_arr = np.arange(row_min, row_max + 1)
    grid_cols_arr = np.arange(col_min, col_max + 1)
    
    grid_rr, grid_cc = np.meshgrid(grid_rows_arr, grid_cols_arr, indexing='ij')
    grid_rr = grid_rr.ravel()
    grid_cc = grid_cc.ravel()
    
    cell_x_min = grid_transform.c + grid_cc * resolution
    cell_x_max = cell_x_min + resolution
    cell_y_max = grid_transform.f - grid_rr * resolution
    cell_y_min = cell_y_max - resolution
    
    grid_coords = np.stack([
        np.column_stack([cell_x_min, cell_y_min]),  # SW
        np.column_stack([cell_x_max, cell_y_min]),  # SE
        np.column_stack([cell_x_max, cell_y_max]),  # NE
        np.column_stack([cell_x_min, cell_y_max]),  # NW
        np.column_stack([cell_x_min, cell_y_min]),  # close ring
    ], axis=1)
    
    grid_polygons = shapely.polygons(grid_coords)
    
    grid_gdf = gpd.GeoDataFrame(
        {
            'grid_row': grid_rr,
            'grid_col': grid_cc,
            'grid_id': [f"r{r:04d}_c{c:04d}" for r, c in zip(grid_rr, grid_cc)],
        },
        geometry=grid_polygons,
        crs=grid_meta['crs']
    )

    if verbose:
        print(f"Fire grid extent:")
        print(f"  Shape: {n_rows} rows × {n_cols} cols ({n_rows * n_cols:,} cells)")
        print(f"  Grid rows: [{row_min}, {row_max}]")
        print(f"  Grid cols: [{col_min}, {col_max}]")
        print(f"  Lon range: [{lons.min():.4f}, {lons.max():.4f}]")
        print(f"  Lat range: [{lats.min():.4f}, {lats.max():.4f}]")
        print(f"  Grid polygons: {len(grid_gdf)}")
    
    return fire_extent, grid_gdf

def create_viirs_pixel_polygons(swath_ds, target_crs, verbose=True):
    """
    CHECKED
    
    Create VIIRS pixel polygons using vectorized corner computation
    and batch polygon creation via Shapely 2.0.
    
    Pixel corners are computed as the centroid of 4 neighboring pixel
    centers meeting at each corner. At VIIRS scan group boundaries
    (every 32 rows), cross-group neighbors are replaced with mirrored
    same-group neighbors to preserve physically accurate pixel geometry.
    
    Parameters
    ----------
    swath_ds : xr.Dataset
        Full swath dataset from Step 1
    target_crs : pyproj.CRS
        Target projection for polygon creation
    verbose : bool
        Print progress messages
    
    Returns
    -------
    swath_gdf : gpd.GeoDataFrame
        Pixel polygons with identifiers and all enriched data variables.
    """
    
    # ================================================================
    # STEP 1: Extract arrays and build valid interior mask
    # ================================================================
    
    lon = swath_ds['longitude'].values
    lat = swath_ds['latitude'].values
    pixel_area_km2 = swath_ds['pixel_area'].values
    scan_coords = swath_ds['scan'].values
    pixel_coords = swath_ds['pixel'].values
    
    n_scans, n_tracks = lon.shape
    
    valid = ~np.isnan(lon) & ~np.isnan(lat) & ~np.isnan(pixel_area_km2)
    
    scan_mesh, _ = np.meshgrid(scan_coords, pixel_coords, indexing='ij')
    scan_group_2d = scan_mesh // 32
    
    # Shift the 2D array "down" by one row: each cell now holds the group of its above neighbor
    above_neighbor_scan_group = np.roll(scan_group_2d, 1, axis=0)
    above_neighbor_scan_group[0, :] = -1   # first row has nothing above it
    
    # Shift the 2D array "up" by one row: each cell now holds the group of its below neighbor
    below_neighbor_scan_group = np.roll(scan_group_2d, -1, axis=0)
    below_neighbor_scan_group[-1, :] = -1  # bottom row has nothing below it
    
    # produce 2D boolean array with respective matches
    same_group_above = scan_group_2d == above_neighbor_scan_group
    same_group_below = scan_group_2d == below_neighbor_scan_group
    
    mask = np.zeros_like(lon, dtype=bool)
    mask[1:-1, 1:-1] = valid[1:-1, 1:-1] # mark out edges on all sides
    
    rows, cols = np.where(mask)
    n_pixels = len(rows)
    
    if verbose:
        print(f"  Swath dimensions: {n_scans} scans × {n_tracks} tracks")
        n_excluded_edge = valid[0, :].sum() + valid[-1, :].sum() + \
                          valid[1:-1, 0].sum() + valid[1:-1, -1].sum()
        n_invalid = (~valid).sum()
        print(f"  Total pixels: {lon.size}")
        print(f"  Invalid (NaN): {n_invalid}")
        print(f"  Edge excluded: {n_excluded_edge}")
        print(f"  Valid interior: {n_pixels}")
    
    if n_pixels == 0:
        if verbose:
            print(f"  WARNING: No valid interior pixels")
        return gpd.GeoDataFrame(
        {'swath_row': [], 'swath_col': [], 'scan': [], 'pixel': [],
         'lon': [], 'lat': [], 'pixel_area_km2_LUT': [],
         'pixel_area_km2_measured': [], 'area_ratio': []},
        geometry=[], crs=target_crs
    )
    
    # ================================================================
    # STEP 2: Project all coordinates to target CRS
    # ================================================================
    
    if verbose:
        print(f"  Projecting coordinates to EPSG:{target_crs.to_epsg()}...")
    
    transformer = pyproj.Transformer.from_crs(
        'EPSG:4326', target_crs, always_xy=True
    )
    x_proj, y_proj = transformer.transform(lon, lat) # same dims as lon, lat
    
    # ================================================================
    # STEP 3: Compute neighbors (then fix scan boundaries)
    # ================================================================
    
    if verbose:
        print(f"  Computing pixel corners for {n_pixels} pixels "
              f"(scan-aware mirroring at scan group boundaries)...")
    
    # Center coordinates
    x_c = x_proj[rows, cols]
    y_c = y_proj[rows, cols]
    
    # find all 8 neighbors
    x_n  = x_proj[rows-1, cols  ]; y_n  = y_proj[rows-1, cols  ]
    x_s  = x_proj[rows+1, cols  ]; y_s  = y_proj[rows+1, cols  ]
    x_e  = x_proj[rows,   cols+1]; y_e  = y_proj[rows,   cols+1]
    x_w  = x_proj[rows,   cols-1]; y_w  = y_proj[rows,   cols-1]
    x_ne = x_proj[rows-1, cols+1]; y_ne = y_proj[rows-1, cols+1]
    x_nw = x_proj[rows-1, cols-1]; y_nw = y_proj[rows-1, cols-1]
    x_se = x_proj[rows+1, cols+1]; y_se = y_proj[rows+1, cols+1]
    x_sw = x_proj[rows+1, cols-1]; y_sw = y_proj[rows+1, cols-1]
    
    # ---- Address scan group boundary crossings ----
    
    # Per-pixel flags
    can_use_above = same_group_above[rows, cols]
    can_use_below = same_group_below[rows, cols]
    
    # Identify places we can mirror
    mirror_above = ~can_use_above & can_use_below  # areas where *above* neighbor is from a different scan BUT same scan below
    mirror_below = ~can_use_below & can_use_above  # areas where *below* neighbor is from a different scan BUT same scan above

    if verbose:
        n_mirror_above = mirror_above.sum()
        n_mirror_below = mirror_below.sum()
        print(f"    Scan boundary pixels mirrored (above): {n_mirror_above}")
        print(f"    Scan boundary pixels mirrored (below): {n_mirror_below}")

    # check for anomalies
    neither_valid = ~can_use_above & ~can_use_below
    n_neither = neither_valid.sum()

    # handle fatal error where a scan has no above or below neighbors from the same group
    # this indicates something is wrong with the swath data
    if n_neither > 0:
        raise ValueError(
            f"{n_neither} pixels have no same-group neighbor above or below. "
            f"This indicates an unexpected scan group configuration "
            f"(e.g., single-row scan group in cropped swath). "
            f"Scan range: [{scan_coords[0]}, {scan_coords[-1]}], "
            f"n_scans: {n_scans}"
        )
    
    # apply above mirror, i.e. take difference between center and south coords, add diff back to center
    x_n[mirror_above] = x_c[mirror_above] + (x_c[mirror_above] - x_s[mirror_above])
    y_n[mirror_above] = y_c[mirror_above] + (y_c[mirror_above] - y_s[mirror_above])
    
    # apply below mirror: reflect below across center
    x_s[mirror_below] = x_c[mirror_below] - (x_n[mirror_below] - x_c[mirror_below])
    y_s[mirror_below] = y_c[mirror_below] - (y_n[mirror_below] - y_c[mirror_below])
    
    # Fix above diagonals: for ne and nw, take difference between e and se, w and sw
    x_ne[mirror_above] = x_e[mirror_above] + (x_e[mirror_above] - x_se[mirror_above])
    y_ne[mirror_above] = y_e[mirror_above] + (y_e[mirror_above] - y_se[mirror_above])
    x_nw[mirror_above] = x_w[mirror_above] + (x_w[mirror_above] - x_sw[mirror_above])
    y_nw[mirror_above] = y_w[mirror_above] + (y_w[mirror_above] - y_sw[mirror_above])
    
    # similar logic for below diagonals
    x_se[mirror_below] = x_e[mirror_below] - (x_ne[mirror_below] - x_e[mirror_below])
    y_se[mirror_below] = y_e[mirror_below] - (y_ne[mirror_below] - y_e[mirror_below])
    x_sw[mirror_below] = x_w[mirror_below] - (x_nw[mirror_below] - x_w[mirror_below])
    y_sw[mirror_below] = y_w[mirror_below] - (y_nw[mirror_below] - y_w[mirror_below])

    # ================================================================
    # STEP 3b: Safety check: filter out pixels with invalid neighbors
    # ================================================================
    
    neighbors_valid = (
        np.isfinite(x_n)  & np.isfinite(y_n)  &
        np.isfinite(x_s)  & np.isfinite(y_s)  &
        np.isfinite(x_e)  & np.isfinite(y_e)  &
        np.isfinite(x_w)  & np.isfinite(y_w)  &
        np.isfinite(x_ne) & np.isfinite(y_ne) &
        np.isfinite(x_nw) & np.isfinite(y_nw) &
        np.isfinite(x_se) & np.isfinite(y_se) &
        np.isfinite(x_sw) & np.isfinite(y_sw)
    )
    
    n_invalid_neighbors = (~neighbors_valid).sum()
    
    if verbose and n_invalid_neighbors > 0:
        print(f"  Removed {n_invalid_neighbors} pixels with invalid neighbors")
    
    rows = rows[neighbors_valid]
    cols = cols[neighbors_valid]
    n_pixels = len(rows)
    
    x_c = x_c[neighbors_valid]; y_c = y_c[neighbors_valid]
    
    x_n  = x_n[neighbors_valid];  y_n  = y_n[neighbors_valid]
    x_s  = x_s[neighbors_valid];  y_s  = y_s[neighbors_valid]
    x_e  = x_e[neighbors_valid];  y_e  = y_e[neighbors_valid]
    x_w  = x_w[neighbors_valid];  y_w  = y_w[neighbors_valid]
    x_ne = x_ne[neighbors_valid]; y_ne = y_ne[neighbors_valid]
    x_nw = x_nw[neighbors_valid]; y_nw = y_nw[neighbors_valid]
    x_se = x_se[neighbors_valid]; y_se = y_se[neighbors_valid]
    x_sw = x_sw[neighbors_valid]; y_sw = y_sw[neighbors_valid]
    
    # ================================================================
    # STEP 4: Compute corner coordinates
    # ================================================================
    
    # Each corner = mean of 4 pixel centers meeting at that corner
    ne_x = (x_c + x_n + x_e + x_ne) / 4; ne_y = (y_c + y_n + y_e + y_ne) / 4
    nw_x = (x_c + x_n + x_w + x_nw) / 4; nw_y = (y_c + y_n + y_w + y_nw) / 4
    se_x = (x_c + x_s + x_e + x_se) / 4; se_y = (y_c + y_s + y_e + y_se) / 4
    sw_x = (x_c + x_s + x_w + x_sw) / 4; sw_y = (y_c + y_s + y_w + y_sw) / 4
    
    # ================================================================
    # STEP 5: Batch-create polygons
    # ================================================================
    
    if verbose:
        print(f"  Building {n_pixels} polygons...")
    
    coords = np.stack([
        np.column_stack([ne_x, ne_y]),
        np.column_stack([se_x, se_y]),
        np.column_stack([sw_x, sw_y]),
        np.column_stack([nw_x, nw_y]),
        np.column_stack([ne_x, ne_y]),
    ], axis=1)
    
    polygons = shapely.polygons(coords)
    
    # ================================================================
    # STEP 6: Validate and compute areas
    # ================================================================
    
    is_valid = shapely.is_valid(polygons)
    is_nonempty = ~shapely.is_empty(polygons)
    geom_areas = shapely.area(polygons)
    has_area = geom_areas > 0
    
    keep = is_valid & is_nonempty & has_area
    
    n_dropped = (~keep).sum()
    if verbose and n_dropped > 0:
        print(f"  Dropped {n_dropped} invalid/empty polygons")
    
    polygons = polygons[keep]
    rows = rows[keep]
    cols = cols[keep]
    geom_areas = geom_areas[keep]
    
    # ================================================================
    # STEP 7: Build GeoDataFrame
    # ================================================================
    
    if verbose:
        print(f"  Assembling GeoDataFrame with {len(polygons)} polygons...")
    
    geom_area_km2 = geom_areas / 1e6
    pix_area_km2 = pixel_area_km2[rows, cols]
    
    swath_gdf = gpd.GeoDataFrame(
        {
            'swath_row': rows,
            'swath_col': cols,
            'scan': scan_coords[rows],
            'pixel': pixel_coords[cols],
            'lon': lon[rows, cols],
            'lat': lat[rows, cols],
            'pixel_area_km2_LUT': pix_area_km2, # from step one look up table
            'pixel_area_km2_measured': geom_area_km2, # computed from geometry
            'area_ratio': geom_area_km2 / pix_area_km2,
        },
        geometry=polygons,
        crs=target_crs
    )
    
    if verbose:
        print(f"  Area ratio — mean: {swath_gdf['area_ratio'].mean():.4f}, "
              f"std: {swath_gdf['area_ratio'].std():.4f}")
    
    # ================================================================
    # STEP 8: Enrich with all data variables from swath_ds
    # ================================================================
    
    if verbose:
        print(f"  Enriching with pixel properties from swath_ds...")
    
    n_enriched = 0
    for var_name in swath_ds.data_vars:
        try:
            var_array = swath_ds[var_name].values
            if var_array.ndim == 2:
                swath_gdf[var_name] = var_array[rows, cols]
                n_enriched += 1
            elif var_array.ndim == 1:
                if var_array.shape[0] == swath_ds.sizes['scan']:
                    swath_gdf[var_name] = var_array[rows]
                    n_enriched += 1
                elif var_array.shape[0] == swath_ds.sizes['pixel']:
                    swath_gdf[var_name] = var_array[cols]
                    n_enriched += 1
        except (IndexError, KeyError, ValueError):
            continue
    
    if verbose:
        print(f"  Enriched with {n_enriched} data variables")
        print(f"  Total columns: {len(swath_gdf.columns)}")
    
    return swath_gdf

# function for grid vectorization and mapping calculations
def map_swath_to_reference_grid(swath_ds, grid_gdf, verbose=True):
    """
    CHECKED
    
    Map swath pixels to reference grid.
    
    Parameters
    ----------
    swath_ds : xr.Dataset
        Swath data from Step 1
    grid_gdf : gpd.GeoDataFrame
        GeoDataFrame of the reference grid subset from create_fire_grid_extent.
        Must contain CRS information. 
    verbose : bool
        Print progress messages
    
    Returns
    -------
    mapping_df : pd.DataFrame
        Mapping table with one row per pixel–grid cell intersection.
        Contains: timestamp, satellite, scan, pixel, grid_row, grid_col,
        grid_id, pixel_overlap_fraction, pixel_intersection_area_km2,
        grid_cell_overlap_fraction
    swath_gdf : gpd.GeoDataFrame
        Enriched pixel data with geometries
    """
    
    # ================================================================
    # STEP 1: Extract metadata from swath
    # ================================================================
    
    # pull metadata
    satellite = str(swath_ds['satellite'].values)
    timestamp = pd.Timestamp(str(swath_ds['timestamp_str'].values))
    
    if verbose:
        print(f"Mapping {satellite} {timestamp} to reference grid...")
    
    # ================================================================
    # STEP 2: Create gdf of all swath pixels
    # ================================================================
    
    target_crs = grid_gdf.crs

    # construct pixels
    swath_gdf = create_viirs_pixel_polygons(swath_ds, target_crs, verbose=verbose)
    
    if len(swath_gdf) == 0:
        if verbose:
            print("  No valid pixels — skipping")
        return pd.DataFrame(), gpd.GeoDataFrame()
    
    if verbose:
        print(f"  Processing {len(swath_gdf)} enriched pixels")
    
    # =====================================================================
    # STEP 3: Spatial join — find which pixels intersect which grid cells
    # =====================================================================
    
    if verbose:
        print(f"  Calculating spatial intersections...")

    # compute spatial join - each row represents a unique pixel ~ grid cell match
    # this is the foundation of our mapping df
    joined = gpd.sjoin(swath_gdf, grid_gdf, how='inner', predicate='intersects')
    
    if len(joined) == 0:
        if verbose:
            print("  No intersections found")
        return pd.DataFrame(), swath_gdf

    if verbose:
        print(f"  Found {len(joined)} pixel–grid intersections")
    
    # ================================================================
    # STEP 4: Calculate overlap fractions
    # ================================================================
    
    if verbose:
        print(f"  Calculating overlap fractions...")
    
    # Extract aligned geometry arrays from the join result
    pixel_geoms = swath_gdf.loc[joined.index, 'geometry'].values # intersecting pixel geoms
    grid_geoms = grid_gdf.loc[joined['index_right'], 'geometry'].values # corresponding grid geoms
    
    # Vectorized intersection and area calculation
    intersections = shapely.intersection(pixel_geoms, grid_geoms)
    pixel_overlap_areas_km2 = shapely.area(intersections) / 1e6 # convert to km2 from m
    pixel_areas_km2 = joined['pixel_area_km2_measured'].values # original pixel areas
    grid_cell_area_km2 = grid_gdf.loc[joined['index_right'], 'geometry'].area.values / 1e6 # compute native grid area
    
    pixel_overlap_fractions = pixel_overlap_areas_km2 / pixel_areas_km2 # compute % of pixel in that grid cell
    grid_cell_overlap_fraction = pixel_overlap_areas_km2 / grid_cell_area_km2 # compute % of pixel overlap area relative to grid cell area
     
    # Filter out zero overlaps (touching edges, numerical noise)
    valid_mask = grid_cell_overlap_fraction > 0
    joined_valid = joined[valid_mask].copy()
    pixel_overlap_fractions_valid = pixel_overlap_fractions[valid_mask]
    
    if len(joined_valid) == 0:
        if verbose:
            print("  No valid overlaps found")
        return pd.DataFrame(), swath_gdf
    
    if verbose:
        n_dropped = (~valid_mask).sum()
        if n_dropped > 0:
            print(f"  Dropped {n_dropped} zero-overlap intersections")
    
    # ================================================================
    # STEP 5: Build lean mapping DataFrame
    # ================================================================
    
    if verbose:
        print(f"  Building mapping DataFrame...")
    
    mapping_df = pd.DataFrame({
        'timestamp': timestamp,
        'satellite': satellite,
        'scan': joined_valid['scan'].values,
        'pixel': joined_valid['pixel'].values,
        'grid_row': joined_valid['grid_row'].values,
        'grid_col': joined_valid['grid_col'].values,
        'grid_id': joined_valid['grid_id'].values,
        'pixel_overlap_fraction': pixel_overlap_fractions_valid, # fraction of pixel in grid cell
        'pixel_intersection_area_km2': pixel_overlap_areas_km2[valid_mask], # total pixel area sitting within grid cell
        'grid_cell_overlap_fraction': grid_cell_overlap_fraction[valid_mask] # fraction of intersecting pixel area and grid cell area
        
    })

    # defensive statement to ensure each row is a unique pixel ~ grid match
    assert mapping_df.duplicated(subset=['scan', 'pixel', 'grid_id']).sum() == 0, \
        "Duplicate (scan, pixel, grid_id) records in mapping"
    
    if verbose:
        n_pixels = mapping_df[['scan', 'pixel']].drop_duplicates().shape[0]
        n_grids = mapping_df['grid_id'].nunique()
        avg_cells_per_pixel = len(mapping_df) / n_pixels if n_pixels > 0 else 0
        print(f"  Mapped {n_pixels} pixels to {n_grids} grid cells")
        print(f"  Total mapping records: {len(mapping_df)}")
        print(f"  Avg grid cells per pixel: {avg_cells_per_pixel:.2f}")

        sums = mapping_df.groupby(['scan', 'pixel'])['pixel_overlap_fraction'].sum()
        n_below_95 = (sums < 0.95).sum()
        print(f"  Pixel overlap fraction sums — mean: {sums.mean():.4f}, "
              f"max: {sums.max():.4f}, "
              f"edge pixels (<0.95): {n_below_95}/{len(sums)}")
    
    return mapping_df, swath_gdf

def clean_swath_mapping(mapping_df, swath_gdf, 
                        remove_bowtie=True,
                        deduplicate_scans=True,
                        scan_gap_threshold=10,
                        verbose=True):
    """

    NOT CURRENTLY IN USE
    
    Clean swath-to-grid mapping by removing bowtie pixels and 
    resolving scan overlaps.
    
    Applied before aggregation. Operates on the mapping_df
    using pixel properties from swath_gdf.
    
    Processing order:
        1. Merge necessary pixel properties from swath_gdf
        2. Remove bowtie pixels (fire_mask == 1)
        3. Deduplicate overlapping scans per grid cell
        4. Return cleaned mapping_df
    
    Parameters
    ----------
    mapping_df : pd.DataFrame
        Mapping from map_swath_to_reference_grid.
        Contains: timestamp, satellite, scan, pixel, grid_row, 
                  grid_col, grid_id, overlap_fraction
    swath_gdf : gpd.GeoDataFrame
        Pixel data with fire_mask, scan_angle, etc.
    remove_bowtie : bool
        Remove pixels with fire_mask == 1 (bowtie-affected)
    deduplicate_scans : bool
        Remove duplicate pixels from overlapping scan lines
    scan_gap_threshold : int
        Minimum gap between scan values to consider them as coming
        from separate (overlapping) scan passes. Within a single 
        pass, consecutive scans differ by 1 (or occasionally 2–3
        if bowtie removal created small gaps). A gap >= this 
        threshold indicates a second pass overlapping the first.
        Default raised to 10 to avoid false triggers from 
        bowtie-induced gaps.
    verbose : bool
        Print cleaning statistics
    
    Returns
    -------
    cleaned_df : pd.DataFrame
        Cleaned lean mapping_df (same schema as input)
    cleaning_report : dict
        Statistics about what was removed and why
    """
    
    if verbose:
        print("=" * 70)
        print("SWATH MAPPING CLEANING")
        print("=" * 70)
    
    report = {
        'initial_mappings': len(mapping_df),
        'initial_pixels': mapping_df[['scan', 'pixel']].drop_duplicates().shape[0],
        'initial_grid_cells': mapping_df['grid_id'].nunique(),
    }
    
    cleaned = mapping_df.copy()
    
    # ================================================================
    # STEP 1: Merge required pixel properties from swath_gdf
    # ================================================================
    
    # Only merge what we need for cleaning
    cleaning_cols = ['scan', 'pixel']
    if remove_bowtie:
        cleaning_cols.append('fire_mask')
    
    cleaning_props = swath_gdf[cleaning_cols].copy() # pull these values from swath_gdf
    cleaned = cleaned.merge(cleaning_props, on=['scan', 'pixel'], how='left') # join with mapping df
    
    # ================================================================
    # STEP 2: Remove bowtie pixels
    # ================================================================
    
    if remove_bowtie and 'fire_mask' in cleaned.columns:
        bowtie_mask = cleaned['fire_mask'] == 1
        n_bowtie = bowtie_mask.sum() # count number of bowtie pixels
        
        cleaned = cleaned[~bowtie_mask].copy() # filter!
        
        report['bowtie_removed'] = int(n_bowtie)
        report['after_bowtie_mappings'] = len(cleaned)
        
        if verbose:
            print(f"\n  BOWTIE REMOVAL:")
            print(f"    Bowtie mappings removed: {n_bowtie}")
            print(f"    Remaining mappings: {len(cleaned)}")
        
        # Drop fire_mask because it's no longer needed
        cleaned = cleaned.drop(columns=['fire_mask'])
        
        report['after_bowtie_grid_cells'] = cleaned['grid_id'].nunique()
    
    # ================================================================
    # STEP 3: Deduplicate overlapping scans
    # ================================================================
    
    if deduplicate_scans:
        if verbose:
            print(f"\n  SCAN OVERLAP DEDUPLICATION:")
            print(f"    Scan gap threshold: {scan_gap_threshold}")
        
        n_before_dedup = len(cleaned)
        
        def identify_overlap_scans(group): # define helper
            """
            Within a grid cell, identify which scan values belong to 
            overlapping (later) passes vs. the primary (first) pass.
            
            Finds the first gap between consecutive scan values that
            exceeds the threshold, and keeps only the scans before it.
            This ensures all subsequent overlapping blocks are removed
            regardless of how many exist.
            
            Returns a boolean mask: True = keep (primary pass)
            """
            # pull out and identify unique scan values
            scans = group['scan'].values
            unique_scans = np.sort(np.unique(scans))
            
            if len(unique_scans) <= 1: # case where there's only one scan present (or no data)
                return pd.Series(True, index=group.index)
            
            scan_diffs = np.diff(unique_scans) # calculate difference between adjacent, sorted unique scan values
            
            # Find the first gap exceeding the threshold
            large_gaps = np.where(scan_diffs >= scan_gap_threshold)[0] # np.where returns a tuple, so we need to parse out the result
            
            if len(large_gaps) == 0: # if everything is below the threshold, we're all set and no de-duplication needed
                return pd.Series(True, index=group.index)
            
            # Otherwise, keep everything up to and including the scan before the first large gap
            last_scan_before_gap = unique_scans[large_gaps[0]]
            
            return group['scan'] <= last_scan_before_gap # scans equal to or below this value are considered True
        
        keep_mask = cleaned.groupby('grid_id', group_keys=False).apply(
            identify_overlap_scans, include_groups=False
        )
        
        cleaned = cleaned.loc[keep_mask].copy() # filter out scans above threshold
        
        n_dedup_removed = n_before_dedup - len(cleaned)
        report['dedup_removed'] = int(n_dedup_removed)
        report['after_dedup_mappings'] = len(cleaned)
        
        if verbose:
            print(f"    Overlap mappings removed: {n_dedup_removed}")
            print(f"    Remaining mappings: {len(cleaned)}")

        # Verify no remaining gaps exceed threshold
        def max_scan_gap(group):
            unique_scans = np.sort(group['scan'].unique())
            if len(unique_scans) <= 1:
                return 0
            return np.diff(unique_scans).max()
        
        remaining_gaps = cleaned.groupby('grid_id', group_keys=False).apply(
            max_scan_gap, include_groups=False
        )
        
        assert (remaining_gaps < scan_gap_threshold).all(), \
            f"Post-dedup scan gaps exceed threshold: " \
            f"{remaining_gaps[remaining_gaps >= scan_gap_threshold].to_dict()}"
            
    # ================================================================
    # FINAL SUMMARY
    # ================================================================
    
    report['final_mappings'] = len(cleaned)
    report['final_pixels'] = cleaned[['scan', 'pixel']].drop_duplicates().shape[0]
    report['final_grid_cells'] = cleaned['grid_id'].nunique()
    report['total_removed'] = report['initial_mappings'] - report['final_mappings']
    report['pct_removed'] = (
        report['total_removed'] / report['initial_mappings'] * 100 
        if report['initial_mappings'] > 0 else 0
    )
    
    if verbose:
        print(f"\n  {'=' * 50}")
        print(f"  CLEANING SUMMARY:")
        print(f"    Mappings: {report['initial_mappings']} → {report['final_mappings']} "
              f"({report['total_removed']} removed, {report['pct_removed']:.1f}%)")
        print(f"    Unique pixels: {report['initial_pixels']} → {report['final_pixels']}")
        print(f"    Grid cells: {report['initial_grid_cells']} → {report['final_grid_cells']}")
    
    return cleaned, report

def aggregate_pixels_to_grid(mapping_df, swath_gdf, verbose=True):
    """
    CHECKED
    
    Aggregate pixel properties to grid cells.
    
    Metrics produced per grid cell:
        BT bands (I4, I5, delta): count, min, max, mean, area-weighted mean
        fire_mask: max, mode, area-weighted majority, fire pixel count
        candidate_confidence: max, area-weighted majority, count
        scan_angle: area-weighted mean
        NaN coverage: fraction of total intersecting area from NaN pixels
                      (computed separately per BT variable)
        Saturation: fraction of total intersecting area from saturated pixels,
                    plus grid cell coverage fraction from saturated pixels
        Pixel counts and total intersecting area 
    
    All area-weighted metrics use pixel_intersection_area_km2 as weights,
    representing the physical area of each pixel-grid cell intersection.
    
    NaN handling: NaN values are excluded from all aggregation metrics.
    A separate nan_area_fraction variable is computed for each BT variable,
    representing the fraction of the grid cell's total intersecting area
    contributed by pixels with NaN values for that variable.
    
    Saturation handling: Saturated pixels have real BT values (e.g.,
    I4 caps at 367K and sometimes experiences folding) and are included 
    in all BT statistics. Two separate metrics track saturation:
        - saturation_area_fraction: fraction of total intersecting area
          from saturated pixels (bounded [0, 1])
        - saturation_grid_coverage: fraction of the grid cell's physical
          area covered by saturated pixel intersections (can exceed 1.0
          when overlapping scans contribute multiple saturated pixels)
    
    Note: Bowtie pixels (fire_mask == 1) are expected to be removed
    by clean_swath_mapping before this function is called.
    
    Parameters
    ----------
    mapping_df : pd.DataFrame
        Cleaned mapping info from clean_swath_mapping.
        Contains: timestamp, satellite, scan, pixel, grid_row, grid_col,
                  grid_id, pixel_overlap_fraction, pixel_intersection_area_km2,
                  grid_cell_overlap_fraction
    swath_gdf : gpd.GeoDataFrame
        Enriched pixel data with fire_mask, scan_angle, I4_bt, I5_bt, etc.
    verbose : bool
        Print progress
    
    Returns
    -------
    grid_agg_df : pd.DataFrame
        One row per grid cell with aggregated properties
    """
    
    if len(mapping_df) == 0:
        if verbose:
            print("Empty mapping_df - nothing to aggregate")
        return pd.DataFrame()
    
    # ================================================================
    # STEP 1: Merge pixel properties from swath_gdf with mapping_df
    # ================================================================
    
    if verbose:
        print(f"Aggregating {len(mapping_df)} pixel mappings to grid cells...")
        print(f"  Timestamp: {mapping_df['timestamp'].unique()[0]}")
        print(f"  Satellite: {mapping_df['satellite'].unique()[0]}")
    
    merge_cols = ['scan', 'pixel']
    property_cols = ['I4_bt', 'I5_bt', 'delta_I4_I5', 'fire_mask', 
                     'scan_angle', 'candidate_confidence', 'qa_saturation']
    available_props = [c for c in property_cols if c in swath_gdf.columns]
    
    swath_properties = swath_gdf[merge_cols + available_props] # pull out subset of all needed cols

    # merge with the mapping_df so each pixel~grid match also contains the relevant pixel properties 
    mapping_full = mapping_df.merge(swath_properties, on=merge_cols, how='left')
    
    if verbose:
        print(f"  Merged properties: {available_props}")
    
    grouped = mapping_full.groupby('grid_id', observed=True)

    # Total area overlap of all pixels per grid cell
    total_intersecting_areas = grouped['pixel_intersection_area_km2'].sum()
        
    # ================================================================
    # STEP 2: BT band statistics — count, min, max, mean
    # ================================================================
    
    if verbose:
        print(f"  Computing BT band statistics...")
    
    bt_vars = ['I4_bt', 'I5_bt', 'delta_I4_I5']
    bt_available = [v for v in bt_vars if v in mapping_full.columns] # defensive check that bt_vars are present
    
    bt_agg = {} 
    for bt_var in bt_available: # define aggregation dict
        bt_agg[bt_var] = ['count', 'min', 'max', 'mean']
    
    if len(bt_agg) == 0: # if the BT variables are missing, there's a problem - report it
        raise ValueError(
            f"No BT variables found in mapping_full. "
            f"Expected at least one of {bt_vars}. "
            f"Available columns: {list(mapping_full.columns)}"
        )
        
    grid_agg_df = grouped.agg(bt_agg) # perform agg for each grid_id
    grid_agg_df.columns = ['_'.join(col) for col in grid_agg_df.columns] # clean columns
    grid_agg_df = grid_agg_df.reset_index()
    
    # ================================================================
    # STEP 3: Area-weighted BT means + NaN coverage fractions
    # ================================================================
    
    if verbose:
        print(f"  Computing area-weighted BT means and NaN fractions...")
    
    for bt_var in bt_available: # loop through eact bt var again
        
        valid = mapping_full[bt_var].notna() # mask any nans for that bt var (i4, i5, delta)
        
        # --- Area-weighted mean (NaN-excluded) ---
        valid_subset = mapping_full[valid] # apply mask
        
        # numerator - multiply each pixel's bt var by its intersection area (weight)
        # then sum these values grouped by each grid_id
        weighted_sum = valid_subset.assign(
            _w=valid_subset[bt_var] * valid_subset['pixel_intersection_area_km2']
        ).groupby('grid_id', observed=True)['_w'].sum()
        
        # denominator - sum all the weights present in a given grid cell
        sum_of_intersecting_areas = valid_subset.groupby(
            'grid_id', observed=True
        )['pixel_intersection_area_km2'].sum()

        # finish equation by computing the fraction
        area_weighted_mean = (weighted_sum / sum_of_intersecting_areas).reset_index(
            name=f'{bt_var}_area_weighted_mean'
        )

        # add to main df
        grid_agg_df = grid_agg_df.merge(area_weighted_mean, on='grid_id', how='left')
        
        # --- NaN coverage fraction ---

        # rationale for this: I4 fill values are originally -999; we replaced them with nan
        # we want to capture if any non-valid BT measurements are present in a given grid cell
        # note: these are separate from saturated or folded pixels - those still return a BT val
        
        # isolate nan pixels and compute their total overlap
        nan_intersection_area = mapping_full[~valid].groupby( 
            'grid_id', observed=True
        )['pixel_intersection_area_km2'].sum()

        # ask: "what fraction of the total contributing area comes from pixels with NaN values?"
        nan_area_frac = (nan_intersection_area / total_intersecting_areas).reset_index(
            name=f'{bt_var}_nan_area_fraction'
        )

        # merge back in
        grid_agg_df = grid_agg_df.merge(nan_area_frac, on='grid_id', how='left')
        
        # for grid cells with no nan coverage, this defaults to 0
        grid_agg_df[f'{bt_var}_nan_area_fraction'] = (
            grid_agg_df[f'{bt_var}_nan_area_fraction'].fillna(0.0)
        )
    
    if verbose:
        for bt_var in bt_available:
            col = f'{bt_var}_nan_area_fraction'
            if col in grid_agg_df.columns:
                n_any_nan = (grid_agg_df[col] > 0).sum()
                max_nan = grid_agg_df[col].max()
                print(f"    {bt_var}: {n_any_nan} cells with NaN "
                      f"(max fraction: {max_nan:.3f})")
    
    # ================================================================
    # STEP 3b: Saturation coverage fraction
    # ================================================================
    
    if 'qa_saturation' in mapping_full.columns:
        if verbose:
            print(f"  Computing saturation coverage fractions...")
        
        saturated = mapping_full[mapping_full['qa_saturation'] == 1] # find any saturation pixels
        
        # same idea as above for nans - compute fraction of intersecting 
        # area from saturated pixels 
        if len(saturated) > 0:
            sat_intersection_area = saturated.groupby(
                'grid_id', observed=True
            )['pixel_intersection_area_km2'].sum()
            
            sat_frac = (sat_intersection_area / total_intersecting_areas).reset_index(
                name='saturation_area_fraction'
            )
            # merge it back
            grid_agg_df = grid_agg_df.merge(sat_frac, on='grid_id', how='left')

            # last saturated metric, which needs explanation: 
            # calculate total saturated pixel area relative to that grid cell area
            # 'grid_cell_overlap_fraction' is the pixel intersection area / grid cell area
            # we sum these fractions across all saturated pixels to get the total fractional area overlap
            # this is the same as if we waited to sum all the intersecting areas, and then divide once by the grid cell area
            # example: (0.12km2 / 0.25km2) + (0.13km2 / 0.25km2) == (0.12km2 + 0.13km2) / 0.25km2
            # individual intersection area ratios are pre-calculated (e.g., [0.12km2 / 0.25km2], [0.13km2 / 0.25km2]), 
            # so we're just summing them below. 
            
            sat_grid_coverage = saturated.groupby('grid_id', observed=True)['grid_cell_overlap_fraction'].sum().reset_index(name='saturation_grid_coverage')
            
            # ^^^ NOTE: this value can (rarely) exceed 1.0 when overlapping scans contribute 
            # multiple saturated pixels covering the same portion of the grid cell
            
            # merge back into grid
            grid_agg_df = grid_agg_df.merge(sat_grid_coverage, on='grid_id', how='left')
        
        else:
            grid_agg_df['saturation_area_fraction'] = 0.0
            grid_agg_df['saturation_grid_coverage'] = 0.0
        
        grid_agg_df['saturation_area_fraction'] = grid_agg_df['saturation_area_fraction'].fillna(0.0)
        grid_agg_df['saturation_grid_coverage'] = grid_agg_df['saturation_grid_coverage'].fillna(0.0)
    
        
        if verbose:
            n_any_sat = (grid_agg_df['saturation_area_fraction'] > 0).sum()
            max_sat = grid_agg_df['saturation_area_fraction'].max()
            print(f"    {n_any_sat} cells with saturation "
                  f"(max area fraction: {max_sat:.3f})")
            max_cov = grid_agg_df['saturation_grid_coverage'].max()
            print(f"    Max grid coverage from saturation: {max_cov:.3f}")

    # ================================================================
    # STEP 4: Fire mask — max, mode, area-weighted majority
    # ================================================================
    
    if 'fire_mask' not in mapping_full.columns:
        raise ValueError(
            f"'fire_mask' not found in mapping_full. "
            f"This is required for fire detection metrics. "
            f"Available columns: {list(mapping_full.columns)}"
        )
        
    # Max fire mask per grid cell
    fm_max = grouped['fire_mask'].max().reset_index(name='fire_mask_max')
    grid_agg_df = grid_agg_df.merge(fm_max, on='grid_id', how='left')
    
    # --- Mode ---
    # Count occurrences of each fire_mask value per grid cell,
    # then pick the value with the highest count.
    # sort_index ensures ties go to the lowest value
    fm_counts = (
        mapping_full
        .groupby(['grid_id', 'fire_mask'], observed=True)
        .size()
        .reset_index(name='_count')
    )
    
    # For each grid_id, keep the fire_mask with the highest count.
    # for ties, the lower fire mask value is returned
    fm_mode = (
        fm_counts
        .sort_values(['grid_id', '_count', 'fire_mask'], 
                     ascending=[True, True, False]) # False leads to lower values winning ties
        .drop_duplicates(subset='grid_id', keep='last')
        .rename(columns={'fire_mask': 'fire_mask_mode'})
    )
    grid_agg_df = grid_agg_df.merge(fm_mode[['grid_id', 'fire_mask_mode']], on='grid_id', how='left')
    
    # --- Area-weighted majority ---
    # Sum intersection area per (grid_id, fire_mask), then pick the
    # fire_mask with the highest total area per grid_id.
    fm_weights = (
        mapping_full
        .groupby(['grid_id', 'fire_mask'], observed=True)['pixel_intersection_area_km2']
        .sum()
        .reset_index(name='_weight')
    )
    
    # Keep the fire_mask with highest weight per grid_id
    fm_majority = (
        fm_weights
        .sort_values(['grid_id', '_weight'], ascending=[True, True])
        .drop_duplicates(subset='grid_id', keep='last')
        .rename(columns={'fire_mask': 'fire_mask_area_weighted_majority'})
    )
    grid_agg_df = grid_agg_df.merge(fm_majority[['grid_id', 'fire_mask_area_weighted_majority']], 
                                    on='grid_id', how='left')
    
    # Fire pixel count (fire_mask >= 7)
    fire_count = mapping_full[mapping_full['fire_mask'] >= 7].groupby(
        'grid_id', observed=True
    ).size().reset_index(name='n_fire_pixels')
    grid_agg_df = grid_agg_df.merge(fire_count, on='grid_id', how='left')
    grid_agg_df['n_fire_pixels'] = (
        grid_agg_df['n_fire_pixels'].fillna(0).astype(int)
    )
    
    # ================================================================
    # STEP 4b: Candidate confidence — max, area-weighted majority
    # ================================================================
    
    if 'candidate_confidence' in mapping_full.columns:
        if verbose:
            print(f"  Computing candidate confidence statistics...")
        
        candidates = mapping_full[
            mapping_full['candidate_confidence'].notna()
        ].copy()
        
        if len(candidates) > 0:
            cand_grouped = candidates.groupby('grid_id', observed=True)
            
            # Max confidence per grid cell
            cc_max = cand_grouped['candidate_confidence'].max().reset_index(
                name='candidate_confidence_max'
            )
            grid_agg_df = grid_agg_df.merge(cc_max, on='grid_id', how='left')
            
            # --- Area-weighted majority ---
            # Sum intersection area per (grid_id, candidate_confidence), then pick the
            # candidate_confidence class with the highest total area per grid_id.
            cc_weights = (
                candidates
                .groupby(['grid_id', 'candidate_confidence'], observed=True)
                ['pixel_intersection_area_km2']
                .sum()
                .reset_index(name='_weight')
            )
            
            cc_majority = (
                cc_weights
                .sort_values(['grid_id', '_weight'], ascending=[True, True])
                .drop_duplicates(subset='grid_id', keep='last')
                .rename(columns={
                    'candidate_confidence': 'candidate_confidence_area_weighted_majority'
                })
            )
            grid_agg_df = grid_agg_df.merge(
                cc_majority[['grid_id', 'candidate_confidence_area_weighted_majority']], 
                on='grid_id', how='left'
            )
            
            # Count of candidate pixels per grid cell
            cc_count = cand_grouped.size().reset_index(
                name='n_candidate_pixels'
            )
            grid_agg_df = grid_agg_df.merge(
                cc_count, on='grid_id', how='left'
            )
            
        else:
            grid_agg_df['candidate_confidence_max'] = np.nan
            grid_agg_df['candidate_confidence_area_weighted_majority'] = np.nan
            grid_agg_df['n_candidate_pixels'] = 0
        
        grid_agg_df['n_candidate_pixels'] = (
            grid_agg_df['n_candidate_pixels'].fillna(0).astype(int)
        )
    
    # ================================================================
    # STEP 5: Area-weighted mean scan angle (NaN-safe)
    # ================================================================
    
    if 'scan_angle' in mapping_full.columns:
        if verbose:
            print(f"  Computing area-weighted mean scan angle...")
        
        valid_sa = mapping_full['scan_angle'].notna()
        valid_sa_subset = mapping_full[valid_sa]
        
        weighted_angle_sum = valid_sa_subset.assign(
            _w=valid_sa_subset['scan_angle'] * valid_sa_subset['pixel_intersection_area_km2']
        ).groupby('grid_id', observed=True)['_w'].sum()
        
        valid_sa_weight_sum = valid_sa_subset.groupby(
            'grid_id', observed=True
        )['pixel_intersection_area_km2'].sum()
        
        sa_weighted = (weighted_angle_sum / valid_sa_weight_sum).reset_index(
            name='scan_angle_area_weighted_mean'
        )
        grid_agg_df = grid_agg_df.merge(sa_weighted, on='grid_id', how='left')
    
    # ================================================================
    # STEP 6: Pixel counts, overlap totals, metadata
    # ================================================================
    
    if verbose:
        print(f"  Computing pixel counts and metadata...")
    
    n_pixels = grouped.size().reset_index(name='n_pixels')
    grid_agg_df = grid_agg_df.merge(n_pixels, on='grid_id', how='left')
    
    total_intersecting_areas_df = total_intersecting_areas.reset_index(name='total_intersecting_area_km2')
    grid_agg_df = grid_agg_df.merge(total_intersecting_areas_df, on='grid_id', how='left')
    
    metadata = grouped[['grid_row', 'grid_col', 'timestamp', 'satellite']].agg(
        'first'
    ).reset_index()
    grid_agg_df = grid_agg_df.merge(metadata, on='grid_id', how='left')
    
    # ================================================================
    # SUMMARY
    # ================================================================
    
    if verbose:
        print(f"\n  Aggregated to {len(grid_agg_df)} unique grid cells")
        print(f"  Pixels per cell: mean={grid_agg_df['n_pixels'].mean():.1f}, "
              f"max={grid_agg_df['n_pixels'].max()}")
        
        if 'n_fire_pixels' in grid_agg_df.columns:
            n_fire_cells = (grid_agg_df['n_fire_pixels'] > 0).sum()
            print(f"  Grid cells with fire (mask >= 7): {n_fire_cells}")
        
    return grid_agg_df

def grid_agg_to_xarray(grid_agg_df, fire_extent, grid_meta, swath_ds, test=True, verbose=True):
    """
    Place aggregated grid results into a fixed-extent xarray Dataset
    with a time dimension, ready for Zarr append.
    
    Parameters
    ----------
    grid_agg_df : pd.DataFrame
        Output from aggregate_pixels_to_grid
    fire_extent : dict
        Fixed spatial extent from create_fire_grid_extent
    grid_meta : dict
        Grid metadata from create_reference_grid
    swath_ds : xr.Dataset
        Original swath dataset — used to extract per-timestep metadata
    test : bool
        Run placement verification tests. Default True.
    verbose : bool
        Print progress messages. Default True.
    
    Returns
    -------
    ds : xr.Dataset
        Dataset with dims (time=1, y, x).
        Per-timestep metadata stored as 1D variables along time.
        CRS written via rioxarray.
    """
    
    if len(grid_agg_df) == 0:
        if verbose:
            print("Empty grid_agg_df — nothing to convert")
        return xr.Dataset()
    
    # ================================================================
    # STEP 1: Unpack fixed extent
    # ================================================================
    
    x_coords = fire_extent['x_coords']
    y_coords = fire_extent['y_coords']
    row_min = fire_extent['row_min']
    col_min = fire_extent['col_min']
    n_rows = fire_extent['n_rows']
    n_cols = fire_extent['n_cols']
    
    # ================================================================
    # STEP 2: Compute local indices and filter to extent
    # ================================================================
    
    local_rows = grid_agg_df['grid_row'].values - row_min
    local_cols = grid_agg_df['grid_col'].values - col_min
    
    # Drop any cells outside the fixed extent
    in_bounds = (
        (local_rows >= 0) & (local_rows < n_rows) &
        (local_cols >= 0) & (local_cols < n_cols)
    )
    
    n_oob = (~in_bounds).sum()
    n_kept = in_bounds.sum()

    if n_oob > 0 and verbose:
        print(f"  WARNING: {n_oob} grid cells fell outside fire extent")
    
    if n_kept == 0 and verbose:
        print(f"  WARNING: ALL grid cells fell outside fire extent!")
    
    local_rows = local_rows[in_bounds]
    local_cols = local_cols[in_bounds]
    grid_agg_valid = grid_agg_df[in_bounds].copy()
    
    # ================================================================
    # STEP 3: Identify data variables to rasterize
    # ================================================================
    
    skip_cols = {
        'grid_id', 'grid_row', 'grid_col',
        'timestamp', 'satellite'
    }
    data_var_names = [c for c in grid_agg_valid.columns if c not in skip_cols]
    
    # ================================================================
    # STEP 4: Fill 2D arrays and expand to (time=1, y, x)
    # ================================================================
    
    arrays = {}
    for var in data_var_names:
        values = grid_agg_valid[var].values
        
        # Choose fill value based on dtype
        # -1 = "no data" for integers
        # NaN = "no data" for floats
        # 0 is reserved for "processed but no hits" (e.g., n_fire_pixels)
        if np.issubdtype(values.dtype, np.integer):
            arr = np.full((n_rows, n_cols), -1, dtype=np.float64)
        else:
            arr = np.full((n_rows, n_cols), np.nan, dtype=np.float64)
        
        arr[local_rows, local_cols] = values # assign values from that col
        
        # Expand: (y, x) → (1, y, x) for time stacking
        arrays[var] = (['time', 'y', 'x'], arr[np.newaxis, :, :])

    if test:
        if verbose:
            print("  [TEST] Verifying grid placement...")
        
        # Verify offset arithmetic
        assert np.array_equal(local_rows + row_min, grid_agg_valid['grid_row'].values), \
            "Row offset arithmetic mismatch"
        assert np.array_equal(local_cols + col_min, grid_agg_valid['grid_col'].values), \
            "Column offset arithmetic mismatch"
        
        # Verify a real data variable round-trips through the array correctly
        test_var = data_var_names[0]
        test_arr = arrays[test_var][1][0]  # unpack (['time','y','x'], arr[newaxis]) → (y, x)
        recovered = test_arr[local_rows, local_cols]
        expected = grid_agg_valid[test_var].values
        
        assert np.allclose(recovered, expected, equal_nan=True), \
            f"Array placement verification failed for '{test_var}'"
        
        if verbose:
            print(f"    Verified {len(local_rows)} cells using '{test_var}'")
    
    # ================================================================
    # STEP 5: Extract per-timestep metadata from swath_ds
    # ================================================================
    
    timestamp = pd.Timestamp(str(swath_ds['timestamp_str'].values))
    satellite = str(swath_ds['satellite'].values)
    daynight = str(swath_ds['daynight'].values)
    overpass_period = str(swath_ds['overpass_period'].values)
    avg_scan_angle = float(swath_ds['avg_scan_angle_scene'].values)
    
    # ================================================================
    # STEP 6: Build Dataset
    # ================================================================
    
    # Pad string variables to fixed width so Zarr dtype is consistent
    # across all appends. Without this, "Day" (<U3) vs "Night" (<U5)
    # causes a dtype mismatch on append.
    max_sat_len = 10     # covers SNPP, NOAA20, NOAA21
    max_dn_len = 5       # covers Day, Night, Both
    max_period_len = 2   # AM, PM
    
    ds = xr.Dataset(
        data_vars={
            **arrays,
            # Per-timestep metadata (1D along time, fixed-width strings)
            'satellite': (['time'], np.array([satellite.ljust(max_sat_len)], dtype=f'U{max_sat_len}')),
            'daynight': (['time'], np.array([daynight.ljust(max_dn_len)], dtype=f'U{max_dn_len}')),
            'overpass_period': (['time'], np.array([overpass_period.ljust(max_period_len)], dtype=f'U{max_period_len}')),
            'avg_scan_angle_scene': (['time'], [avg_scan_angle]),
            'n_populated_cells': (['time'], [len(grid_agg_valid)]),
        },
        coords={
            'time': [timestamp],
            'y': y_coords,
            'x': x_coords,
            'lon': (['y', 'x'], fire_extent['lon']),
            'lat': (['y', 'x'], fire_extent['lat']),
        },
    )
    
    # ================================================================
    # STEP 7: Write CRS and transform via rioxarray
    # ================================================================
    
    ds = ds.rio.write_crs(grid_meta['crs'])
    ds = ds.rio.set_spatial_dims(x_dim='x', y_dim='y')
    ds = ds.rio.write_transform(fire_extent['transform'])
    
    # ================================================================
    # STEP 8: Fire-level attributes
    # ================================================================
    
    ds.attrs = {
        'reference_grid_id': f"EPSG:{grid_meta['crs_epsg']}_{grid_meta['resolution_m']}m",
        'crs_epsg': grid_meta['crs_epsg'],
        'pixel_size_x_m': grid_meta['resolution_m'],
        'pixel_size_y_m': grid_meta['resolution_m'],
        'reference_grid_row_offset': int(row_min),
        'reference_grid_col_offset': int(col_min),
        'n_populated_cells': int(len(grid_agg_valid)),
        'n_total_cells': int(n_rows * n_cols),
    }
    
    return ds

def plot_gridded_swath(ds, swath_ds, grid_meta, save_path=None):
    """
    Plot original swath vs. gridded results.
    
    Layout (2x2):
        Row 1: Original swath I4 BT | Gridded area-weighted mean I4 BT
        Row 2: Gridded I4 BT max    | Gridded fire mask majority + candidates
    
    Parameters
    ----------
    ds : xr.Dataset
        Gridded output from grid_agg_to_xarray
    swath_ds : xr.Dataset
        Original swath dataset
    grid_meta : dict
        Reference grid metadata
    save_path : str, optional
        If provided, save figure to this path and close without displaying
    """
    
    # Extract swath metadata
    sat = str(swath_ds['satellite'].values)
    timestamp = pd.Timestamp(str(swath_ds['timestamp_str'].values))
    bbox = swath_ds.attrs.get('bbox', None)
    
    # Get swath data
    lon = swath_ds['longitude'].values
    lat = swath_ds['latitude'].values
    i4_bt = swath_ds['I4_bt'].values
    
    # Shared BT colormap and range
    cmap_bt = plt.cm.plasma.copy()
    cmap_bt.set_bad(color='white', alpha=1)
    vmin_bt, vmax_bt = 250, 360
    
    # Fire mask colormap (categorical)
    mask_colors = [mpl.colormaps['tab10'](c) for c in [4, 6, 5, 0, 9, 2, 7, 8, 1, 3]]
    cmap_fire = ListedColormap(mask_colors)
    cmap_fire.set_bad(color='white', alpha=1)
    
    # Common extent
    if bbox is not None:
        plot_extent = [bbox[0], bbox[2], bbox[1], bbox[3]]
        
    fig = plt.figure(figsize=(20, 20))
    gs = fig.add_gridspec(2, 2, hspace=0.10, wspace=0.05, top=0.93)
    
    # ---- ROW 1, LEFT: Original swath I4 BT ----
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    
    plot_swath = ax1.pcolormesh(lon, lat, i4_bt,
                                vmin=vmin_bt, vmax=vmax_bt,
                                cmap=cmap_bt,
                                transform=ccrs.PlateCarree())
    
    ax1.set_title(f"Original Swath — I4 BT (3.75 µm)\n"
                  f"{sat} {timestamp.strftime('%Y-%m-%d %H:%M')} UTC\n",
                  fontsize=12, fontweight='bold')
    ax1.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    if bbox is not None:
        ax1.set_extent(plot_extent)
    
    # ---- ROW 1, RIGHT: Gridded I4 BT area-weighted mean ----
    ax2 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    
    ax2.pcolormesh(ds['lon'].values, ds['lat'].values,
                   ds['I4_bt_area_weighted_mean'].values,
                   vmin=vmin_bt, vmax=vmax_bt,
                   cmap=cmap_bt,
                   transform=ccrs.PlateCarree())
    
    ax2.set_title(f"Gridded — I4 BT Area-Weighted Mean\n"
                  f"{grid_meta['resolution_m']}m reference grid\n",
                  fontsize=12, fontweight='bold')
    ax2.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    if bbox is not None:
        ax2.set_extent(plot_extent)
    
    cbar_bt1 = fig.colorbar(plot_swath, ax=[ax1, ax2], orientation='horizontal',
                             pad=0.06, aspect=40, shrink=0.7)
    cbar_bt1.set_label('Brightness Temperature (K)', fontsize=12)
    
    # ---- ROW 2, LEFT: Gridded I4 BT max ----
    ax3 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    
    plot_grid_max = ax3.pcolormesh(ds['lon'].values, ds['lat'].values,
                                    ds['I4_bt_max'].values,
                                    vmin=vmin_bt, vmax=vmax_bt,
                                    cmap=cmap_bt,
                                    transform=ccrs.PlateCarree())
    
    ax3.set_title(f"Gridded — I4 BT Max\n"
                  f"Highest BT per grid cell\n",
                  fontsize=12, fontweight='bold')
    ax3.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    if bbox is not None:
        ax3.set_extent(plot_extent)
    
    cbar_bt2 = fig.colorbar(plot_grid_max, ax=ax3, orientation='horizontal',
                             pad=0.06, aspect=30)
    cbar_bt2.set_label('Max Brightness Temperature (K)', fontsize=11)
    
    # ---- ROW 2, RIGHT: Fire mask majority + candidate overlay ----
    ax4 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    
    fire_mask_gridded = ds['fire_mask_area_weighted_majority'].values.copy()
    fire_mask_gridded = np.where(fire_mask_gridded < 0, np.nan, fire_mask_gridded)
    
    plot_fire = ax4.pcolormesh(ds['lon'].values, ds['lat'].values,
                                fire_mask_gridded,
                                vmin=0, vmax=9,
                                cmap=cmap_fire,
                                transform=ccrs.PlateCarree())
    
    n_cand_cells = 0
    if 'n_candidate_pixels' in ds:
        cand_mask = ds['candidate_confidence_area_weighted_majority'].values == 0
        if cand_mask.any():
            cand_lons = ds['lon'].values[cand_mask]
            cand_lats = ds['lat'].values[cand_mask]
            n_cand_cells = int(cand_mask.sum())
            
            ax4.scatter(cand_lons, cand_lats,
                       c='black', s=2, marker='.',
                       transform=ccrs.PlateCarree(), zorder=10)
    
    ax4.set_title(f"Gridded — Fire Mask (Area-Weighted Majority)\n"
                  f"Dominant class per cell | "
                  f"Black dots = candidates ({n_cand_cells})\n",
                  fontsize=12, fontweight='bold')
    ax4.gridlines(draw_labels=True, linestyle='--', alpha=0.5)
    if bbox is not None:
        ax4.set_extent(plot_extent)
    
    cbar_fire = fig.colorbar(plot_fire, ax=ax4, orientation='horizontal',
                              pad=0.06, aspect=30)
    cbar_fire.set_label('Fire Mask Category', fontsize=11)
    fire_labels = ['0: Not\nproc.', '1: Bow-\ntie', '2: Not\nproc.',
                   '3: Water', '4: Cloud', '5: Clear\nland',
                   '6: Unclass.\nfire', '7: Low\nconf.', '8: Nom.\nconf.',
                   '9: High\nconf.']
    cbar_fire.ax.set_xticks(np.arange(10))
    cbar_fire.ax.set_xticklabels(fire_labels, fontsize=7)
    
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    else:
        plt.show()


# ===================================================================
# ZARR ENCODING (defined once, used for both initial write and flush)
# ===================================================================

def get_zarr_encoding(ds, fire_extent):
    """Helper to build Zarr encoding dict for a batched dataset."""
    compressor = BloscCodec(cname='zstd', clevel=3)
    encoding = {}
    for var in ds.data_vars:
        if ds[var].dims == ('time', 'y', 'x'):
            encoding[var] = {
                'chunks': (1, fire_extent['n_rows'], fire_extent['n_cols']),
                'compressors': compressor, 
            }
        elif ds[var].dims == ('time',):
            encoding[var] = {'chunks': (1,)}
    encoding['time'] = {
        'units': 'minutes since 2000-01-01',
        'dtype': 'int64',
    }
    return encoding


def _write_persistence_vars(ds_with_persistence, zarr_store, suffix_list):
    """Write only the persistence variables into zarr_store, in-place."""
    _var_templates = [
        'persistence_hours_baseline_{s}', 't_fire_start_{s}',
        't_fire_end_baseline_{s}',        'n_am_detection_windows_{s}',
        'n_pm_detection_windows_{s}',     'n_total_detection_windows_{s}',
        'dp_ratio_{s}',                   'n_cloud_detection_windows_{s}',
    ]
    z = zarr.open(zarr_store, mode='r+')
    for suffix in suffix_list:
        for tmpl in _var_templates:
            var_name = tmpl.format(s=suffix)
            arr = ds_with_persistence[var_name].compute().values
            if var_name in z:
                z[var_name][:] = arr
            else:
                z.create_dataset(var_name, data=arr, overwrite=True)
    zarr.consolidate_metadata(zarr_store)


def standardize_swaths(fire_name, bbox, start, end, n_timesteps,
                       grid_region='conus', grid_resolution=375,
                       overwrite=False, make_plots=False, copy_to_s3=False,
                       s3_prefix=None, batch_size=50, grid_pad=10,
                       remove_bowtie=False, deduplicate_scans=False,
                       output_dir='VIIRS-cubed-outputs', remove_local=False,
                       add_persistence=False, persistence_fire_mask_col=None,
                       persistence_suffix=None, persistence_start_threshold=6,
                       persistence_end_threshold=6):

    '''Full workflow for loading and aggregating swath data into a regular grid.'''

    if copy_to_s3 and s3_prefix is None:
        raise ValueError("s3_prefix is required when copy_to_s3=True")
    if remove_local and not copy_to_s3:
        raise ValueError("remove_local=True requires copy_to_s3=True")
    if persistence_fire_mask_col is not None and persistence_suffix is None:
        raise ValueError("persistence_suffix is required when persistence_fire_mask_col is set")

    base_output_dir = os.path.join(os.path.abspath(output_dir), f"{fire_name}_Gridded_VIIRS")
    step2_plots_dir = os.path.join(base_output_dir, "Plots", "Step2_Gridded_Swaths")
    logs_dir = os.path.join(base_output_dir, "Logs")
    mapping_output_dir = os.path.join(base_output_dir, "Data", "mappings")
    
    for directory in [step2_plots_dir, logs_dir, mapping_output_dir]:
        os.makedirs(directory, exist_ok=True)
    
    swath_df = query_available_swath_data(fire_name, output_dir=output_dir)
    grid_meta = create_reference_grid(region=grid_region, resolution=grid_resolution)
    fire_extent, grid_gdf = create_fire_grid_extent(bbox, grid_meta, pad=grid_pad)
    
    print(f"\nReference grid: EPSG:{grid_meta['crs_epsg']}, {grid_meta['resolution_m']}m")
    print(f"Fire extent: {fire_extent['n_rows']}×{fire_extent['n_cols']} "
          f"= {fire_extent['n_rows'] * fire_extent['n_cols']:,} cells")

    # ===================================================================
    # ZARR STORE SETUP
    # ===================================================================
    
    local_zarr_path = os.path.join(base_output_dir, "Data", f"{fire_name}_datacube.zarr")
    # initialize fs and s3 path variables
    fs = None
    s3_zarr_path = None
    if s3_prefix:
        fs = s3fs.S3FileSystem()
        s3_zarr_path = f"{s3_prefix}{fire_name}_Gridded_VIIRS/Data/{fire_name}_datacube.zarr"
    
    # Handle existing local Zarr: remove if overwriting, otherwise append new timesteps to it
    if os.path.exists(local_zarr_path):
        if overwrite:
            shutil.rmtree(local_zarr_path)
            local_zarr_written = False
        else:
            local_zarr_written = True  # append to it; prevents mode='w' on first batch flush
    else:
        local_zarr_written = False

    # Populate existing_times for per-swath skip logic
    existing_times = set()

    if s3_zarr_path and fs.exists(s3_zarr_path) and not overwrite:
        # S3 is the reference copy — read existing timesteps from S3 Zarr
        try:
            store = s3fs.S3Map(root=s3_zarr_path, s3=fs)
            existing_ds = xr.open_zarr(store)
            existing_times = set(pd.DatetimeIndex(existing_ds['time'].values))
            existing_ds.close()
            print(f"Existing S3 Zarr store found with {len(existing_times)} timesteps")
        except Exception as e:
            print(f"Could not read existing S3 Zarr store: {e}")
    elif local_zarr_written:
        # Local is the reference copy — read existing timesteps from local Zarr
        try:
            local_ds = xr.open_zarr(local_zarr_path)
            existing_times = set(pd.DatetimeIndex(local_ds['time'].values))
            local_ds.close()
            print(f"Existing local Zarr store found with {len(existing_times)} timesteps")
        except Exception as e:
            print(f"Could not read existing local Zarr store: {e}")

    existing_s3_store = s3_zarr_path is not None and len(existing_times) > 0 and not overwrite

    run_timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"{fire_name}_step2_gridding_log_{run_timestamp}.txt"
    log_path = os.path.join(logs_dir, log_filename)
    
    log_file = open(log_path, 'w')

    try:

        log_message("=" * 70, log_file, include_timestamp=False)
        log_message("STEP 2: SWATH-TO-GRID PROCESSING LOG",log_file, include_timestamp=False)
        log_message(f"Run started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file,
                    include_timestamp=False)
        log_message("=" * 70, log_file, include_timestamp=False)
        log_message(f"Fire name: {fire_name}", log_file)
        log_message(f"BBOX: {bbox}", log_file)
        log_message(f"Date range: {start} to {end}", log_file)
        log_message(f"Reference grid: EPSG:{grid_meta['crs_epsg']}, {grid_meta['resolution_m']}m",log_file)
        log_message(f"Fire extent: {fire_extent['n_rows']}×{fire_extent['n_cols']}",log_file)
        log_message(f"Zarr path: {s3_zarr_path}",log_file)
        log_message(f"Batch size: {batch_size}",log_file)
        log_message(f"Total swath files: {len(swath_df)}",log_file)
        log_message(f"Max to process: {n_timesteps}",log_file)
        log_message(f"Overwrite: {overwrite}",log_file)
        log_message(f"Make plots: {make_plots}",log_file)
        log_message(f"Existing timesteps: {len(existing_times)}",log_file)
        log_message("",log_file)
    
        # ===================================================================
        # PROCESS ALL SWATHS
        # ===================================================================
    
        log_message(f"{'=' * 70}", log_file, include_timestamp=False)
        log_message(f"PROCESSING {min(n_timesteps, len(swath_df))} SWATHS",
                    log_file, include_timestamp=False)
        log_message(f"{'=' * 70}",log_file, include_timestamp=False)
        log_message("",log_file)
    
        processed_count = 0
        skipped_count = 0
        already_exists_count = 0
        error_count = 0
        all_cleaning_reports = []
    
        # Batched Zarr write accumulator
        pending_datasets = []
        pending_metadata = []  # track what's in the batch for logging
    
        # Timing accumulators
        timing_records = []
        
        swaths_to_process = swath_df.iloc[:n_timesteps]
        pbar = tqdm(swaths_to_process.iterrows(), total=len(swaths_to_process),
                    desc="Gridding swaths", unit="swath")
    
        for idx, swath_info in pbar:
            
            sat = swath_info['satellite']
            timestamp = swath_info['timestamp']
            filepath = swath_info['filepath']
            file_timestamp = timestamp.strftime('%Y%m%d_%H%M')
            
            pbar.set_description(f"Gridding {sat} {timestamp.strftime('%Y-%m-%d %H:%M')}")
            
            # --- Check if already in Zarr ---
            if not overwrite and pd.Timestamp(timestamp) in existing_times:
                already_exists_count += 1
                log_message(f"Skipping {sat}_{file_timestamp} — already in Zarr",log_file,
                            print_to_console=False)
                pbar.set_postfix({
                    'done': processed_count, 'exists': already_exists_count,
                    'skip': skipped_count, 'err': error_count,
                    'batch': len(pending_datasets)
                })
                continue
            
            # --- Check if plot exists ---
            plot_filename = f"{sat}_{file_timestamp}_gridded.png"
            plot_output_path = os.path.join(step2_plots_dir, plot_filename)
            plot_exists = os.path.exists(plot_output_path) if make_plots else False
            
            t_total_start = time.time()
            timing = {'filename': f"{sat}_{file_timestamp}"}
            
            try:
                # ---- Load swath ----
                t0 = time.time()
                swath_ds = xr.open_dataset(filepath)
                timing['t_load'] = time.time() - t0
                
                # ---- Map to grid ----
                t0 = time.time()
                mapping_df, swath_gdf = map_swath_to_reference_grid(
                    swath_ds, grid_gdf, verbose=False
                )
                timing['t_map'] = time.time() - t0
                
                if len(mapping_df) == 0:
                    skipped_count += 1
                    log_message(f"Skipping {sat}_{file_timestamp} — no valid mappings",
                                log_file, print_to_console=False)
                    swath_ds.close()
                    del mapping_df, swath_gdf
                    pbar.set_postfix({
                        'done': processed_count, 'exists': already_exists_count,
                        'skip': skipped_count, 'err': error_count,
                        'batch': len(pending_datasets)
                    })
                    continue
                
                # ---- Clean (optional)----
                if remove_bowtie or deduplicate_scans:
                    t0 = time.time()
                    mapping_df, report = clean_swath_mapping(mapping_df, swath_gdf, remove_bowtie=remove_bowtie,
                                                             deduplicate_scans=deduplicate_scans, verbose=False)
                    timing['t_clean'] = time.time() - t0
                    
                    report['filename'] = f"{sat}_{file_timestamp}"
                    report['satellite'] = sat
                    report['timestamp'] = str(timestamp)
                    all_cleaning_reports.append(report)
                
                if len(mapping_df) == 0:
                    skipped_count += 1
                    log_message(f"Skipping {sat}_{file_timestamp} — no pixels after cleaning",
                                log_file, print_to_console=False)
                    swath_ds.close()
                    pbar.set_postfix({
                        'done': processed_count, 'exists': already_exists_count,
                        'skip': skipped_count, 'err': error_count,
                        'batch': len(pending_datasets)
                    })
                    continue
    
                # ---- Save mapping ----
                t0 = time.time()
                mapping_df.to_parquet(
                    os.path.join(mapping_output_dir, f"{sat}_{file_timestamp}_mapping.parquet"),
                    index=False
                )
                timing['t_save_mapping'] = time.time() - t0
                
                # ---- Aggregate ----
                t0 = time.time()
                grid_agg_df = aggregate_pixels_to_grid(mapping_df, swath_gdf, verbose=False)
                timing['t_aggregate'] = time.time() - t0
                
                # ---- Convert to xarray ----
                t0 = time.time()
                ds = grid_agg_to_xarray(grid_agg_df, fire_extent, grid_meta, swath_ds,
                                        test=True, verbose=False)
                timing['t_to_xarray'] = time.time() - t0
                
                timing['t_total'] = time.time() - t_total_start
                timing_records.append(timing)
                
                # ---- Accumulate for batched Zarr write ----
                pending_datasets.append(ds)
                pending_metadata.append(f"{sat}_{file_timestamp}")
                
                log_message(f"Processed: {sat}_{file_timestamp} ({timing['t_total']:.1f}s)",
                            log_file, print_to_console=False)
                
                # ---- Save plot (separate from data pipeline) ----
                if make_plots and (overwrite or not plot_exists):
                    try:
                        plot_gridded_swath(
                            ds.isel(time=0), swath_ds, grid_meta,
                            save_path=plot_output_path
                        )
                        log_message(f"Plotted: {plot_filename}",log_file, print_to_console=False)
                    except Exception as plot_err:
                        log_message(
                            f"PLOT ERROR {sat}_{file_timestamp}: "
                            f"{type(plot_err).__name__} - {str(plot_err)}",
                            log_file, print_to_console=True
                        )
                    finally:
                        plt.close('all')
                        gc.collect()
                
                swath_ds.close()
                # del mapping_df, swath_gdf, cleaned_df, grid_agg_df
                processed_count += 1
                
                # ---- Flush batch to Zarr if full ----
                if len(pending_datasets) >= batch_size:
                    t0 = time.time()
                    batch_ds = xr.concat(pending_datasets, dim='time')
                    t_concat = time.time() - t0
                    
                    t1 = time.time()
                    if not local_zarr_written:
                        encoding = get_zarr_encoding(batch_ds, fire_extent)
                        batch_ds.to_zarr(local_zarr_path, mode='w', encoding=encoding)
                        local_zarr_written = True
                        log_message(f"Created local Zarr store with {len(pending_datasets)} timesteps "
                                    f"(concat={t_concat:.1f}s, write={time.time()-t1:.1f}s)",
                                    log_file, print_to_console=True)
                    else:
                        batch_ds.to_zarr(local_zarr_path, mode='a', append_dim='time')
                        log_message(f"Appended batch of {len(pending_datasets)} timesteps "
                                    f"to local Zarr (concat={t_concat:.1f}s, write={time.time()-t1:.1f}s)",
                                    log_file, print_to_console=True)
                    
                    for d in pending_datasets:
                        d.close()
                    batch_ds.close()
                    del batch_ds
                    pending_datasets = []
                    pending_metadata = []
                    gc.collect()
                
                pbar.set_postfix({
                    'done': processed_count, 'exists': already_exists_count,
                    'skip': skipped_count, 'err': error_count,
                    'batch': len(pending_datasets)
                })
    
            except Exception as e:
                error_count += 1
                timing['t_total'] = time.time() - t_total_start
                timing['error'] = f"{type(e).__name__}: {str(e)}"
                timing_records.append(timing)
                log_message(f"ERROR {sat}_{file_timestamp}: {type(e).__name__} - {str(e)}",
                            log_file, print_to_console=True)
                pbar.set_postfix({
                    'done': processed_count, 'exists': already_exists_count,
                    'skip': skipped_count, 'err': error_count,
                    'batch': len(pending_datasets)
                })
                continue
    
        pbar.close()
        
        # ===================================================================
        # FLUSH REMAINING BATCH
        # ===================================================================
    
        if len(pending_datasets) > 0:
            log_message(f"\nFlushing final batch of {len(pending_datasets)} timesteps...",
                       log_file)
            
            t0 = time.time()
            batch_ds = xr.concat(pending_datasets, dim='time')
            
            if not local_zarr_written:
                encoding = get_zarr_encoding(batch_ds, fire_extent)
                batch_ds.to_zarr(local_zarr_path, mode='w', encoding=encoding)
                local_zarr_written = True
            else:
                batch_ds.to_zarr(local_zarr_path, mode='a', append_dim='time')
            
            t_write = time.time() - t0
            log_message(f"Final batch written ({t_write:.1f}s)",log_file)

            for d in pending_datasets:
                d.close()
            batch_ds.close()
            del batch_ds
            pending_datasets = []
            gc.collect()


        # ===================================================================
        # COPY LOCAL ZARR TO S3
        # ===================================================================
        
        if local_zarr_written and copy_to_s3:
            if existing_s3_store:
                # Resume — append only new timesteps
                log_message("Appending new timesteps to existing S3 store...",log_file)
                t0 = time.time()
                
                local_ds = xr.open_zarr(local_zarr_path)
                store = s3fs.S3Map(root=s3_zarr_path, s3=fs)
                local_ds.to_zarr(store, mode='a', append_dim='time')
                local_ds.close()
                
                zarr.consolidate_metadata(fs.get_mapper(s3_zarr_path))

                log_message(f"Appended to S3 ({time.time() - t0:.1f}s)",log_file)
                if remove_local:
                    shutil.rmtree(base_output_dir)
                    log_message(f"Local files removed: {base_output_dir}",log_file)
            else:
                # Fresh run — full copy
                log_message("Copying Zarr store to S3...",log_file)
                t0 = time.time()

                if fs.exists(s3_zarr_path):
                    fs.rm(s3_zarr_path, recursive=True)

                local_ds = xr.open_zarr(local_zarr_path)
                store = s3fs.S3Map(root=s3_zarr_path, s3=fs)
                encoding = get_zarr_encoding(local_ds, fire_extent)
                local_ds.to_zarr(store, mode='w', encoding=encoding)
                local_ds.close()

                zarr.consolidate_metadata(fs.get_mapper(s3_zarr_path))

                log_message(f"Copied to {s3_zarr_path} ({time.time() - t0:.1f}s)",log_file)
                if remove_local:
                    shutil.rmtree(base_output_dir)
                    log_message(f"Local files removed: {base_output_dir}",log_file)

        else:
            log_message("No new data written — skipping S3 copy",log_file)


        # ===================================================================
        # PERSISTENCE CALCULATION
        # ===================================================================

        if add_persistence:
            if persistence_fire_mask_col is None:
                _suffixes = ['max', 'aw']
                _cols = ['fire_mask_max', 'fire_mask_area_weighted_majority']
            else:
                _suffixes = [persistence_suffix]
                _cols = [persistence_fire_mask_col]

            if not local_zarr_written:
                log_message("Skipping persistence — no Zarr store was written this run.", log_file)

            elif copy_to_s3:
                # S3 is truth — compute from the full S3 store (all timesteps present after append)
                log_message("Computing fire persistence metrics from S3 store...", log_file)
                s3_store = s3fs.S3Map(root=s3_zarr_path, s3=fs)
                all_data = xr.open_zarr(s3_store)
                for col, suf in zip(_cols, _suffixes):
                    all_data = compute_fire_persistence_baseline(
                        all_data, col, suf,
                        start_fire_mask_value=persistence_start_threshold,
                        end_fire_mask_value=persistence_end_threshold,
                    )
                _write_persistence_vars(all_data, s3_store, _suffixes)
                all_data.close()
                log_message("Persistence metrics written to S3 Zarr.", log_file)

                # Mirror full S3 store (including persistence) to local if local was kept
                if not remove_local:
                    log_message("Mirroring full S3 store to local Zarr...", log_file)
                    full_ds = xr.open_zarr(s3_store)
                    encoding = get_zarr_encoding(full_ds, fire_extent)
                    full_ds.to_zarr(local_zarr_path, mode='w', encoding=encoding)
                    full_ds.close()
                    log_message("Local Zarr updated from S3.", log_file)

            else:
                # Local is truth
                if existing_s3_store:
                    log_message(
                        "WARNING: Skipping persistence — copy_to_s3=False but S3 has prior data "
                        "this local run is unaware of. Re-run with copy_to_s3=True to compute "
                        "persistence over the full accumulated dataset.",
                        log_file
                    )
                else:
                    log_message("Computing fire persistence metrics from local Zarr...", log_file)
                    all_data = xr.open_zarr(local_zarr_path)
                    for col, suf in zip(_cols, _suffixes):
                        all_data = compute_fire_persistence_baseline(
                            all_data, col, suf,
                            start_fire_mask_value=persistence_start_threshold,
                            end_fire_mask_value=persistence_end_threshold,
                        )
                    _write_persistence_vars(all_data, local_zarr_path, _suffixes)
                    all_data.close()
                    log_message("Persistence metrics written to local Zarr.", log_file)


        # ===================================================================
        # SUMMARY
        # ===================================================================
    
        log_message(f"\n{'=' * 70}", log_file, include_timestamp=False)
        log_message("PROCESSING COMPLETE", log_file, include_timestamp=False)
        log_message(f"{'=' * 70}", log_file, include_timestamp=False)
        log_message(f"  Successfully processed: {processed_count}", log_file)
        log_message(f"  Already existed (skipped): {already_exists_count}", log_file)
        log_message(f"  Skipped (no data/empty): {skipped_count}", log_file)
        log_message(f"  Errors: {error_count}", log_file)
        log_message(f"  Total: {processed_count + already_exists_count + skipped_count + error_count}",
                   log_file)
    
        # Cleaning summary
        if len(all_cleaning_reports) > 0:
            reports_df = pd.DataFrame(all_cleaning_reports)
            
            log_message(f"\n  Cleaning statistics across {len(reports_df)} swaths:", log_file)
            log_message(f"    Mean bowtie removed: "
                        f"{reports_df.get('bowtie_removed', pd.Series([0])).mean():.1f}",log_file)
            log_message(f"    Mean dedup removed: "
                        f"{reports_df.get('dedup_removed', pd.Series([0])).mean():.1f}",log_file)
            log_message(f"    Mean % removed: {reports_df['pct_removed'].mean():.1f}%",log_file)
            log_message(f"    Max % removed: {reports_df['pct_removed'].max():.1f}%",log_file)
    
        # Timing summary
        if len(timing_records) > 0:
            timing_df = pd.DataFrame(timing_records)
            if 'error' in timing_df.columns:
                completed = timing_df[timing_df['error'].isna()].copy()
            else:
                completed = timing_df.copy()
            
            if len(completed) > 0:
                step_cols = ['t_load', 't_map', 't_clean', 't_save_mapping', 't_aggregate', 't_to_xarray']
                available_steps = [c for c in step_cols if c in completed.columns]
                
                log_message(f"\n  Timing (excluding Zarr writes):",log_file)
                for col in available_steps:
                    vals = completed[col]
                    log_message(f"    {col}: mean={vals.mean():.2f}s, total={vals.sum():.1f}s",
                               log_file)
                
                t_processing = sum(completed[c].sum() for c in available_steps)
                log_message(f"    Processing total: {t_processing:.1f}s ({t_processing/60:.1f} min)",
                           log_file)
    
        # Zarr store summary
        if (existing_s3_store or local_zarr_written) and s3_zarr_path:
            try:
                store = s3fs.S3Map(root=s3_zarr_path, s3=fs)
                final_ds = xr.open_zarr(store)
                log_message(f"\n  Zarr datacube summary:",log_file)
                log_message(f"    Path: {s3_zarr_path}",log_file)
                log_message(f"    Time steps: {final_ds.sizes['time']}",log_file)
                log_message(f"    Spatial dims: y={final_ds.sizes['y']}, x={final_ds.sizes['x']}",log_file)
                log_message(f"    Variables: {len(final_ds.data_vars)}",log_file)
                log_message(f"    Time range: {final_ds.time.values[0]} to {final_ds.time.values[-1]}",log_file)
                final_ds.close()
            except Exception as e:
                log_message(f"  Could not read final Zarr store: {e}",log_file)
    
        log_message(f"\nOutputs:",log_file)
        log_message(f"  Zarr: {s3_zarr_path}",log_file)
        if make_plots:
            log_message(f"  Plots: {step2_plots_dir}",log_file)
        log_message(f"  Log: {log_path}",log_file)
        log_message(f"\nRun completed: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",log_file)
    
    finally:
        log_file.close()
    
    print(f"\n{'=' * 70}")
    print(f"Log file saved: {log_path}")
    print(f"{'=' * 70}")


if __name__ == "__main__":

    import argparse
    import ast

    parser = argparse.ArgumentParser(
        description="Standardize and grid VIIRS swath data into a Zarr datacube."
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
        "--grid_region",
        type=str,
        default='conus',
        help="Reference grid region. Options: 'conus', 'global', 'custom'. Default: 'conus'."
    )
    parser.add_argument(
        "--grid_resolution",
        type=int,
        default=375,
        help="Reference grid cell size in meters. Default: 375."
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="If set, reprocess and overwrite existing output files."
    )
    parser.add_argument(
        "--make_plots",
        action="store_true",
        default=False,
        help="If set, generate and save gridded swath plots."
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="Number of swaths to accumulate before flushing to the local Zarr store. Default: 50."
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
        "--add_persistence",
        action="store_true",
        default=False,
        help="If set, compute fire persistence metrics after gridding and write them to the Zarr store."
    )
    parser.add_argument(
        "--persistence_fire_mask_col",
        type=str,
        default=None,
        help="Fire mask column to use for persistence. If not set, runs both 'fire_mask_max' and 'fire_mask_area_weighted_majority'."
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
    # CALL standardize_swaths
    # ===================================================================

    bbox = ast.literal_eval(args.bbox)

    standardize_swaths(
        fire_name=args.fire_name,
        bbox=bbox,
        start=args.start,
        end=args.end,
        n_timesteps=args.n_timesteps,
        grid_region=args.grid_region,
        grid_resolution=args.grid_resolution,
        overwrite=args.overwrite,
        make_plots=args.make_plots,
        copy_to_s3=args.copy_to_s3,
        s3_prefix=args.s3_prefix,
        output_dir=args.output_dir,
        remove_local=args.remove_local,
        batch_size=args.batch_size,
        grid_pad=args.grid_pad,
        remove_bowtie=args.remove_bowtie,
        deduplicate_scans=args.deduplicate_scans,
        add_persistence=args.add_persistence,
        persistence_fire_mask_col=args.persistence_fire_mask_col,
        persistence_suffix=args.persistence_suffix,
        persistence_start_threshold=args.persistence_start_threshold,
        persistence_end_threshold=args.persistence_end_threshold,
    )

    # ===================================================================
    # USAGE EXAMPLE
    # ===================================================================
    #
    # python standardizations.py \
    #     --fire_name 'Dragon_Bravo_TEST' \
    #     --start '2025-07-01' \
    #     --end '2025-07-10' \
    #     --bbox '[-112.309113, 36.112467, -111.800995, 36.748712]' \
    #     --n_timesteps -1 \
    #     --grid_region 'conus' \
    #     --grid_resolution 375 \
    #     --batch_size 50 \
    #     --grid_pad 10 \
    #     --copy_to_s3 \
    #     --s3_prefix 's3://maap-ops-workspace/shared/gsfc_landslides/FireSense/' \
    #     --output_dir 'VIIRS-cubed-outputs' \
    #     --overwrite \
    #     --add_persistence \
    #     --persistence_start_threshold 6 \
    #     --persistence_end_threshold 6