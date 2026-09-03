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
from zarr.codecs import BloscCodec

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


def query_available_swath_data(fire_name):

    # ===================================================================
    # LOCATE STEP 1 OUTPUT FILES
    # ===================================================================
    
    # Define paths based on Step 1 naming convention
    base_output_dir = os.path.expanduser(f"~/VIIRS_L1_Outputs/{NAME}_Gridded_VIIRS")
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
        print(f"\nYou may want to delete or re-process these files:")
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