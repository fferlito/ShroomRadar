import os
import re
import zipfile
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from whitebox.whitebox_tools import WhiteboxTools
import earthaccess
from tqdm import tqdm
import contextlib
import io
import math
import geopandas as gpd
import shutil
import logging
from datetime import datetime

def tile_id_from_coords(lat: float, lon: float) -> str | None:
    """Convert coords to tile ID (e.g. N40W106)."""
    if pd.isna(lat) or pd.isna(lon):
        return None
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    # Use floor for both positive and negative coordinates to handle edge cases properly
    lat_tile = math.floor(lat)
    lon_tile = math.floor(lon)
    return f"{ns}{abs(lat_tile):02d}{ew}{abs(lon_tile):03d}"


def download_dem_bbox(
    min_lon: float,
    min_lat: float,
    max_lon: float,
    max_lat: float,
    out_dir: str = "dem_tiles",
    prefer: str = "SRTMGL1",
):
    """
    Download DEM tiles within a specified bounding box.

    Uses earthaccess to search for and download either SRTMGL1 (1-arc second, ~30m)
    or Copernicus DEM GLO-30 data.

    Args:
        min_lon (float): The minimum longitude of the bounding box.
        min_lat (float): The minimum latitude of the bounding box.
        max_lon (float): The maximum longitude of the bounding box.
        max_lat (float): The maximum latitude of the bounding box.
        out_dir (str, optional): The directory to save downloaded tiles.
            Defaults to "dem_tiles".
        prefer (str, optional): The preferred DEM dataset. Can be 'SRTMGL1'
            or 'COPDEM'. Defaults to "SRTMGL1".

    Returns:
        list: A list of local file paths to the downloaded DEM tiles. Returns
              an empty list if no data is found or an error occurs.
    """
    os.makedirs(out_dir, exist_ok=True)
    earthaccess.login(strategy="environment", persist=True)
    dataset = ("SRTMGL1", "003") if prefer == "SRTMGL1" else ("COPDEM_GLO_30", "001")
    try:
        results = earthaccess.search_data(
            short_name=dataset[0],
            version=dataset[1],
            bounding_box=(min_lon, min_lat, max_lon, max_lat),
            count=10,
        )
    except IndexError:
        return []
    if not results or len(results) == 0:
        return []
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        paths = earthaccess.download(results, out_dir)
    return paths


def download_dem_point(
    lat: float, lon: float, out_dir: str = "dem_tiles", buffer: float = 0.1
) -> tuple[list[str], str]:
    """
    Download DEM tiles covering a specific point.

    Creates a small bounding box around the given point and downloads the
    corresponding DEM tiles. It first attempts to download SRTMGL1 data and
    falls back to Copernicus DEM GLO-30 if SRTMGL1 is not available.

    Args:
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.
        out_dir (str, optional): The directory to save downloaded tiles.
            Defaults to "dem_tiles".
        buffer (float, optional): The buffer in degrees to create a bounding
            box around the point. Defaults to 0.1.

    Returns:
        tuple[list[str], str]: A tuple containing a list of local file paths
                               to the downloaded DEM files and a string
                               indicating the source ('SRTM', 'Copernicus',
                               or 'None').
    """

    min_lon = max(-180.0, lon - buffer)
    max_lon = min(180.0, lon + buffer)
    min_lat = max(-90.0, lat - buffer)
    max_lat = min(90.0, lat + buffer)
    paths = download_dem_bbox(
        min_lon, min_lat, max_lon, max_lat, out_dir=out_dir, prefer="SRTMGL1"
    )
    if paths:
        return paths, "SRTM"
    paths = download_dem_bbox(
        min_lon, min_lat, max_lon, max_lat, out_dir=out_dir, prefer="COPDEM"
    )
    if paths:
        return paths, "Copernicus"
    return [], "None"


def parse_hgt_bounds(hgt_path: str) -> tuple[float, float, float, float]:
    """
    Parse the geographic bounds from an HGT filename.

    HGT filenames encode the latitude and longitude of the
    bottom-left corner of the tile, e.g., 'N40W106.hgt'. This
    function extracts these coordinates and calculates the bounding box.

    Args:
        hgt_path (str): The file path to the HGT file.

    Returns:
        tuple[float, float, float, float]: A tuple containing the
            bounding box coordinates (west, south, east, north).

    Raises:
        ValueError: If the filename does not match the expected HGT format.
    """
    name = os.path.splitext(os.path.basename(hgt_path))[0]
    m = re.match(r"([NS])(\d{1,2})([EW])(\d{1,3})", name, re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot parse HGT name: {hgt_path}")
    lat_sign = 1 if m.group(1).upper() == "N" else -1
    lon_sign = 1 if m.group(3).upper() == "E" else -1
    lat0 = lat_sign * math.floor(int(m.group(2)))
    lon0 = lon_sign * math.floor(int(m.group(4)))
    west, south = float(lon0), float(lat0)
    east, north = west + 1.0, south + 1.0
    return west, south, east, north


def hgt_to_gtiff(hgt_path: str, tif_path: str) -> None:
    """
    Convert an HGT file to a georeferenced GeoTIFF.

    This function reads an SRTM HGT file, which contains raw elevation
    data, and converts it into a standard GeoTIFF format with the correct
    georeferencing information (CRS, transform, bounds).

    Args:
        hgt_path (str): Path to the input HGT file. The filename must
                        follow the standard HGT naming convention (e.g.,
                        'N40W106.hgt') to parse the bounds.
        tif_path (str): Path for the output GeoTIFF file.

    Raises:
        ValueError: If the HGT file has an unexpected side length (not
                    3601 or 1201 pixels).
    """
    west, _, _, north = parse_hgt_bounds(hgt_path)
    nbytes = os.path.getsize(hgt_path)
    side = int(np.sqrt(nbytes // 2))
    if side not in (3601, 1201):
        raise ValueError(f"Unexpected HGT side length: {side}")
    data = np.fromfile(hgt_path, dtype=">i2").reshape((side, side))
    data = data[:-1, :-1]
    res = 1.0 / (side - 1)
    transform = from_origin(west, north, res, res)
    profile = {
        "driver": "GTiff",
        "height": data.shape[0],
        "width": data.shape[1],
        "count": 1,
        "dtype": "int16",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": -32768,
        "tiled": True,
        "compress": "LZW",
    }
    with rasterio.open(tif_path, "w", **profile) as dst:
        dst.write(data, 1)


def prepare_tif(path: str) -> str:
    """
    Prepare a DEM file, ensuring it is a usable GeoTIFF.

    This function handles different input formats:
    - If the input is already a .tif, it returns its absolute path.
    - If the input is a .zip, it extracts the content.
        - If the zip contains a .tif, it's extracted.
        - If it contains a .hgt, it's extracted and converted to a .tif.
    - After successful extraction/conversion, the original .zip and any
      intermediate .hgt files are removed.

    Args:
        path (str): The path to the input file (.tif or .zip).

    Returns:
        str: The absolute path to the final GeoTIFF file.

    Raises:
        FileNotFoundError: If the input file is not a .tif or .zip, or if
                           the zip archive does not contain a .tif or .hgt file.
    """
    if path.lower().endswith(".tif"):
        return os.path.abspath(path)
    if path.lower().endswith(".zip"):
        tif_out, hgt_out = None, None
        with zipfile.ZipFile(path, "r") as z:
            tifs = [m for m in z.namelist() if m.lower().endswith(".tif")]
            if tifs:
                tif_out = os.path.join(os.path.dirname(path), os.path.basename(tifs[0]))
                if not os.path.exists(tif_out):
                    z.extract(tifs[0], os.path.dirname(path))
                tif_out = os.path.abspath(tif_out)
            else:
                hgts = [m for m in z.namelist() if m.lower().endswith(".hgt")]
                if hgts:
                    hgt_out = os.path.join(
                        os.path.dirname(path), os.path.basename(hgts[0])
                    )
                    if not os.path.exists(hgt_out):
                        z.extract(hgts[0], os.path.dirname(path))
                    tif_out = hgt_out.replace(".hgt", ".tif")
                    if not os.path.exists(tif_out):
                        hgt_to_gtiff(hgt_out, tif_out)
                    try:
                        os.remove(hgt_out)
                    except PermissionError:
                        pass
                    tif_out = os.path.abspath(tif_out)
        try:
            os.remove(path)
        except PermissionError:
            pass
        if tif_out:
            return tif_out
        else:
            raise FileNotFoundError(f"No .tif or .hgt in {path}")
    raise FileNotFoundError(f"Unsupported DEM format: {path}")



wbt = WhiteboxTools()
wbt.verbose = False


def file_exists_and_valid(path: str) -> bool:
    return os.path.exists(path) and file_exists_and_valid(path)

def run_whitebox(
    tif_file: str,
    need_slope: bool = False,
    need_aspect: bool = False,
    need_geomorph: bool = False,
    slope_dir: str | None = None,
    aspect_dir: str | None = None,
    geomorph_dir: str | None = None,
) -> tuple[str, str | None, str | None, str | None]:

    # Always use absolute Windows-style paths
    tif_file = os.path.abspath(tif_file)

    # Ensure output directories exist
    if slope_dir:
        os.makedirs(slope_dir, exist_ok=True)
    if aspect_dir:
        os.makedirs(aspect_dir, exist_ok=True)
    if geomorph_dir:
        os.makedirs(geomorph_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(tif_file))[0]

    slope_tif = os.path.abspath(
        os.path.join(slope_dir, f"{base_name}_slope.tif")
        if slope_dir else f"{os.path.splitext(tif_file)[0]}_slope.tif"
    )
    aspect_tif = os.path.abspath(
        os.path.join(aspect_dir, f"{base_name}_aspect.tif")
        if aspect_dir else f"{os.path.splitext(tif_file)[0]}_aspect.tif"
    )
    geomorph_tif = os.path.abspath(
        os.path.join(geomorph_dir, f"{base_name}_geomorph.tif")
        if geomorph_dir else f"{os.path.splitext(tif_file)[0]}_geomorph.tif"
    )

    # Debug: print exactly what paths Whitebox will see
    print(f"DEM input: {tif_file}")
    print(f"Slope out: {slope_tif}")
    print(f"Aspect out: {aspect_tif}")
    print(f"Geomorph out: {geomorph_tif}")

    if need_slope and not file_exists_and_valid(slope_tif):
        wbt.slope(dem=tif_file, output=slope_tif, zfactor=1.0, units="degrees")
    if need_aspect and not file_exists_and_valid(aspect_tif):
        wbt.aspect(dem=tif_file, output=aspect_tif)
    if need_geomorph and not file_exists_and_valid(geomorph_tif):
        wbt.geomorphons(
            dem=tif_file, output=geomorph_tif, search=50, threshold=0.0, forms=True
        )

    return (
        tif_file,
        slope_tif if file_exists_and_valid(slope_tif) else None,
        aspect_tif if file_exists_and_valid(aspect_tif) else None,
        geomorph_tif if file_exists_and_valid(geomorph_tif) else None,
    )



def extract_value(raster: str | None, lat: float, lon: float) -> float | None:
    """
    Extracts the value of a raster at a specific latitude and longitude.

    Args:
        raster (str | None): Path to the raster file.
        lat (float): The latitude of the point.
        lon (float): The longitude of the point.

    Returns:
        float | None: The raster value at the given coordinates. Returns None
                      if the raster is invalid, the point is outside the
                      raster's extent, or the value corresponds to the
                      nodata value.
    """
    if not file_exists_and_valid(raster):
        return None
    try:
        with rasterio.open(raster) as src:
            nd = src.nodata
            # rasterio.sample returns a generator
            for val in src.sample([(lon, lat)]):
                v = float(val[0])
                # Check for NaN and nodata value
                if np.isnan(v) or (nd is not None and v == nd):
                    return None
                return v
    except Exception:
        # This could happen if coordinates are outside the raster extent,
        # or other rasterio errors.
        return None

def valid_raster(path: str | None) -> bool:
    """Check if a raster exists, non-empty, and can be opened by rasterio."""
    if not path or not os.path.exists(path) or os.path.getsize(path) == 0:
        return False
    try:
        with rasterio.open(path) as src:
            _ = src.count
        return True
    except Exception:
        return False

def file_exists_and_valid(path: str) -> bool:
    return os.path.exists(path) and valid_raster(path)


def enrich_geojson(
    input_geojson: str,
    output_geojson: str,
    out_dir: str = "dem_tiles",
    download_tiles: bool = True,
    variables: list[str] | None = None,
    verbose: bool = True,
) -> None:
    """
    Enriches a GeoJSON file with topographic data derived from Digital Elevation Models (DEMs).
    """

    if variables is None:
        variables = ["dem", "slope", "aspect", "geomorphons"]

    # Logging setup
    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

    os.makedirs(out_dir, exist_ok=True)

    # Create subfolders
    dem_dir = os.path.join(out_dir, "dem")
    slope_dir = os.path.join(out_dir, "slope")
    aspect_dir = os.path.join(out_dir, "aspect")
    geomorph_dir = os.path.join(out_dir, "geomorphons")
    for d in [dem_dir, slope_dir, aspect_dir, geomorph_dir]:
        os.makedirs(d, exist_ok=True)

    # Validate variables list
    valid_variables = ["dem", "slope", "aspect", "geomorphons"]
    invalid_vars = [var for var in variables if var not in valid_variables]
    if invalid_vars:
        raise ValueError(
            f"Invalid variables: {invalid_vars}. Valid options are: {valid_variables}"
        )

    # Load GeoJSON
    if not os.path.exists(input_geojson):
        raise FileNotFoundError(f"Input GeoJSON file not found: {input_geojson}")

    gdf = gpd.read_file(input_geojson)

    # Ensure WGS84
    if gdf.crs is None:
        logger.warning("No CRS found, assuming EPSG:4326")
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # ✅ Compute centroids safely in a projected CRS
    gdf_proj = gdf.to_crs(epsg=3857)  # Web Mercator (planar)
    centroids = gdf_proj.geometry.centroid.to_crs(epsg=4326)

    # Filter SRTM coverage (-56 to 60 in latitude)
    gdf = gdf[centroids.y.between(-56, 60)]
    centroids = centroids[gdf.index]  # align indices

    logger.info(f"Loaded {len(gdf)} features")

    # Initialize columns
    base_cols = []
    if "dem" in variables:
        base_cols.extend(["dem", "dem_source"])
    if "slope" in variables:
        base_cols.append("slope")
    if "aspect" in variables:
        base_cols.append("aspect")
    if "geomorphons" in variables:
        base_cols.extend(["geomorphon", "geomorphon_class"])

    for col in base_cols:
        if col not in gdf.columns:
            gdf[col] = None

    # Failure columns
    failure_cols = []
    if "dem" in variables:
        failure_cols.append("dem_failure")
    if "slope" in variables:
        failure_cols.append("slope_failure")
    if "aspect" in variables:
        failure_cols.append("aspect_failure")
    if "geomorphons" in variables:
        failure_cols.append("geomorphon_failure")

    for col in failure_cols:
        if col not in gdf.columns:
            gdf[col] = None

    # Stats
    stats = {
        "total_features": len(gdf),
        "invalid_coords": 0,
        "missing_tiles": 0,
        "extraction_failures": 0,
    }
    if "dem" in variables:
        stats["successful_dem"] = 0
    if "slope" in variables:
        stats["successful_slope"] = 0
    if "aspect" in variables:
        stats["successful_aspect"] = 0
    if "geomorphons" in variables:
        stats["successful_geomorphon"] = 0

    # Step 1: Collect per-tile needs
    logger.info("Step 1: Scanning features for tile requirements...")
    tile_needs = {}
    for idx, pt in tqdm(zip(gdf.index, centroids), total=len(gdf), desc="Scanning features"):
        lat, lon = pt.y, pt.x
        try:
            tid = tile_id_from_coords(lat, lon)
        except Exception as e:
            gdf.at[idx, "dem_failure"] = f"Coordinate conversion error: {str(e)}"
            stats["invalid_coords"] += 1
            continue
        if tid is None:
            gdf.at[idx, "dem_failure"] = f"Invalid coordinates: lat={lat}, lon={lon}"
            stats["invalid_coords"] += 1
            continue

        if tid not in tile_needs:
            tile_needs[tid] = {"dem": False, "slope": False, "aspect": False, "geomorphon": False}

        dem_file = os.path.join(dem_dir, f"{tid}.tif")
        slope_file = os.path.join(slope_dir, f"{tid}_slope.tif")
        aspect_file = os.path.join(aspect_dir, f"{tid}_aspect.tif")
        geomorph_file = os.path.join(geomorph_dir, f"{tid}_geomorph.tif")

        if "dem" in variables and pd.isna(gdf.at[idx, "dem"]) and not file_exists_and_valid(dem_file):
            tile_needs[tid]["dem"] = True
        if "slope" in variables and pd.isna(gdf.at[idx, "slope"]) and not file_exists_and_valid(slope_file):
            tile_needs[tid]["slope"] = True
        if "aspect" in variables and pd.isna(gdf.at[idx, "aspect"]) and not file_exists_and_valid(aspect_file):
            tile_needs[tid]["aspect"] = True
        if "geomorphons" in variables and pd.isna(gdf.at[idx, "geomorphon"]) and not file_exists_and_valid(geomorph_file):
            tile_needs[tid]["geomorphon"] = True

    tile_needs = {tid: needs for tid, needs in tile_needs.items() if any(needs.values())}
    logger.info(f"Found {len(tile_needs)} tiles that need processing")

    # Step 2: Acquire DEM tiles (download only for needed tiles)
    logger.info("Step 2: Acquiring DEM tiles...")
    available_dems = {}
    for tid, needs in tqdm(tile_needs.items(), desc="Acquiring tiles"):
        local_tif = os.path.join(dem_dir, f"{tid}.tif")
        if file_exists_and_valid(local_tif):
            available_dems[tid] = ([local_tif], "Local")
            continue
        if download_tiles:
            try:
                m = re.match(r"([NS])(\d{2})([EW])(\d{3})", tid)
                lat0 = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
                lon0 = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
                # center of the 1x1 tile
                zip_paths, source = download_dem_point(lat0 + 0.5, lon0 + 0.5, out_dir=out_dir)
                if zip_paths:
                    tifs = [prepare_tif(zp) for zp in zip_paths]
                    moved_tifs = []
                    for tif in tifs:
                        target_path = os.path.join(dem_dir, f"{tid}.tif")
                        if tif != target_path:
                            shutil.move(tif, target_path)
                        moved_tifs.append(target_path)
                    available_dems[tid] = (moved_tifs, source)
                else:
                    stats["missing_tiles"] += 1
            except Exception as e:
                logger.error(f"Error downloading tile {tid}: {str(e)}")
                stats["missing_tiles"] += 1

    # Step 3: Run Whitebox (only for tiles that need derived rasters)
    logger.info("Step 3: Running Whitebox processing...")
    tile_results = {}
    for tid, (tifs, source) in tqdm(available_dems.items(), desc="Running Whitebox"):
        needs = tile_needs.get(tid, {})
        for tif in tifs:
            try:
                tif_path, slope_path, aspect_path, geomorph_path = run_whitebox(
                    tif,
                    need_slope=("slope" in variables) and needs.get("slope", True),
                    need_aspect=("aspect" in variables) and needs.get("aspect", True),
                    need_geomorph=("geomorphons" in variables) and needs.get("geomorphon", True),
                    slope_dir=slope_dir,
                    aspect_dir=aspect_dir,
                    geomorph_dir=geomorph_dir,
                )
                tile_results[tid] = {
                    "tif": tif_path,
                    "slope": slope_path,
                    "aspect": aspect_path,
                    "geomorphon": geomorph_path,
                    "source": source,
                }
                break
            except Exception as e:
                logger.error(f"Whitebox processing failed for tile {tid}: {str(e)}")
                continue

    # Step 4: Extract values (✅ includes local fallback even if no tiles were processed)
    logger.info("Step 4: Extracting values...")
    geomorph_classes = {
        1: "flat",
        2: "summit",
        3: "ridge",
        4: "shoulder",
        5: "spur",
        6: "slope",
        7: "hollow",
        8: "footslope",
        9: "valley",
        10: "pit",
    }

    for idx, pt in tqdm(zip(gdf.index, centroids), total=len(gdf), desc="Extracting values"):
        lat, lon = pt.y, pt.x
        tid = tile_id_from_coords(lat, lon)
        if tid is None:
            continue

        # Local fallback paths (so extraction works even if tile_needs == {}):
        dem_base = os.path.join(dem_dir, f"{tid}.tif")
        slope_base = os.path.join(slope_dir, f"{tid}_slope.tif")
        aspect_base = os.path.join(aspect_dir, f"{tid}_aspect.tif")
        geomorph_base = os.path.join(geomorph_dir, f"{tid}_geomorph.tif")

        tr = tile_results.get(
            tid,
            {
                "tif": dem_base if file_exists_and_valid(dem_base) else None,
                "slope": slope_base if file_exists_and_valid(slope_base) else None,
                "aspect": aspect_base if file_exists_and_valid(aspect_base) else None,
                "geomorphon": geomorph_base if file_exists_and_valid(geomorph_base) else None,
                "source": "Local",
            },
        )

        # DEM
        if "dem" in variables and pd.isna(gdf.at[idx, "dem"]) and tr.get("tif"):
            try:
                dem_value = extract_value(tr["tif"], lat, lon)
                if dem_value is not None:
                    gdf.at[idx, "dem"] = dem_value
                    gdf.at[idx, "dem_source"] = tr["source"]
                    stats["successful_dem"] += 1
                else:
                    gdf.at[idx, "dem_failure"] = "No data value or out of bounds"
            except Exception as e:
                gdf.at[idx, "dem_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif "dem" in variables and pd.isna(gdf.at[idx, "dem"]) and not tr.get("tif"):
            gdf.at[idx, "dem_failure"] = "No DEM raster available"

        # Slope
        if "slope" in variables and pd.isna(gdf.at[idx, "slope"]) and tr.get("slope"):
            try:
                slope_value = extract_value(tr["slope"], lat, lon)
                if slope_value is not None:
                    gdf.at[idx, "slope"] = slope_value
                    stats["successful_slope"] += 1
                else:
                    gdf.at[idx, "slope_failure"] = "No data value or out of bounds"
            except Exception as e:
                gdf.at[idx, "slope_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif "slope" in variables and pd.isna(gdf.at[idx, "slope"]):
            gdf.at[idx, "slope_failure"] = "No slope raster available"

        # Aspect
        if "aspect" in variables and pd.isna(gdf.at[idx, "aspect"]) and tr.get("aspect"):
            try:
                aspect_value = extract_value(tr["aspect"], lat, lon)
                if aspect_value is not None:
                    gdf.at[idx, "aspect"] = aspect_value
                    stats["successful_aspect"] += 1
                else:
                    gdf.at[idx, "aspect_failure"] = "No data value or out of bounds"
            except Exception as e:
                gdf.at[idx, "aspect_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif "aspect" in variables and pd.isna(gdf.at[idx, "aspect"]):
            gdf.at[idx, "aspect_failure"] = "No aspect raster available"

        # Geomorphon
        if "geomorphons" in variables and pd.isna(gdf.at[idx, "geomorphon"]) and tr.get("geomorphon"):
            try:
                geomorph_value = extract_value(tr["geomorphon"], lat, lon)
                if geomorph_value is not None:
                    gdf.at[idx, "geomorphon"] = geomorph_value
                    gdf.at[idx, "geomorphon_class"] = geomorph_classes.get(geomorph_value, None)
                    stats["successful_geomorphon"] += 1
                else:
                    gdf.at[idx, "geomorphon_failure"] = "No data value or out of bounds"
            except Exception as e:
                gdf.at[idx, "geomorphon_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif "geomorphons" in variables and pd.isna(gdf.at[idx, "geomorphon"]):
            gdf.at[idx, "geomorphon_failure"] = "No geomorphon raster available"

    # Save enriched GeoJSON
    gdf.to_file(output_geojson, driver="GeoJSON")
    logger.info(f"✅ Done! Saved {output_geojson}")
    logger.info(f"Stats: {stats}")
