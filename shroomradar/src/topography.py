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

def run_whitebox(
    tif_file: str,
    need_slope: bool = False,
    need_aspect: bool = False,
    need_geomorph: bool = False,
    slope_dir: str | None = None,
    aspect_dir: str | None = None,
    geomorph_dir: str | None = None,
) -> tuple[str, str | None, str | None, str | None]:
    """
    Run WhiteboxTools to generate slope, aspect, and geomorphon rasters from a DEM.

    This function takes a DEM file and selectively runs WhiteboxTools processes
    to create derived raster products. It checks if the output files already
    exist and are valid before running the tools, avoiding redundant processing.

    Args:
        tif_file (str): Path to the input DEM GeoTIFF file.
        need_slope (bool, optional): Flag to generate a slope raster.
            Defaults to False.
        need_aspect (bool, optional): Flag to generate an aspect raster.
            Defaults to False.
        need_geomorph (bool, optional): Flag to generate a geomorphons raster.
            Defaults to False.
        slope_dir (str | None, optional): Directory to save the slope raster.
            If None, saved next to the input DEM. Defaults to None.
        aspect_dir (str | None, optional): Directory to save the aspect raster.
            If None, saved next to the input DEM. Defaults to None.
        geomorph_dir (str | None, optional): Directory to save the geomorphons raster.
            If None, saved next to the input DEM. Defaults to None.

    Returns:
        tuple[str, str | None, str | None, str | None]: A tuple containing the
            absolute path to the input DEM, and the paths to the generated slope,
            aspect, and geomorphon rasters. If a raster was not generated or is
            invalid, its path will be None.
    """
    tif_file = os.path.abspath(tif_file).replace("\\", "/")
    base_name = os.path.splitext(os.path.basename(tif_file))[0]

    # Create output paths in subfolders
    slope_tif = (
        os.path.join(slope_dir, f"{base_name}_slope.tif")
        if slope_dir
        else f"{os.path.splitext(tif_file)[0]}_slope.tif"
    )
    aspect_tif = (
        os.path.join(aspect_dir, f"{base_name}_aspect.tif")
        if aspect_dir
        else f"{os.path.splitext(tif_file)[0]}_aspect.tif"
    )
    geomorph_tif = (
        os.path.join(geomorph_dir, f"{base_name}_geomorph.tif")
        if geomorph_dir
        else f"{os.path.splitext(tif_file)[0]}_geomorph.tif"
    )

    if need_slope and not valid_raster(slope_tif):
        wbt.slope(dem=tif_file, output=slope_tif, zfactor=1.0, units="degrees")
    if need_aspect and not valid_raster(aspect_tif):
        wbt.aspect(dem=tif_file, output=aspect_tif)
    if need_geomorph and not valid_raster(geomorph_tif):
        wbt.geomorphons(
            dem=tif_file, output=geomorph_tif, search=50, threshold=0.0, forms=True
        )
    return (
        tif_file,
        slope_tif if valid_raster(slope_tif) else None,
        aspect_tif if valid_raster(aspect_tif) else None,
        geomorph_tif if valid_raster(geomorph_tif) else None,
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
    if not valid_raster(raster):
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



def enrich_csv(
    input_csv: str,
    output_csv: str,
    out_dir: str = "dem_tiles",
    download_tiles: bool = True,
    variables: list[str] = ["dem", "slope", "aspect", "geomorphons"],
    verbose: bool = True,
) -> None:
    """
    Enriches a CSV file with topographic data derived from Digital Elevation Models (DEMs).

    This function processes a CSV file containing latitude and longitude columns ('y', 'x').
    For each row, it identifies the required DEM tile, downloads it if necessary,
    calculates specified topographic variables (DEM, slope, aspect, geomorphons),
    and appends these values to the corresponding row in a new output CSV.

    It is optimized to minimize downloads and processing by first scanning the entire
    CSV to determine which unique tiles are needed.

    Args:
        input_csv (str): Path to the input CSV file. Must contain 'x' (longitude)
                         and 'y' (latitude) columns.
        output_csv (str): Path to save the enriched CSV file.
        out_dir (str, optional): Directory to store downloaded and processed
                                 raster tiles. Defaults to "dem_tiles".
        download_tiles (bool, optional): If True, automatically downloads missing
                                         DEM tiles. If False, only uses existing
                                         local tiles. Defaults to True.
        variables (list[str], optional): A list of variables to extract.
                                         Valid options are 'dem', 'slope',
                                         'aspect', and 'geomorphons'.
                                         Defaults to all four.
        verbose (bool, optional): If True, print progress and informational
                                  messages to the console. Defaults to True.

    Raises:
        FileNotFoundError: If the input CSV file does not exist.
        ValueError: If the CSV is missing 'x' or 'y' columns, or if `variables`
                    contains invalid options.
    """


    if verbose:
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )
        logger = logging.getLogger(__name__)
    else:
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.WARNING)

    os.makedirs(out_dir, exist_ok=True)

    # Create subfolders for different file types
    dem_dir = os.path.join(out_dir, "dem")
    slope_dir = os.path.join(out_dir, "slope")
    aspect_dir = os.path.join(out_dir, "aspect")
    geomorph_dir = os.path.join(out_dir, "geomorphons")

    os.makedirs(dem_dir, exist_ok=True)
    os.makedirs(slope_dir, exist_ok=True)
    os.makedirs(aspect_dir, exist_ok=True)
    os.makedirs(geomorph_dir, exist_ok=True)

    # Derive individual boolean flags from variables list
    generate_dem = "dem" in variables
    generate_slope = "slope" in variables
    generate_aspect = "aspect" in variables
    generate_geomorphons = "geomorphons" in variables

    # Validate variables list
    valid_variables = ["dem", "slope", "aspect", "geomorphons"]
    invalid_vars = [var for var in variables if var not in valid_variables]
    if invalid_vars:
        raise ValueError(
            f"Invalid variables: {invalid_vars}. Valid options are: {valid_variables}"
        )

    # Validate input file
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input CSV file not found: {input_csv}")

    logger.info(f"Loading CSV: {input_csv}")
    df = pd.read_csv(input_csv)
    df = df[df["y"].between(-56, 60)]

    logger.info(f"Loaded {len(df)} rows")

    # Validate required columns
    required_cols = ["x", "y"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Initialize columns based on requested variables
    base_cols = []
    if generate_dem:
        base_cols.extend(["dem", "dem_source"])
    if generate_slope:
        base_cols.append("slope")
    if generate_aspect:
        base_cols.append("aspect")
    if generate_geomorphons:
        base_cols.extend(["geomorphon", "geomorphon_class"])

    for col in base_cols:
        if col not in df.columns:
            df[col] = None

    # Add failure tracking columns for requested variables
    failure_cols = []
    if generate_dem:
        failure_cols.append("dem_failure")
    if generate_slope:
        failure_cols.append("slope_failure")
    if generate_aspect:
        failure_cols.append("aspect_failure")
    if generate_geomorphons:
        failure_cols.append("geomorphon_failure")

    for col in failure_cols:
        if col not in df.columns:
            df[col] = None

    # Statistics tracking
    stats = {
        "total_rows": len(df),
        "invalid_coords": 0,
        "missing_tiles": 0,
        "extraction_failures": 0,
    }

    # Add variable-specific stats based on requested variables
    if generate_dem:
        stats["successful_dem"] = 0
    if generate_slope:
        stats["successful_slope"] = 0
    if generate_aspect:
        stats["successful_aspect"] = 0
    if generate_geomorphons:
        stats["successful_geomorphon"] = 0

    # Step 1: Collect per-tile needs
    logger.info("Step 1: Scanning CSV for tile requirements...")
    tile_needs = {}
    invalid_coord_rows = []

    for i, row in tqdm(df.iterrows(), total=len(df), desc="Scanning CSV"):
        lat, lon = row["y"], row["x"]

        # Check for invalid coordinates
        if pd.isna(lat) or pd.isna(lon):
            invalid_coord_rows.append(i)
            df.at[i, "dem_failure"] = "Invalid coordinates: NaN values"
            stats["invalid_coords"] += 1
            continue

        try:
            tid = tile_id_from_coords(lat, lon)
            if tid is None:
                invalid_coord_rows.append(i)
                df.at[i, "dem_failure"] = f"Invalid coordinates: lat={lat}, lon={lon}"
                stats["invalid_coords"] += 1
                continue
        except Exception as e:
            invalid_coord_rows.append(i)
            df.at[i, "dem_failure"] = f"Coordinate conversion error: {str(e)}"
            stats["invalid_coords"] += 1
            continue

        base = os.path.join(dem_dir, tid)
        slope_file = os.path.join(slope_dir, f"{tid}_slope.tif")
        aspect_file = os.path.join(aspect_dir, f"{tid}_aspect.tif")
        geomorph_file = os.path.join(geomorph_dir, f"{tid}_geomorph.tif")

        if tid not in tile_needs:
            tile_needs[tid] = {
                "dem": False,
                "slope": False,
                "aspect": False,
                "geomorphon": False,
            }

        if generate_dem and pd.isna(row.get("dem")) and not valid_raster(f"{base}.tif"):
            tile_needs[tid]["dem"] = True
        if (
            generate_slope
            and pd.isna(row.get("slope"))
            and not valid_raster(slope_file)
        ):
            tile_needs[tid]["slope"] = True
        if (
            generate_aspect
            and pd.isna(row.get("aspect"))
            and not valid_raster(aspect_file)
        ):
            tile_needs[tid]["aspect"] = True
        if (
            generate_geomorphons
            and pd.isna(row.get("geomorphon"))
            and not valid_raster(geomorph_file)
        ):
            tile_needs[tid]["geomorphon"] = True

    tile_needs = {
        tid: needs for tid, needs in tile_needs.items() if any(needs.values())
    }
    logger.info(f"Found {len(tile_needs)} tiles that need processing")

    # Step 2 & 3: Only run if download_tiles is True
    downloaded = {}
    tile_results = {}
    if download_tiles:
        logger.info("Step 2: Downloading and preparing tiles...")
        for tid, needs in tqdm(tile_needs.items(), desc="Preparing tiles"):
            local_tif = os.path.join(dem_dir, f"{tid}.tif")
            if valid_raster(local_tif):
                downloaded[tid] = ([local_tif], "Local")
                continue

            m = re.match(r"([NS])(\d{2})([EW])(\d{3})", tid)
            if not m:
                logger.warning(f"Could not parse tile ID: {tid}")
                continue

            try:
                lat0 = int(m.group(2)) * (1 if m.group(1) == "N" else -1)
                lon0 = int(m.group(4)) * (1 if m.group(3) == "E" else -1)
                zip_paths, source = download_dem_point(
                    lat0 + 0.5, lon0 + 0.5, out_dir=out_dir
                )
                if zip_paths:
                    tifs = [prepare_tif(zp) for zp in zip_paths]
                    # Move processed files to dem subfolder
                    moved_tifs = []
                    for tif in tifs:
                        target_path = os.path.join(dem_dir, f"{tid}.tif")
                        if tif != target_path:
                            import shutil

                            shutil.move(tif, target_path)
                        moved_tifs.append(target_path)
                    downloaded[tid] = (moved_tifs, source)
                else:
                    logger.warning(f"No DEM data available for tile {tid}")
                    stats["missing_tiles"] += 1
            except Exception as e:
                logger.error(f"Error downloading tile {tid}: {str(e)}")
                stats["missing_tiles"] += 1

        logger.info("Step 3: Running Whitebox processing...")
        for tid, (tifs, source) in tqdm(downloaded.items(), desc="Running Whitebox"):
            needs = tile_needs.get(tid, {})
            for tif in tifs:
                try:
                    tif_path, slope_path, aspect_path, geomorph_path = run_whitebox(
                        tif,
                        need_slope=generate_slope and needs.get("slope", False),
                        need_aspect=generate_aspect and needs.get("aspect", False),
                        need_geomorph=generate_geomorphons
                        and needs.get("geomorphon", False),
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

    # Step 4: Extract values from whatever exists
    logger.info("Step 4: Extracting values from rasters...")
    for i, row in tqdm(df.iterrows(), total=len(df), desc="Extracting values"):
        # Skip rows with invalid coordinates
        if i in invalid_coord_rows:
            continue

        lat, lon = row["y"], row["x"]
        tid = tile_id_from_coords(lat, lon)
        if tid is None:
            continue

        dem_base = os.path.join(dem_dir, tid)
        slope_base = os.path.join(slope_dir, f"{tid}_slope.tif")
        aspect_base = os.path.join(aspect_dir, f"{tid}_aspect.tif")
        geomorph_base = os.path.join(geomorph_dir, f"{tid}_geomorph.tif")

        tr = tile_results.get(
            tid,
            {
                "tif": f"{dem_base}.tif" if valid_raster(f"{dem_base}.tif") else None,
                "slope": slope_base if valid_raster(slope_base) else None,
                "aspect": aspect_base if valid_raster(aspect_base) else None,
                "geomorphon": geomorph_base if valid_raster(geomorph_base) else None,
                "source": "Local",
            },
        )

        # Extract DEM value
        if generate_dem and pd.isna(row.get("dem")) and tr["tif"]:
            try:
                dem_value = extract_value(tr["tif"], lat, lon)
                if dem_value is not None:
                    df.at[i, "dem"] = dem_value
                    df.at[i, "dem_source"] = tr["source"]
                    stats["successful_dem"] += 1
                else:
                    df.at[i, "dem_failure"] = "No data value or out of bounds"
            except Exception as e:
                df.at[i, "dem_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif generate_dem and pd.isna(row.get("dem")):
            df.at[i, "dem_failure"] = "No DEM raster available"

        # Extract slope value
        if generate_slope and pd.isna(row.get("slope")) and tr["slope"]:
            try:
                slope_value = extract_value(tr["slope"], lat, lon)
                if slope_value is not None:
                    df.at[i, "slope"] = slope_value
                    stats["successful_slope"] += 1
                else:
                    df.at[i, "slope_failure"] = "No data value or out of bounds"
            except Exception as e:
                df.at[i, "slope_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif generate_slope and pd.isna(row.get("slope")):
            df.at[i, "slope_failure"] = "No slope raster available"

        # Extract aspect value
        if generate_aspect and pd.isna(row.get("aspect")) and tr["aspect"]:
            try:
                aspect_value = extract_value(tr["aspect"], lat, lon)
                if aspect_value is not None:
                    df.at[i, "aspect"] = aspect_value
                    stats["successful_aspect"] += 1
                else:
                    df.at[i, "aspect_failure"] = "No data value or out of bounds"
            except Exception as e:
                df.at[i, "aspect_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif generate_aspect and pd.isna(row.get("aspect")):
            df.at[i, "aspect_failure"] = "No aspect raster available"

        # Extract geomorphon value
        if generate_geomorphons and pd.isna(row.get("geomorphon")) and tr["geomorphon"]:
            try:
                geomorph_value = extract_value(tr["geomorphon"], lat, lon)
                if geomorph_value is not None:
                    df.at[i, "geomorphon"] = geomorph_value
                    stats["successful_geomorphon"] += 1
                else:
                    df.at[i, "geomorphon_failure"] = "No data value or out of bounds"
            except Exception as e:
                df.at[i, "geomorphon_failure"] = f"Extraction error: {str(e)}"
                stats["extraction_failures"] += 1
        elif generate_geomorphons and pd.isna(row.get("geomorphon")):
            df.at[i, "geomorphon_failure"] = "No geomorphon raster available"

    # Step 5: Decode geomorphons (only if geomorphons were generated)
    if generate_geomorphons:
        logger.info("Step 5: Decoding geomorphon classes...")
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
        df["geomorphon_class"] = df["geomorphon"].map(geomorph_classes)

    # Save results
    df.to_csv(output_csv, index=False)

    return


def enrich_geojson(
    input_geojson: str,
    output_geojson: str,
    out_dir: str = "dem_tiles",
    download_tiles: bool = True,
) -> None:
    """
    Enriches a GeoJSON file with topographic data derived from DEMs.

    This function processes a GeoJSON file, calculating the centroid for each feature.
    For each centroid, it identifies the required DEM tile, downloads it if necessary,
    calculates topographic variables (DEM, slope, aspect, geomorphons), and
    appends these values as properties to the corresponding feature in a new
    output GeoJSON file.

    Args:
        input_geojson (str): Path to the input GeoJSON file.
        output_geojson (str): Path to save the enriched GeoJSON file.
        out_dir (str, optional): Directory to store downloaded and processed
                                 raster tiles. Defaults to "dem_tiles".
        download_tiles (bool, optional): If True, automatically downloads missing
                                         DEM tiles. If False, only uses existing
                                         local tiles. Defaults to True.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Create subfolders
    dem_dir = os.path.join(out_dir, "dem")
    slope_dir = os.path.join(out_dir, "slope")
    aspect_dir = os.path.join(out_dir, "aspect")
    geomorph_dir = os.path.join(out_dir, "geomorphons")
    for d in [dem_dir, slope_dir, aspect_dir, geomorph_dir]:
        os.makedirs(d, exist_ok=True)

    gdf = gpd.read_file(input_geojson)

    # Ensure WGS84
    if gdf.crs is None:
        print("⚠️ No CRS found, assuming EPSG:4326")
        gdf.set_crs(epsg=4326, inplace=True)
    else:
        gdf = gdf.to_crs(epsg=4326)

    # Filter SRTM coverage (lat -56 to 60)
    gdf = gdf[gdf.geometry.centroid.y.between(-56, 60)]

    # Add expected cols
    for col in [
        "dem",
        "slope",
        "aspect",
        "geomorphon",
        "dem_source",
        "geomorphon_class",
    ]:
        if col not in gdf.columns:
            gdf[col] = None

    # Collect centroids
    centroids = gdf.geometry.centroid
    coords = [(pt.y, pt.x) for pt in centroids]

    # Step 1: collect tiles
    needed_tiles = {}
    for lat, lon in tqdm(coords, desc="Collecting tiles"):
        tid = tile_id_from_coords(lat, lon)
        if tid and tid not in needed_tiles:
            needed_tiles[tid] = (lat, lon)

    # Step 2: download & prepare tiles
    downloaded = {}
    for tid, (lat, lon) in tqdm(needed_tiles.items(), desc="Preparing tiles"):
        tif_path = os.path.join(dem_dir, f"{tid}.tif")
        if valid_raster(tif_path):
            downloaded[tid] = ([tif_path], "Local")
        elif download_tiles:
            zip_paths, source = download_dem_point(lat, lon, out_dir=out_dir)
            if zip_paths:
                tifs = [prepare_tif(zp) for zp in zip_paths]
                moved_tifs = []
                for tif in tifs:
                    target = os.path.join(dem_dir, f"{tid}.tif")
                    if tif != target:
                        shutil.move(tif, target)
                    moved_tifs.append(target)
                downloaded[tid] = (moved_tifs, source)

    # Step 3: run Whitebox
    tile_results = {}
    for tid, (tifs, source) in tqdm(downloaded.items(), desc="Running Whitebox"):
        for tif in tifs:
            tif_path, slope_tif, aspect_tif, geomorph_tif = run_whitebox(
                tif, slope_dir, aspect_dir, geomorph_dir
            )
            tile_results[tid] = (tif_path, slope_tif, aspect_tif, geomorph_tif, source)

    # Step 4: extract values
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

    for idx, (lat, lon) in enumerate(tqdm(coords, desc="Extracting values")):
        tid = tile_id_from_coords(lat, lon)
        if tid is None or tid not in tile_results:
            continue
        tif, slope_tif, aspect_tif, geomorph_tif, source = tile_results[tid]
        gdf.at[idx, "dem"] = extract_value(tif, lat, lon)
        gdf.at[idx, "dem_source"] = source
        gdf.at[idx, "slope"] = extract_value(slope_tif, lat, lon)
        gdf.at[idx, "aspect"] = extract_value(aspect_tif, lat, lon)
        gdf.at[idx, "geomorphon"] = extract_value(geomorph_tif, lat, lon)
        gdf.at[idx, "geomorphon_class"] = geomorph_classes.get(
            gdf.at[idx, "geomorphon"], None
        )

    # Save enriched GeoJSON
    gdf.to_file(output_geojson, driver="GeoJSON")
    print(f"✅ Done! Saved {output_geojson}")
