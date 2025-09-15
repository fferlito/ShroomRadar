import os
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import xarray as xr
import rioxarray
from rasterstats import zonal_stats
from tqdm import tqdm
from icecream import ic
from concurrent.futures import ProcessPoolExecutor


def process_raster(data_file, polygon_gdf):
    """Open NetCDF with xarray (CF-aware) and compute polygon means."""
    try:
        with xr.open_dataset(data_file, engine="netcdf4", decode_cf=True) as ds:
            # find first real variable
            var_candidates = [v for v in ds.data_vars
                              if v.lower() not in ("lat", "latitude", "lon", "longitude", "time")]
            if not var_candidates:
                return [np.nan] * len(polygon_gdf)

            vname = var_candidates[0]
            da = ds[vname]

            # standardize dimension names
            rename_map = {}
            if "latitude" in da.dims: rename_map["latitude"] = "lat"
            if "longitude" in da.dims: rename_map["longitude"] = "lon"
            if rename_map:
                da = da.rename(rename_map)

            # select first time step if present
            if "time" in da.dims:
                da = da.isel(time=0)

            # must have lat/lon dims
            if not (("lat" in da.dims) and ("lon" in da.dims)):
                return [np.nan] * len(polygon_gdf)

            # write CRS and spatial dims
            da = da.rio.write_crs("EPSG:4326")
            da = da.rio.set_spatial_dims(x_dim="lon", y_dim="lat")

            # zonal stats on polygons
            zs = zonal_stats(polygon_gdf, da.values, affine=da.rio.transform(), stats="mean")
            return [stat["mean"] if stat["mean"] is not None else np.nan for stat in zs]

    except Exception as e:
        ic(f"⚠️ Error processing {data_file}: {e}")
        return [np.nan] * len(polygon_gdf)


def get_average_environmental_data(polygon_gdf, date, base_dir, variables, num_days, num_threads):
    mean_values = {var: [] for var in variables}
    tasks = []

    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for i in tqdm(range(num_days), desc="Processing days"):
            current_date = date - timedelta(days=i)
            file_date_str = current_date.strftime("%Y") + str(current_date.timetuple().tm_yday).zfill(3)

            for variable in variables:
                nrt_file = os.path.join(base_dir, "NRT", variable, "Daily", f"{file_date_str}.nc")
                past_file = os.path.join(base_dir, "Past", variable, "Daily", f"{file_date_str}.nc")

                if os.path.isfile(nrt_file):
                    data_file = nrt_file
                elif os.path.isfile(past_file):
                    data_file = past_file
                else:
                    ic(f"File not found for {variable} on {file_date_str}")
                    mean_values[variable].extend([np.nan] * len(polygon_gdf))
                    continue

                tasks.append((variable, executor.submit(process_raster, data_file, polygon_gdf)))

        # collect results
        for variable, future in tqdm(tasks, desc="Processing tasks"):
            mean_values[variable].extend(future.result())

    return mean_values


def generate_input_model(input_geojson, output_geojson,
                         data_dir: str = "climate_data", num_days: int = 14, num_threads: int = 8):
    gdf = gpd.read_file(input_geojson).to_crs("EPSG:4326")
    test_date = datetime.today() - timedelta(days=1)
    variables = ["P", "Pres", "RelHum", "SpecHum", "Temp", "Tmax", "Tmin"]

    ic("Starting to calculate average environmental data...")
    mean_values = get_average_environmental_data(gdf, test_date, data_dir, variables, num_days, num_threads)

    for variable, values in tqdm(mean_values.items(), desc="Adding results to GeoDataFrame"):
        for day_index in range(num_days):
            gdf[f"{variable}_{day_index+1}"] = values[day_index::num_days]

    gdf.to_file(output_geojson, driver="GeoJSON")
    ic("✅ Data has been updated and saved to", output_geojson)
