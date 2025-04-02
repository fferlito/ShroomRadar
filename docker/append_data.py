import os
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import rioxarray
from rasterstats import zonal_stats
from tqdm import tqdm
from icecream import ic
from concurrent.futures import ProcessPoolExecutor

def process_raster(data_file, polygon_gdf):
    try:
        with rioxarray.open_rasterio(data_file) as src:
            zs = zonal_stats(polygon_gdf, src[0].data, affine=src.rio.transform(), stats='mean')
    except ValueError as e:
        if 'unable to decode time units' in str(e):
            ic(f"Decoding time units issue with {data_file}. Trying with decode_times=False")
            with rioxarray.open_rasterio(data_file, decode_times=False) as src:
                zs = zonal_stats(polygon_gdf, src[0].data, affine=src.rio.transform(), stats='mean')
        else:
            raise e
    return [stat['mean'] if stat['mean'] is not None else np.nan for stat in zs]

def get_average_environmental_data(polygon_gdf, date, data_dir, variables, num_days, num_threads):
    mean_values = {var: [] for var in variables}
    tasks = []
    
    with ProcessPoolExecutor(max_workers=num_threads) as executor:
        for i in tqdm(range(num_days), desc="Processing days"):
            current_date = date - timedelta(days=i)
            file_date_str = current_date.strftime('%Y') + str(current_date.timetuple().tm_yday).zfill(3)
            for variable in variables:
                data_file = os.path.join(data_dir, variable, "Daily", f"{file_date_str}.nc")
                if not os.path.isfile(data_file):
                    ic(f"File not found for {variable} on {file_date_str}")
                    mean_values[variable].extend([np.nan] * len(polygon_gdf))
                    continue
                tasks.append((variable, executor.submit(process_raster, data_file, polygon_gdf)))

        for variable, future in tqdm(tasks, desc="Processing tasks"):
            mean_values[variable].extend(future.result())

    return mean_values

def generate_input_model(input_geojson, output_geojson, data_dir: str = "data/environmental_data/NRT", num_days: int = 14, num_threads: int = 24):
    gdf = gpd.read_file(input_geojson)
    gdf = gdf.to_crs('EPSG:4326')
    test_date = datetime.today() - timedelta(days=1)
    variables = ['P', 'Pres', 'RelHum', 'SpecHum', 'Temp', 'Tmax', 'Tmin']

    ic("Starting to calculate average environmental data...")
    mean_values = get_average_environmental_data(gdf, test_date, data_dir, variables, num_days, num_threads)

    for variable, values in tqdm(mean_values.items(), desc="Adding results to GeoDataFrame"):
        for day_index in range(num_days):
            gdf[f'{variable}_{day_index+1}'] = values[day_index::num_days]

    output_file = output_geojson
    gdf.to_file(output_file, driver='GeoJSON')
    ic("Data has been updated and saved to", output_file)