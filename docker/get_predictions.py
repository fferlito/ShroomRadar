import os
import geopandas as gpd
import numpy as np
from datetime import datetime, timedelta
import rasterio
from shapely.geometry import Point
from tqdm import tqdm
import pandas as pd
from joblib import Parallel, delayed
import multiprocessing
import joblib
from icecream import ic
from sklearn.ensemble import GradientBoostingClassifier

def extract_data_from_raster(data_file, polygons):
    values = []
    if not os.path.isfile(data_file):
        print(f"File not found: {data_file}")
        return [np.nan] * len(polygons)
    
    with rasterio.open(data_file, mode="r") as src:
        for polygon in tqdm(polygons, desc=data_file):
            min_x, min_y, max_x, max_y = polygon.bounds
            centroid_x = (min_x + max_x) / 2
            centroid_y = (min_y + max_y) / 2
            centroid = Point(centroid_x, centroid_y)
            px, py = src.index(centroid.x, centroid.y)
            try:
                value = src.read(1, window=((py, py+1), (px, px+1)))
                values.append(value[0, 0])
            except:
                values.append(np.nan)
    
    return values

def get_environmental_data(polygons, date, data_dir, variables, num_days):
    data = {variable: [[] for _ in range(len(polygons))] for variable in variables}
    
    dates = [date - timedelta(days=i) for i in range(num_days)]
    
    for variable in tqdm(variables, desc="Variable"):
        print(f"Processing variable: {variable}")
        for current_date in dates:
            file_date_str = current_date.strftime('%Y') + str(current_date.timetuple().tm_yday).zfill(3)
            data_file = os.path.join(data_dir, variable, "Daily", f"{file_date_str}.nc")
            values = extract_data_from_raster(data_file, polygons)
            for i, value in enumerate(values):
                data[variable][i].append(value)
    
    for variable in variables:
        for i in range(len(polygons)):
            data[variable][i] = data[variable][i][::-1]  # Reverse the list to align with the order needed (recent to past)
    
    return data

def prepare_data_for_prediction(polygon_index, all_data, variables, num_days, row):
    variables_for_prediction = {}
    for variable in variables:
        for day_number in range(num_days):
            variable_name = f"{variable}_{day_number+1}"
            variables_for_prediction[variable_name] = all_data[variable][polygon_index][day_number]
    
    variables_for_prediction['LC'] = row['mode_value']
    variables_for_prediction['elevation'] = row['mean_elevation']
    
    return variables_for_prediction


def generarate_predictions(input_geojson, output_geojson, data_dir: str = "data/environmental_data/NRT", num_days: int = 14, num_threads: int = 24):
    gdf = gpd.read_file(input_geojson)
    gdf = gdf.to_crs('EPSG:4326')
    gb_clf = joblib.load('data/models/gradient_boosting_model.pkl')



    variables = ['Pres', 'P', 'RelHum', 'SpecHum', 'Temp', 'Tmax', 'Tmin']
    test_date = datetime.today() - timedelta(days=1)

    # Prepare polygon geometries
    polygon_geometries = gdf['geometry']

    # Get environmental data for all polygons
    all_data = get_environmental_data(polygon_geometries, test_date, data_dir, variables, num_days)

    # Sequentially process each row for predictions
    predictions = []
    for index, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Making predictions"):
        data_dict = prepare_data_for_prediction(index, all_data, variables, num_days, row)
        df = pd.DataFrame(data_dict, index=[0])
        predicted_species = gb_clf.predict_proba(df)[0][0]  # Make prediction
        print(predicted_species)
        predictions.append(predicted_species)

    # Add predictions to the GeoDataFrame
    gdf['species_prediction'] = predictions
    # Write the GeoDataFrame with predictions to a new GeoJSON file
    gdf[['geometry', 'species_prediction']].to_file(output_geojson, driver='GeoJSON')
    ic("Predictions have been made and saved to", output_geojson)


def filter_predictions(path_output_geojson, path_filtered_geojson, threshold=0.01):
    # Remove polygons with a species_prediction value lower than 0.0001
    df = gpd.read_file(path_output_geojson)
    threshold = 0.005
    initial_count = len(df)
    df = df[df['species_prediction'] >= threshold]
    removed_count = initial_count - len(df)
    df = df[['geometry', 'species_prediction']]


    # Print the number of polygons removed
    ic(f"Number of polygons removed: {removed_count}")
    ic(f"Number of polygons left: {len(df)}")

    # Write the GeoDataFrame with predictions to a new GeoJSON file
    df.to_file(path_filtered_geojson, driver='GeoJSON')
    ic("Predictions have been made and saved to", path_filtered_geojson)