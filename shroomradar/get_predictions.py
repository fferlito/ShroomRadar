import os
import geopandas as gpd
import numpy as np
from tqdm import tqdm
import pandas as pd
import joblib
from icecream import ic



def prepare_data_for_prediction(row, climate_variables, num_days):
    """Prepare data for prediction by extracting climate data directly from GeoDataFrame row"""
    # Create features in the EXACT same order as the training data
    training_feature_order = []
    
    # Build features in training order: each variable for all days, then next variable
    for variable in climate_variables:
        for day_number in range(1, num_days + 1):  # 1 to 14 (num_days)
            feature_name = f"{variable}_P{day_number}"
            training_feature_order.append(feature_name)
    
    # Add static variables in the same order as training
    training_feature_order.extend([ 'elevation', 'aspect'])
    
    # Create the variables dictionary in the exact training order
    variables_for_prediction = {}
    for feature_name in training_feature_order:
        if feature_name == 'LC':
            variables_for_prediction[feature_name] = row['mode_value']
        elif feature_name in ['elevation', 'aspect']:
            variables_for_prediction[feature_name] = row[feature_name]
        else:
            # Climate variable
            if feature_name in row:
                variables_for_prediction[feature_name] = row[feature_name]
            else:
                # If column doesn't exist, fill with NaN (will be handled later)
                variables_for_prediction[feature_name] = np.nan
    
    return variables_for_prediction


def generarate_predictions(input_geojson, output_geojson, num_days: int = 14):
    """Generate predictions using climate data already embedded in the GeoJSON from API"""
    gdf = gpd.read_file(input_geojson)
    gdf = gdf.to_crs('EPSG:4326')
    gb_clf = joblib.load('data/models/lr_model_v5.pkl')

    # Climate variables that should be in the GeoJSON (matching model feature names and ORDER)
    climate_variables = ['tmin', 'tmax', 'temp', 'rel_humidity', 'precipitation', 'wind_speed']

    ic(f"🔮 Making predictions for {len(gdf)} grid cells...")
    
    # Sequentially process each row for predictions
    predictions = []
    for index, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Making predictions"):
        try:
            data_dict = prepare_data_for_prediction(row, climate_variables, num_days)
            df = pd.DataFrame(data_dict, index=[0])
            
            # Check for NaN values and handle them
            if df.isnull().any().any():
                df = df.fillna(0)
            
            predicted_species = gb_clf.predict_proba(df)[0][0]  # Make prediction
            predictions.append(predicted_species)
            
        except Exception as e:
            ic(f"❌ Error processing row {index}: {e}")
            predictions.append(0.0)  # Default prediction for failed cases
    
    # Add predictions to the GeoDataFrame
    gdf['species_prediction'] = predictions
    
    # Ensure output directory exists
    output_dir = os.path.dirname(output_geojson)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        print(f"📁 Created output directory: {output_dir}")
    
    # Write the GeoDataFrame with predictions to a new GeoJSON file
    gdf[['geometry', 'species_prediction']].to_file(output_geojson, driver='GeoJSON')
    ic("Predictions have been made and saved to", output_geojson)


def filter_predictions(path_output_geojson, path_filtered_geojson, threshold=0.01):
    # Remove polygons with a species_prediction value lower than 0.0001
    df = gpd.read_file(path_output_geojson)
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