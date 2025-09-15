import os
import geopandas as gpd
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import requests
import time
from tqdm import tqdm
from icecream import ic

def get_weather_data(lat, lon, start_date, end_date):
    """
    Fetch historical weather data from Open-Meteo API (FREE!)
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': [
            'temperature_2m_min',
            'temperature_2m_max', 
            'temperature_2m_mean',
            'relative_humidity_2m_mean',
            'precipitation_sum',
            'wind_speed_10m_max'
        ],
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        ic(f"Error fetching weather data: {e}")
        return None

def get_15day_weather_history(lat, lon, obs_date, num_days=14):
    """
    Get 15-day weather history for a specific location and observation date
    Following the exact pattern from append_openweather_climate.ipynb
    """
    
    # Parse the observation date
    if isinstance(obs_date, str):
        # Handle different date formats
        try:
            if 'T' in obs_date or '+' in obs_date:
                # ISO format with timezone
                obs_date = pd.to_datetime(obs_date).date()
            else:
                obs_date = pd.to_datetime(obs_date).date()
        except:
            ic(f"⚠️ Could not parse date: {obs_date}")
            return None
    else:
        obs_date = obs_date
    
    # Calculate date range (15 days ending on observation date)
    end_date = obs_date
    start_date = obs_date - timedelta(days=num_days)
    
    # Get weather data
    weather_data = get_weather_data(lat, lon, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
    
    if weather_data and 'daily' in weather_data:
        daily = weather_data['daily']
        
        # Create result dictionary with variable_1, variable_2, etc. columns (matching NetCDF structure)
        result = {}
        
        # Map Open-Meteo API variables to model feature names (matching training data)
        api_to_model = {
            'precipitation_sum': 'precipitation',
            'relative_humidity_2m_mean': 'rel_humidity',
            'temperature_2m_mean': 'temp',
            'temperature_2m_max': 'tmax',
            'temperature_2m_min': 'tmin',
            'wind_speed_10m_max': 'wind_speed'
        }
        
        for api_var, model_var in api_to_model.items():
            if api_var in daily:
                values = daily[api_var]
                # Reverse order so day 1 is most recent (observation day), day 14 is 13 days ago
                values = list(reversed(values))
                
                for i in range(min(num_days, len(values))):
                    col_name = f"{model_var}_P{i+1}"
                    result[col_name] = values[i] if i < len(values) else np.nan
        
        # Add missing variables as NaN since they're not available in Open-Meteo
        # Note: Pressure and specific humidity are not available in Open-Meteo free tier
        # These were not used in the final model according to training data
        
        return result
    
    return None

def generate_input_model(input_geojson, output_geojson, num_days: int = 14, delay_between_requests: float = 0.1):
    """
    Generate input model data by adding climate variables to GeoJSON using Open-Meteo API
    Following the exact pattern from append_openweather_climate.ipynb but for GeoJSON files
    """
    # Load GeoJSON file
    gdf = gpd.read_file(input_geojson)
    gdf = gdf.to_crs('EPSG:4326')
    
    # Use a recent past date that has weather data available (e.g., 30 days ago)
    test_date = datetime.today() - timedelta(days=30)
    
    # Initialize all climate columns with NaN using model feature names and ORDER
    model_variables = ['tmin', 'tmax', 'temp', 'rel_humidity', 'precipitation', 'wind_speed']
    for var in model_variables:
        for day in range(1, num_days + 1):  # 1 to 14 (num_days)
            col_name = f"{var}_P{day}"
            gdf[col_name] = np.nan
    
    ic(f"🌦️ Adding climate data for {len(gdf)} grid cells...")
    ic(f"📊 This will add {len(model_variables) * num_days} = {len(model_variables) * num_days} new columns")
    
    # Track progress
    successful = 0
    failed = 0
    
    # Process each grid cell
    for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="Fetching climate data"):
        try:
            # Extract coordinates from geometry (centroid for polygons, direct for points)
            if gdf.geometry.iloc[idx].geom_type == 'Point':
                lon, lat = gdf.geometry.iloc[idx].x, gdf.geometry.iloc[idx].y
            else:
                # For polygons, use centroid
                centroid = gdf.geometry.iloc[idx].centroid
                lon, lat = centroid.x, centroid.y
            
            # Get climate data for this location
            climate_data = get_15day_weather_history(lat, lon, test_date.strftime('%Y-%m-%d'), num_days)
            
            if climate_data:
                # Add climate data to the row
                for col_name, value in climate_data.items():
                    if col_name in gdf.columns:
                        gdf.at[idx, col_name] = value
                successful += 1
            else:
                failed += 1
                if failed <= 5:  # Only print first few failures
                    ic(f"⚠️ Failed to get climate for row {idx}: lat={lat}, lon={lon}")
            
            # Be respectful to the API
            if delay_between_requests > 0:
                time.sleep(delay_between_requests)
                
        except Exception as e:
            failed += 1
            if failed <= 5:
                ic(f"❌ Error processing row {idx}: {e}")
    
    ic(f"✅ Climate data fetching complete!")
    ic(f"📈 Successful: {successful}")
    ic(f"❌ Failed: {failed}")
    ic(f"📊 Success rate: {successful/(successful+failed)*100:.1f}%")
    
    # Save the updated GeoJSON
    output_file = output_geojson
    gdf.to_file(output_file, driver='GeoJSON')
    ic("Data has been updated and saved to", output_file)