import requests
import numpy as np
from datetime import datetime, timedelta
import time
import csv

class FreeClimateData:
    def __init__(self):
        """
        Initialize Open-Meteo API client - completely free, no API key required!
        """
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
    def get_environmental_data(self, coord, date, num_days=14):
        """
        Get environmental data for a given coordinate and date using Open-Meteo API (FREE!)
        
        Args:
            coord: tuple (longitude, latitude)
            date: datetime object for the observation date
            num_days: number of days to look back (default 14)
            
        Returns:
            list: List of dictionaries with climate variables for each day
        """
        # Calculate date range
        end_date = date
        start_date = date - timedelta(days=num_days-1)
        
        # Format dates for API
        start_date_str = start_date.strftime('%Y-%m-%d')
        end_date_str = end_date.strftime('%Y-%m-%d')
        
        # Make single API request for all days
        weather_data = self._fetch_weather_data(coord[1], coord[0], start_date_str, end_date_str)
        
        data = []
        if weather_data and 'daily' in weather_data:
            daily_data = weather_data['daily']
            
            # Process each day (reverse order to match your NetCDF approach - recent to past)
            for i in range(len(daily_data['time'])):
                day_values = {
                    'P': daily_data['precipitation_sum'][i] if 'precipitation_sum' in daily_data else np.nan,
                    'Pres': daily_data['surface_pressure'][i] if 'surface_pressure' in daily_data else np.nan,
                    'RelHum': daily_data['relative_humidity_2m_mean'][i] if 'relative_humidity_2m_mean' in daily_data else np.nan,
                    'Temp': daily_data['temperature_2m_mean'][i] if 'temperature_2m_mean' in daily_data else np.nan,
                    'Tmax': daily_data['temperature_2m_max'][i] if 'temperature_2m_max' in daily_data else np.nan,
                    'Tmin': daily_data['temperature_2m_min'][i] if 'temperature_2m_min' in daily_data else np.nan,
                    'Wind': daily_data['wind_speed_10m_max'][i] if 'wind_speed_10m_max' in daily_data else np.nan
                }
                data.append(day_values)
            
            # Reverse to match NetCDF approach (most recent day first)
            data.reverse()
        else:
            # If API call fails, return NaN values for all days
            for i in range(num_days):
                data.append({
                    'P': np.nan,
                    'Pres': np.nan,
                    'RelHum': np.nan,
                    'Temp': np.nan,
                    'Tmax': np.nan,
                    'Tmin': np.nan,
                    'Wind': np.nan
                })
        
        return data
    
    def _fetch_weather_data(self, lat, lon, start_date, end_date):
        """
        Fetch weather data from Open-Meteo API (FREE!)
        """
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
                'surface_pressure',
                'wind_speed_10m_max'
            ],
            'timezone': 'auto'
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weather data: {e}")
            return None
    
    def append_climate_data_to_csv(self, input_csv, output_csv, coord_column='location', date_column='observed_on'):
        """
        Append climate data to existing CSV file (similar to your current script)
        """
        # Read the CSV file
        data = []
        with open(input_csv, "r") as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if coord_column in row and row[coord_column]:
                    data.append(row)
        
        # Process each row
        for idx, row in enumerate(data):
            print(f"Processing row {idx+1}/{len(data)}")
            
            # Extract coordinates
            coords = self._extract_coordinates(row[coord_column])
            if coords is None:
                print(f"Skipping row {idx+1}: Missing or invalid coordinates")
                continue
            
            # Parse date
            observed_date = self._parse_datetime(row[date_column])
            
            # Get climate data
            climate_data = self.get_environmental_data(coords, observed_date)
            
            # Add to row (same format as NetCDF approach)
            variables = ['P', 'Pres', 'RelHum', 'Temp', 'Tmax', 'Tmin', 'Wind']
            for variable in variables:
                for i, day_data in enumerate(climate_data):
                    row[f"{variable}_{i+1}"] = day_data[variable]
        
        # Write updated data
        if data:
            fieldnames = list(data[0].keys())
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            print(f"Data updated and saved to {output_csv}")
    
    def _extract_coordinates(self, coord_str):
        """Extract coordinates from string format"""
        if not coord_str or coord_str == '()':
            return None
        try:
            return tuple(map(float, coord_str.strip("()").split(', ')))
        except:
            return None
    
    def _parse_datetime(self, datetime_str):
        """Parse datetime strings"""
        datetime_str = datetime_str.split(" ")[0]  # Extract the date part
        return datetime.strptime(datetime_str, "%Y-%m-%d")


# Example usage
if __name__ == "__main__":
    # Initialize the climate data fetcher (NO API KEY NEEDED!)
    climate_fetcher = FreeClimateData()
    
    # Example: Get climate data for a specific location and date
    coords = (11.3426, 43.8148)  # Siena, Italy (longitude, latitude)
    date = datetime(2024, 9, 15)
    
    climate_data = climate_fetcher.get_environmental_data(coords, date)
    
    print("Climate data for the last 14 days:")
    for i, day_data in enumerate(climate_data):
        print(f"Day {i+1}: {day_data}")
    
    # Example: Process entire CSV file
    # climate_fetcher.append_climate_data_to_csv(
    #     "data/inaturalist_boletus_edulis_with_el_aspect_corine.csv",
    #     "data/inaturalist_boletus_edulis_with_free_climate.csv"
    # )
