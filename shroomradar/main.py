from ellipsis import upload_file, delete_timestamp, create_timestamp, refresh_timestamps
from get_predictions import filter_predictions, generarate_predictions
from get_data import run_rclone_sync, write_structure_to_file
from append_data import generate_input_model
from icecream import ic
import fire


def run_pipeline():
      """
    path_input_geojson = 'data/base_maps/siena_05_with_elevation_aspect_LC.geojson'
    path_geojson_with_climate = 'data/base_maps/siena_with_climate.geojson'
    path_output_geojson = 'data/outputs/map_siena.geojson'
    path_filtered_geojson = 'data/outputs/siena_readyclean.geojson'
    """
    path_input_geojson = 'data/base_maps/basque_country_05_with_elevation_aspect.geojson'
    path_geojson_with_climate = 'data/base_maps/BC_with_climate.geojson'
    path_output_geojson = 'data/outputs/map_BC.geojson'
    path_filtered_geojson = 'data/outputs/BC_readyclean.geojson'

    file_path = "data/environmental_data/file_structure_with_14_days.txt"
    dest = "data/environmental_data"

    # Step 1: Add climate data to input GeoJSON using Open-Meteo API
    ic("🌦️ Adding climate data using Open-Meteo API...")
    run_rclone_sync(file_path, dest)
    generate_input_model(path_input_geojson, path_geojson_with_climate, num_days=14, delay_between_requests=0.0)
    ic("Climate data added!")
    
    # Step 2: Generate predictions using the GeoJSON with climate data
    generarate_predictions(path_geojson_with_climate, path_output_geojson)
    filter_predictions(path_output_geojson, path_filtered_geojson, threshold=0.0008)
    
    # upload geojson
    ## TODO: add multiple geojsons
    refresh_timestamps(path_output_geojson)

    


if __name__ == '__main__':
  fire.Fire(run_pipeline)


