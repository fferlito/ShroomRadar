from ellipsis import upload_file, delete_timestamp, create_timestamp, refresh_timestamps
from get_predictions import filter_predictions, generarate_predictions
from get_data import run_rclone_sync, write_structure_to_file
from icecream import ic
import fire

def run_pipeline():
    
    # define vars
    file_path = "data/environmental_data/file_structure_with_14_days.txt"
    dest = "data/environmental_data"
    path_input_geojson = 'data/base_maps/north_spain.geojson'
    path_output_geojson = 'data/outputs/map.geojson'
    path_filtered_geojson = 'data/outputs/north_spain_clean.geojson'

    
    # get weather data
    
    write_structure_to_file(file_path)
    ic(f"File '{file_path}' has been created with the specified structure.")
    run_rclone_sync(file_path, dest)
    ic("Weather data downloaded!")
    
    # generate geojson

    generarate_predictions(path_input_geojson, path_output_geojson)
    filter_predictions(path_output_geojson, path_filtered_geojson, threshold=0.05)
    

    # upload geojson
    ## TODO: add multiple geojsons
    refresh_timestamps(path_filtered_geojson)
    


if __name__ == '__main__':
  fire.Fire(run_pipeline)
