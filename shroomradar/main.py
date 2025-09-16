from icecream import ic
import fire

from .src.ellipsis import refresh_timestamps
from .src.climate import write_today_structure_to_file, run_rclone_sync, generate_input_model
from .src.model import filter_predictions, generarate_predictions


def run_pipeline():
    """
    path_input_geojson = 'data/base_maps/siena_05_with_elevation_aspect_LC.geojson'
    path_geojson_with_climate = 'data/base_maps/siena_with_climate.geojson'
    path_output_geojson = 'data/outputs/map_siena.geojson'
    path_filtered_geojson = 'data/outputs/siena_readyclean.geojson'
    """

    path_input_geojson = (
        "data/base_maps/basque_country_05_with_elevation_aspect.geojson"
    )
    path_geojson_with_climate = "data/base_maps/BC_with_climate.geojson"
    path_output_geojson = "data/outputs/map_BC.geojson"
    path_filtered_geojson = "data/outputs/BC_readyclean.geojson"
    file_path = "data/environmental_data/file_structure_with_14_days.txt"
    dest = "data/environmental_data"


    ic("🌦️ Adding climate data...")
    write_today_structure_to_file(file_path)
    run_rclone_sync(file_path, dest)
    generate_input_model(
        path_input_geojson,
        path_geojson_with_climate,
        num_days=14,
    )
    ic("Climate data added!")

    ic("Generate predictions...")
    generarate_predictions(path_geojson_with_climate, path_output_geojson)
    filter_predictions(path_output_geojson, path_filtered_geojson, threshold=0.0008)

    ## TODO: add multiple geojsons
    ic("Upload to ellipsis...")
    refresh_timestamps(path_output_geojson)


if __name__ == "__main__":
    fire.Fire(run_pipeline)
