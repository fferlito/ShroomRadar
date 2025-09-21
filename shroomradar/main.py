from icecream import ic
import fire
import os
import platform
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from src.climate import (
    write_today_structure_to_file,
    run_rclone_sync,
    generate_input_model,
)  # pylint: disable=import-error
from src.model import generarate_predictions  # pylint: disable=import-error
from src.gcp import refresh_timestamps

MODELS = {
    "boletus_aestivalis_eng": os.path.join(
        "data", "models", "Summer Bolete_XGBClassifier_eng.pkl"
    ),
    "boletus_aestivalis_eng_1000": os.path.join(
        "data", "models", "Summer Bolete_XGBClassifier_eng_1000.pkl"
    ),
    "boletus_aestivalis": os.path.join(
        "data", "models", "Summer Bolete_XGBClassifier.pkl"
    ),
}


AOIS = {
    "tuscany": {"input_path": "data/base_maps/grid_tuscany_with_topography.geojson"},
    "basque_country": {
        "input_path": "data/base_maps/grid_basque_country_with_topography.geojson"
    },
}


def run_pipeline():
    """
    Main pipeline to generate mushroom predictions and upload tiles.
    """
    file_path = "data/environmental_data/file_structure_with_14_days.txt"
    dest = "data/environmental_data"

    if platform.system() == "Windows":
        rclone_path = os.path.join("data", "rclone.exe")
    else:
        rclone_path = "rclone"

    ic("🌦️ Downloading climate data...")
    write_today_structure_to_file(file_path)
    run_rclone_sync(file_path, dest, rclone_path)

    output_dir = "data/outputs"
    os.makedirs(output_dir, exist_ok=True)

    for model_name, model_path in MODELS.items():
        for aoi_name, aoi_info in AOIS.items():
            ic(f"Processing {model_name} for {aoi_name}")

            path_input_geojson = aoi_info["input_path"]
            base_name = os.path.splitext(os.path.basename(path_input_geojson))[0]

            path_geojson_with_climate = os.path.join(
                output_dir, f"{base_name}_climate.geojson"
            )
            path_output_geojson = os.path.join(
                output_dir, f"{base_name}_{model_name}_predictions.geojson"
            )

            ic("🌦️ Adding climate data to AOI...")
            generate_input_model(
                path_input_geojson,
                path_geojson_with_climate,
                data_dir=dest,
                num_days=14,
            )

            ic("🍄 Generating predictions...")
            generarate_predictions(
                path_geojson_with_climate,
                path_output_geojson,
                model_path=model_path,
                use_engineered=True,
            )

            ic("☁️ Uploading to GCP...")
            refresh_timestamps(path_output_geojson, model_name, aoi_name)
            ic(f"✅ Finished processing {model_name} for {aoi_name}")


if __name__ == "__main__":
    fire.Fire(run_pipeline)
