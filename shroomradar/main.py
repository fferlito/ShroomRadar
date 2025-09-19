from icecream import ic
import fire
import os
import platform
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from google.cloud import storage
from src.climate import (
    write_today_structure_to_file,
    run_rclone_sync,
    generate_input_model,
)  # pylint: disable=import-error
from src.model import (
    filter_predictions,
    generarate_predictions,
)  # pylint: disable=import-error


def refresh_timestamps(path_output_geojson: str):
    tiles_dir = "/data/tiles"
    bucket_name = "mushroom-radar-tiles"

    os.makedirs(tiles_dir, exist_ok=True)

    ic("🧩 Running Tippecanoe to generate vector tiles...")
    subprocess.run(
        [
            "tippecanoe",
            "-f",
            "-e",
            tiles_dir,
            "-Z1",
            "-z14",
            "--no-tile-compression",
            "--no-feature-limit",
            "--no-tile-size-limit",
            "--preserve-input-order",
            path_output_geojson,
        ],
        check=True,
    )

    ic("✅ Tippecanoe finished successfully!")
    ic("📂 Tiles generated:", os.listdir(tiles_dir)[:10])  # Debug: print first 10

    client = storage.Client()
    bucket = client.bucket(bucket_name)

    file_list = []
    for root, _, files in os.walk(tiles_dir):
        for file in files:
            local_path = os.path.join(root, file)
            remote_path = "tiles/" + os.path.relpath(local_path, tiles_dir).replace(
                "\\", "/"
            )
            file_list.append((local_path, remote_path))

    ic(f"☁️ Found {len(file_list)} tiles to upload")

    def upload_file(local_remote):
        local_path, remote_path = local_remote
        blob = bucket.blob(remote_path)
        blob.cache_control = "no-store"
        if local_path.endswith(".pbf"):
            blob.content_type = "application/octet-stream"
        elif local_path.endswith(".json"):
            blob.content_type = "application/json"

        blob.upload_from_filename(local_path)
        blob.patch()  # actually apply metadata
        return remote_path

    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(upload_file, f): f for f in file_list}
        for i, future in enumerate(as_completed(futures), 1):
            try:
                remote_path = future.result()
                if i % 1000 == 0 or i == len(file_list):
                    ic(f"[{i}/{len(file_list)}] Uploaded {remote_path}")
            except Exception as e:
                ic(f"❌ Error uploading {futures[future][1]}: {e}")

    ic("🎉 All tiles uploaded successfully to GCS!")


def run_pipeline():
    """
    path_input_geojson = 'data/base_maps/siena_05_with_elevation_aspect_LC.geojson'
    path_geojson_with_climate = 'data/base_maps/siena_with_climate.geojson'
    path_output_geojson = 'data/outputs/map_siena.geojson'
    path_filtered_geojson = 'data/outputs/siena_readyclean.geojson'
    """

    """
    path_input_geojson = (
        "data/base_maps/basque_country_05_with_elevation_aspect.geojson"
    )
    path_geojson_with_climate = "data/base_maps/BC_with_climate.geojson"
    path_output_geojson = "data/outputs/map_BC.geojson"
    path_filtered_geojson = "data/outputs/BC_readyclean.geojson"
    file_path = "data/environmental_data/file_structure_with_14_days.txt"
    dest = "data/environmental_data"
    """

    path_input_geojson = "data/base_maps/grid_tuscany_with_topography.geojson"
    path_geojson_with_climate = (
        "data/base_maps/grid_tuscany_with_topography_climate.geojson"
    )
    path_output_geojson = (
        "data/outputs/grid_tuscany_with_topography_predictions4.geojson"
    )
    file_path = "data/environmental_data/file_structure_with_14_days.txt"
    dest = "data//environmental_data"
    if platform.system() == "Windows":
        rclone_path = os.path.join("data", "rclone.exe")
    else:
        rclone_path = "rclone"
    model_path = os.path.join("data", "models", "XGBClassifier.pkl")

    ic("🌦️ Adding climate data...")
    write_today_structure_to_file(file_path)
    run_rclone_sync(file_path, dest, rclone_path)
    generate_input_model(
        path_input_geojson,
        path_geojson_with_climate,
        data_dir=dest,
        num_days=14,
    )
    ic("Climate data added!")
    ic("Generate predictions...")
    generarate_predictions(
        path_geojson_with_climate,
        path_output_geojson,
        model_path=model_path,
        use_engineered=True,
    )

    ## TODO: add multiple geojsons
    ic("Upload to GCP...")
    refresh_timestamps(path_output_geojson)


if __name__ == "__main__":
    fire.Fire(run_pipeline)
