import os
import glob
import logging
from google.cloud import storage
from google.api_core import exceptions
from tqdm import tqdm
from icecream import ic
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class GCSOperationError(Exception):
    """Custom exception for GCS operation failures."""


class GCSManager:
    """A manager for handling operations with Google Cloud Storage."""

    def __init__(self, bucket_name: str):
        """
        Initializes the GCSManager.

        Args:
            bucket_name: The name of the GCS bucket.

        Raises:
            GCSOperationError: If the client cannot be initialized or the bucket is not found.
        """
        try:
            self.client = storage.Client()
            self.bucket = self.client.get_bucket(bucket_name)
        except exceptions.NotFound as e:
            raise GCSOperationError(f"Bucket '{bucket_name}' not found.") from e
        except exceptions.GoogleAPICallError as e:
            raise GCSOperationError(f"Failed to initialize GCS client: {e}") from e

    def upload_files(
        self, local_dir: str, gcs_prefix: str, file_pattern: str = "**/*.tif"
    ):
        """
        Uploads files from a local directory to a GCS bucket.

        Args:
            local_dir: The local directory containing files to upload.
            gcs_prefix: The prefix to use for the files in the GCS bucket.
            file_pattern: The glob pattern to match files (e.g., '**/*.tif').

        Returns:
            A tuple containing lists of successfully uploaded and failed file paths.
        """
        pattern = os.path.join(local_dir, file_pattern)
        local_files = glob.glob(pattern, recursive=True)

        if not local_files:
            logging.warning(
                "No files found in '%s' matching '%s'", local_dir, file_pattern
            )
            return [], []

        logging.info("Found %d files to upload.", len(local_files))
        uploaded_files, failed_uploads = [], []

        for local_file in tqdm(local_files, desc="Uploading to GCS"):
            try:
                relative_path = os.path.relpath(local_file, local_dir)
                blob_name = os.path.join(gcs_prefix, relative_path).replace("\\", "/")
                blob = self.bucket.blob(blob_name)
                blob.upload_from_filename(local_file)
                uploaded_files.append(blob.public_url)
            except (exceptions.GoogleAPICallError, OSError) as e:
                failed_uploads.append((local_file, str(e)))
                logging.error("Failed to upload %s: %s", local_file, e)

        self._log_summary("Upload", len(uploaded_files), failed_uploads)
        return uploaded_files, failed_uploads

    def download_files(
        self, gcs_prefix: str, local_dir: str, file_suffix: str = ".tif"
    ):
        """
        Downloads files from a GCS bucket to a local directory.

        Args:
            gcs_prefix: The prefix of the files to download from the GCS bucket.
            local_dir: The local directory to download files to.
            file_suffix: The suffix of files to download (e.g., '.tif').

        Returns:
            A tuple containing lists of successfully downloaded and failed blob names.
        """
        blobs_to_download = self.list_files(gcs_prefix, file_suffix)

        if not blobs_to_download:
            logging.warning(
                "No files found in bucket '%s' with prefix '%s' and suffix '%s'",
                self.bucket.name,
                gcs_prefix,
                file_suffix,
            )
            return [], []

        logging.info("Found %d files to download.", len(blobs_to_download))
        downloaded_files, failed_downloads = [], []
        os.makedirs(local_dir, exist_ok=True)

        for blob_name in tqdm(blobs_to_download, desc="Downloading from GCS"):
            try:
                blob = self.bucket.blob(blob_name)
                relative_path = os.path.relpath(blob_name, gcs_prefix)
                local_path = os.path.join(local_dir, relative_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                blob.download_to_filename(local_path)
                downloaded_files.append(local_path)
            except (exceptions.GoogleAPICallError, OSError) as e:
                failed_downloads.append((blob_name, str(e)))
                logging.error("Failed to download %s: %s", blob_name, e)

        self._log_summary("Download", len(downloaded_files), failed_downloads)
        return downloaded_files, failed_downloads

    def list_files(self, gcs_prefix: str, file_suffix: str = ".tif") -> list[str]:
        """
        Lists files in the GCS bucket with a given prefix and suffix.

        Args:
            gcs_prefix: The prefix to filter by.
            file_suffix: The suffix of files to list (e.g., '.tif').

        Returns:
            A list of blob names.
        """
        blobs = self.client.list_blobs(self.bucket, prefix=gcs_prefix)
        filtered_blobs = [
            blob.name for blob in blobs if blob.name.endswith(file_suffix)
        ]
        logging.info(
            "Found %d files in bucket '%s' with prefix '%s'",
            len(filtered_blobs),
            self.bucket.name,
            gcs_prefix,
        )
        return filtered_blobs

    def _log_summary(self, operation: str, success_count: int, failures: list):
        """Logs a summary of an operation."""
        logging.info("📊 %s Summary:", operation)
        logging.info("✅ Successfully completed: %d files", success_count)
        logging.info("❌ Failed: %d files", len(failures))
        if failures:
            for item, error in failures:
                logging.error("  - %s: %s", item, error)


def refresh_timestamps(path_output_geojson: str, model_name: str, aoi_name: str):
    tiles_dir = "data/tiles"
    bucket_name = "mushroom-radar-tiles"

    if os.path.exists(tiles_dir):
        shutil.rmtree(tiles_dir)
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
            rel_path = os.path.relpath(local_path, tiles_dir).replace("\\", "/")
            remote_path = f"{model_name}/{aoi_name}/today/{rel_path}"
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
