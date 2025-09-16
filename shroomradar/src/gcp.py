import os
from google.cloud import storage
from tqdm import tqdm
import glob

def upload_dem_tiles_to_gcs(local_dir="dem_tiles", bucket_name="your-bucket-name", gcs_prefix="dem_tiles/"):
    """
    Upload all DEM tiles from local directory to Google Cloud Storage bucket.
    
    Args:
        local_dir: Local directory containing DEM tiles
        bucket_name: Name of the GCS bucket
        gcs_prefix: Prefix for files in the bucket (e.g., "dem_tiles/")
    
    Returns:
        List of uploaded file names
    """
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # Find all .tif files in the local directory
    pattern = os.path.join(local_dir, "**", "*.tif")
    local_files = glob.glob(pattern, recursive=True)
    
    if not local_files:
        print(f"No .tif files found in {local_dir}")
        return []
    
    print(f"Found {len(local_files)} .tif files to upload")
    
    uploaded_files = []
    failed_uploads = []
    
    for local_file in tqdm(local_files, desc="Uploading to GCS"):
        try:
            # Create the destination blob name
            relative_path = os.path.relpath(local_file, local_dir)
            blob_name = gcs_prefix + relative_path.replace("\\", "/")  # Ensure forward slashes
            
            # Create blob and upload
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(local_file)
            
            uploaded_files.append(blob_name)
            print(f"✅ Uploaded: {blob_name}")
            
        except Exception as e:
            failed_uploads.append((local_file, str(e)))
            print(f"❌ Failed to upload {local_file}: {str(e)}")
    
    # Summary
    print(f"\n📊 Upload Summary:")
    print(f"✅ Successfully uploaded: {len(uploaded_files)} files")
    print(f"❌ Failed uploads: {len(failed_uploads)} files")
    
    if failed_uploads:
        print("\nFailed uploads:")
        for file_path, error in failed_uploads:
            print(f"  - {file_path}: {error}")
    
    return uploaded_files

def download_dem_tiles_from_gcs(bucket_name="your-bucket-name", gcs_prefix="dem_tiles/", local_dir="dem_tiles"):
    """
    Download DEM tiles from Google Cloud Storage bucket to local directory.
    
    Args:
        bucket_name: Name of the GCS bucket
        gcs_prefix: Prefix for files in the bucket (e.g., "dem_tiles/")
        local_dir: Local directory to download files to
    
    Returns:
        List of downloaded file names
    """
    # Initialize the GCS client
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    # List all blobs with the given prefix
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    
    # Filter for .tif files
    tif_blobs = [blob for blob in blobs if blob.name.endswith('.tif')]
    
    if not tif_blobs:
        print(f"No .tif files found in bucket {bucket_name} with prefix {gcs_prefix}")
        return []
    
    print(f"Found {len(tif_blobs)} .tif files to download")
    
    downloaded_files = []
    failed_downloads = []
    
    for blob in tqdm(tif_blobs, desc="Downloading from GCS"):
        try:
            # Create local file path
            relative_path = blob.name[len(gcs_prefix):]  # Remove prefix
            local_file = os.path.join(local_dir, relative_path)
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(local_file), exist_ok=True)
            
            # Download the file
            blob.download_to_filename(local_file)
            
            downloaded_files.append(local_file)
            print(f"✅ Downloaded: {blob.name}")
            
        except Exception as e:
            failed_downloads.append((blob.name, str(e)))
            print(f"❌ Failed to download {blob.name}: {str(e)}")
    
    # Summary
    print(f"\n📊 Download Summary:")
    print(f"✅ Successfully downloaded: {len(downloaded_files)} files")
    print(f"❌ Failed downloads: {len(failed_downloads)} files")
    
    if failed_downloads:
        print("\nFailed downloads:")
        for blob_name, error in failed_downloads:
            print(f"  - {blob_name}: {error}")
    
    return downloaded_files

def list_gcs_dem_tiles(bucket_name="your-bucket-name", gcs_prefix="dem_tiles/"):
    """
    List all DEM tiles in the GCS bucket.
    
    Args:
        bucket_name: Name of the GCS bucket
        gcs_prefix: Prefix for files in the bucket (e.g., "dem_tiles/")
    
    Returns:
        List of blob names
    """
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    
    blobs = bucket.list_blobs(prefix=gcs_prefix)
    tif_blobs = [blob.name for blob in blobs if blob.name.endswith('.tif')]
    
    print(f"Found {len(tif_blobs)} .tif files in bucket {bucket_name}")
    for blob_name in tif_blobs:
        print(f"  - {blob_name}")
    
    return tif_blobs
