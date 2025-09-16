import subprocess
import platform
import shutil
import os

def push_docker_to_gcr(project_id, image_name):
    """
    Builds and pushes a Docker image to Google Container Registry (GCR).

    Args:
        project_id (str): GCP Project ID.
        image_name (str): Name for the Docker image (e.g., "my-rclone-python").
    """
    
    # Determine the correct gcloud command for the platform
    gcloud_cmd = "gcloud"
    if platform.system() == "Windows":
        # Try to find gcloud.cmd first, fallback to gcloud
        gcloud_path = shutil.which("gcloud.cmd")
        if gcloud_path:
            gcloud_cmd = "gcloud.cmd"
        else:
            gcloud_path = shutil.which("gcloud")
            if not gcloud_path:
                print("❌ Error: gcloud not found in PATH. Please ensure Google Cloud SDK is installed and in your PATH.")
                return
    
    try:
        # Step 1: Authenticate with GCP
        print("🔑 Authenticating with GCP...")
        subprocess.run([gcloud_cmd, "auth", "configure-docker"], check=True, shell=True)

        # Step 2: Set the GCP project
        print(f"📌 Setting GCP project: {project_id}")
        subprocess.run([gcloud_cmd, "config", "set", "project", project_id], check=True, shell=True)

        # Step 3: Build the Docker image
        full_image_path = f"gcr.io/{project_id}/{image_name}:latest"
        print(f"🐳 Building Docker image: {full_image_path}")
        
        # Get the directory where this script is located (should be the docker directory)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        print(f"📁 Building from directory: {script_dir}")
        
        # Run docker build from the script's directory (where Dockerfile is located)
        subprocess.run(["docker", "build", "-t", full_image_path, "."], 
                      check=True, shell=True, cwd=script_dir)

        # Step 4: Push the image to Google Container Registry (GCR)
        print(f"📤 Pushing Docker image to GCR: {full_image_path}")
        subprocess.run(["docker", "push", full_image_path], check=True, shell=True)

        print("✅ Image successfully pushed to GCR!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
    except FileNotFoundError as e:
        print(f"❌ Command not found: {e}")
        print("Please ensure Google Cloud SDK and Docker are installed and in your PATH.")


push_docker_to_gcr(
    project_id="mushroom-detector-450321",
    image_name="rclone-container-test-v1"
)