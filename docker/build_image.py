import subprocess

def push_docker_to_gcr(project_id, image_name):
    """
    Builds and pushes a Docker image to Google Container Registry (GCR).

    Args:
        project_id (str): GCP Project ID.
        image_name (str): Name for the Docker image (e.g., "my-rclone-python").
    """
    
    try:
        # Step 1: Authenticate with GCP
        print("🔑 Authenticating with GCP...")
        subprocess.run(["gcloud", "auth", "configure-docker"], check=True)

        # Step 2: Set the GCP project
        print(f"📌 Setting GCP project: {project_id}")
        subprocess.run(["gcloud", "config", "set", "project", project_id], check=True)

        # Step 3: Build the Docker image
        full_image_path = f"gcr.io/{project_id}/{image_name}:latest"
        print(f"🐳 Building Docker image: {full_image_path}")
        subprocess.run(["docker", "build", "-t", full_image_path, "."], check=True)

        # Step 4: Push the image to Google Container Registry (GCR)
        print(f"📤 Pushing Docker image to GCR: {full_image_path}")
        subprocess.run(["docker", "push", full_image_path], check=True)

        print("✅ Image successfully pushed to GCR!")

    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")


push_docker_to_gcr(
    project_id="mushroom-detector-450321",
    image_name="rclone-container-test-v1"
)