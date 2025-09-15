import requests
import json
from datetime import datetime, timedelta
from icecream import ic

base_url = "https://api.ellipsis-drive.com/v3"
token = "epat_6tkXsariLig2fHaPVpQLrzbqUYxR9hgscum7VZAbu6OhkESNUePqeYgh2DTBz9IT"


def list_timestamps(path_id):
    """Lists all timestamps for a given path ID."""
    url_list_folder = f"{base_url}/path/{path_id}/folder/list"
    response = requests.get(url_list_folder)
    data = response.json()
    return data["result"]


# Function to delete a timestamp
def delete_timestamp(path_id_layer, timestamp_id):
    """Deletes a timestamp with the given ID."""
    delete_url = (
        f"{base_url}/path/{path_id_layer}/vector/timestamp/{timestamp_id}/trashed"
    )
    params = {"pathId": path_id_layer, "timestampId": timestamp_id}
    headers_upload = {"Authorization": f"Bearer {token}"}

    response = requests.put(
        delete_url,
        data=params,
        headers=headers_upload,
    )

    delete_url = f"{base_url}/path/{path_id_layer}/vector/timestamp/{timestamp_id}"

    response = requests.delete(delete_url, headers=headers_upload)
    return response


# Function to create a new timestamp
def create_timestamp(path_id):
    """Creates a new timestamp for the given path ID."""
    date_to = datetime.today().strftime("%Y-%m-%d")
    date_from = (datetime.today() - timedelta(1)).strftime("%Y-%m-%d")
    add_timestamp_url = f"{base_url}/path/{path_id}/vector/timestamp"

    payload_timestamp = {
        "pathId": path_id,
        "date": {"from": date_from, "to": date_to},
        "description": "",
    }

    headers_timestamp = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    response_timestamp = requests.post(
        add_timestamp_url, headers=headers_timestamp, data=json.dumps(payload_timestamp)
    )
    return response_timestamp


# Function to upload a file to a timestamp
def upload_file(path_id, timestamp_id, file_name):
    """Uploads a file to a specific timestamp."""
    upload_url = f"{base_url}/path/{path_id}/vector/timestamp/{timestamp_id}/file"

    files = {
        "data": (file_name, open(file_name, "rb"), "application/json"),
    }

    params = {
        "pathId": path_id,
        "timestampId": timestamp_id,
        "name": file_name,
        "format": "geojson",
        "epsg": 4326,
        "method": "full",
    }

    headers_upload = {"Authorization": f"Bearer {token}"}

    response_upload = requests.post(
        upload_url, files=files, data=params, headers=headers_upload
    )
    return response_upload


def refresh_timestamps(file_name, path_id="b11909f0-3c51-4ab2-a57e-ffaa60335770"):
    """Refreshes timestamps by deleting all old ones and uploading a new file."""
    timestamps = list_timestamps(path_id)
    for item in timestamps:
        path_id_layer = item["id"]
        for timestamp in item["vector"]["timestamps"]:
            timestamp_id = timestamp["id"]

            delete_response = delete_timestamp(path_id_layer, timestamp_id)
            if delete_response.status_code == 200:
                ic(f"Successfully deleted timestamp ID: {timestamp_id}")
            else:
                ic(
                    f"Failed to delete timestamp ID: {timestamp_id}. Status code: {delete_response.status_code}"
                )
                ic(delete_response.text)
    # path_id = "49821b18-5a0f-4b5f-871d-6442d1c72d86"
    create_response = create_timestamp("49821b18-5a0f-4b5f-871d-6442d1c72d86")
    if create_response.status_code == 200:
        timestamp_data = create_response.json()
        timestamp_id = timestamp_data["id"]
        ic(f"New timestamp added successfully. Timestamp ID: {timestamp_id}")

        upload_response = upload_file(
            "49821b18-5a0f-4b5f-871d-6442d1c72d86", timestamp_id, file_name
        )
        if upload_response.status_code == 200:
            print("File uploaded successfully.")
        else:
            ic(f"Failed to upload file. Status code: {upload_response.status_code}")
            ic(upload_response.text)

    else:
        ic(f"Failed to add new timestamp. Status code: {create_response.status_code}")
        ic(create_response.text)
