import os
import subprocess
import csv
from datetime import datetime, timedelta
from tqdm import tqdm
from icecream import ic
import pathlib


def get_unique_dates_from_csv(
    csv_file_path: str, limit_rows: int | None = None
) -> list[datetime]:
    """
    Extracts unique observation dates from a CSV file.

    The function reads a CSV file, expecting a column named 'observed_on'. It parses the date
    from this column, handling various common date formats. It returns a sorted list of unique
    datetime objects. An optional row limit can be applied for faster processing on large files.

    Args:
        csv_file_path (str): The path to the input CSV file.
        limit_rows (Optional[int]): The maximum number of rows to process from the CSV.
                                    If None, all rows are processed. Defaults to None.

    Returns:
        List[datetime]: A sorted list of unique datetime objects found in the CSV.
    """
    unique_dates: set[datetime] = set()

    with open(csv_file_path, "r", encoding="utf-8") as csvfile:
        # First, get total rows for tqdm without consuming the reader
        total_rows = sum(1 for _ in open(csv_file_path, "r", encoding="utf-8")) - 1
        csvfile.seek(0)  # Reset file pointer
        reader = csv.DictReader(csvfile)

        progress_bar = tqdm(reader, total=limit_rows or total_rows, desc="Reading CSV")
        for i, row in enumerate(progress_bar):
            if limit_rows is not None and i >= limit_rows:
                break

            if "observed_on" in row and row["observed_on"]:
                date_str = row["observed_on"].split(" ")[0]
                date_obj = None

                # Common date formats to try
                date_formats = [
                    "%Y-%m-%d",
                    "%m/%d/%Y",
                    "%d/%m/%Y",
                    "%Y/%m/%d",
                    "%m-%d-%Y",
                    "%d-%m-%Y",
                ]

                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue

                if date_obj:
                    unique_dates.add(date_obj)
                else:
                    tqdm.write(f"⚠️ Could not parse date: {row['observed_on']}")

    return sorted(list(unique_dates))


def generate_file_structure_from_csv(
    csv_file_path: str, output_file_path: str, limit_rows: int | None = None
) -> list[str]:
    """
    Generates an rclone filter file from a CSV of observations.

    This function reads unique observation dates from a given CSV file. For each date,
    it identifies the preceding 15 days (including the observation day itself) and
    constructs file paths for various climate variables (e.g., Wind, Temperature)
    for both 'Past' and 'NRT' (Near-Real-Time) datasets.

    The output is a text file formatted for use with rclone's `--filter-from`
    option, which will include only the required daily data files and exclude
    all others.

    Args:
        csv_file_path (str): The path to the input CSV file containing observation dates.
        output_file_path (str): The path where the generated filter file will be saved.
        limit_rows (Optional[int]): The maximum number of rows to process from the CSV.
                                    If None, all rows are processed. Defaults to None.

    Returns:
        List[str]: A list of the lines written to the filter file.
    """
    unique_dates = get_unique_dates_from_csv(csv_file_path, limit_rows=limit_rows)
    print(f"📅 Found {len(unique_dates)} unique observation dates")

    all_files: set[str] = set()
    variables = ["Wind", "P", "Pres", "RelHum", "SpecHum", "Tmin", "Tmax", "Temp"]

    for obs_date in tqdm(unique_dates, desc="Building file list"):
        for i in range(15):
            target_date = obs_date - timedelta(days=i)
            date_string = target_date.strftime("%Y%j")

            for variable in variables:
                all_files.add(f"+ /Past/{variable}/Daily/{date_string}.nc")
                all_files.add(f"+ /NRT/{variable}/Daily/{date_string}.nc")

    file_list = sorted(list(all_files))
    file_list.append("- *")  # exclude everything else

    with open(output_file_path, "w", encoding="utf-8") as file:
        for line in file_list:
            file.write(line + "\n")

    print(f"📝 Generated {len(file_list)-1} unique file entries")
    print(f"✅ File structure saved to: {output_file_path}")
    return file_list


def get_date_strings():
    today = datetime.today()
    # Start from yesterday (skip today) and get 14 days
    date_strings = [(today - timedelta(days=i + 1)).strftime("%Y%j") for i in range(15)]
    return date_strings


def write_today_structure_to_file(file_path):

    file_path = pathlib.Path(file_path)
    ic(file_path)
    os.makedirs(file_path.parent, exist_ok=True)

    date_strings = get_date_strings()

    structure = ["+ /NRT/Wind/Daily/" + date + ".nc" for date in date_strings]
    structure.extend(["+ /NRT/P/Daily/" + date + ".nc" for date in date_strings])
    structure.extend(["+ /NRT/Pres/Daily/" + date + ".nc" for date in date_strings])
    structure.extend(["+ /NRT/RelHum/Daily/" + date + ".nc" for date in date_strings])
    structure.extend(["+ /NRT/SpecHum/Daily/" + date + ".nc" for date in date_strings])
    structure.extend(["+ /NRT/Tmin/Daily/" + date + ".nc" for date in date_strings])
    structure.extend(["+ /NRT/Tmax/Daily/" + date + ".nc" for date in date_strings])
    structure.extend(["+ /NRT/Temp/Daily/" + date + ".nc" for date in date_strings])
    structure.append("- *")

    with open(file_path, "w") as file:
        for line in structure:
            file.write(line + "\n")


def run_rclone_sync(
    filter_file_path: str, dest_folder: str = "climate_data"
) -> None:
    """
    Executes the rclone sync command to download specified climate data files.

    This function constructs and runs an `rclone sync` command to transfer files
    from a shared Google Drive folder ('google:/MSWX_V100') to a local destination.
    It uses a filter file to selectively download only the required data, making the
    process efficient. The function assumes `rclone.exe` is located at `../docker/rclone.exe`.

    Args:
        filter_file_path (str): The path to the text file containing rclone filter rules.
        dest_folder (str): The local directory where the climate data will be saved.
                           Defaults to "climate_data".

    Returns:
        None
    """
    rclone_path = os.path.join("..", "docker", "rclone.exe")

    if not os.path.exists(rclone_path):
        print(f"❌ Error: rclone not found at {rclone_path}")
        return

    command = [
        rclone_path,
        "sync",
        "-v",
        "--filter-from",
        filter_file_path,
        "--drive-shared-with-me",
        "google:/MSWX_V100",  # root folder
        dest_folder,
    ]

    print(f"▶️ Running command: {' '.join(command)}")
    try:
        subprocess.run(command, check=True)
        print("✅ rclone sync command executed successfully.")
        print(f"📂 Climate data downloaded to: {dest_folder}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running rclone: {e}")
    except FileNotFoundError:
        print("❌ Error: rclone executable not found.")


def download_climate_data_from_csv_fixed(
    csv_file_path: str,
    dest_folder: str = "..//climate_data",
    limit_rows: int | None = None,
) -> str:
    """
    Orchestrates the download of climate data based on a CSV file of observations.

    This function performs a two-step process:
    1. It calls `generate_file_structure_from_csv` to create an rclone filter file
       containing the paths of all required climate data files.
    2. It then calls `run_rclone_sync_fixed` to execute an rclone command that
       downloads only the files specified in the filter file.

    Args:
        csv_file_path (str): Path to the input CSV file with an 'observed_on' column.
        dest_folder (str): The local directory where the climate data will be saved.
                           Defaults to "..//climate_data".
        limit_rows (Optional[int]): The maximum number of rows to process from the CSV.
                                    If None, all rows are processed. Defaults to None.

    Returns:
        str: The path to the generated filter file.
    """
    filter_file_path = "climate_files_from_csv.txt"

    print("Step 1: Generating file structure from CSV...")
    generate_file_structure_from_csv(
        csv_file_path, filter_file_path, limit_rows=limit_rows
    )

    print("Step 2: Downloading climate data files...")
    run_rclone_sync(filter_file_path, dest_folder)

    return filter_file_path
