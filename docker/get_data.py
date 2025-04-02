from datetime import datetime, timedelta
import subprocess
import pathlib
import os 
from icecream import ic

def get_date_strings():
    today = datetime.today()
    date_strings = [(today - timedelta(days=i)).strftime("%Y%j") for i in range(15)]
    return date_strings

def write_structure_to_file(file_path):

    file_path = pathlib.Path(file_path)
    ic(file_path)
    os.makedirs(file_path.parent, exist_ok=True)

    date_strings = get_date_strings()
    
    structure = [
        "+ /NRT/Wind/Daily/"  + date + ".nc" for date in date_strings
    ]
    structure.extend([
        "+ /NRT/P/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.extend([
        "+ /NRT/Pres/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.extend([
        "+ /NRT/RelHum/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.extend([
        "+ /NRT/SpecHum/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.extend([
        "+ /NRT/Tmin/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.extend([
        "+ /NRT/Tmax/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.extend([
        "+ /NRT/Temp/Daily/"  + date + ".nc" for date in date_strings
    ])
    structure.append("- *")
    
    with open(file_path, 'w') as file:
        for line in structure:
            file.write(line + '\n')

def run_rclone_sync(filter_file_path, 
                    destination_path):
    # Define the command to run
    command = [
        "rclone", "sync", "-v", "--filter-from", filter_file_path, "--drive-shared-with-me",
        "google:/MSWX_V100", destination_path
    ]
    
    # Run the command
    try:
        subprocess.run(command, check=True)
        print("rclone sync command executed successfully.")
    except subprocess.CalledProcessError as e:
        print(f"Error: {e}")
