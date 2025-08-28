__author__ = "Shawn Foster"

import os
import pandas as pd
import random  # Add this import

import os
import pandas as pd
import random  # Add this import


def select_random_files(input_dir, num_files):
    """
    Select a random subset of CSV files from the input directory.

    Args:
        input_dir (str): Directory containing CSV files
        num_files (int): Number of files to select

    Returns:
        list: Selected filenames
    """
    # Get all CSV files in directory
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    # Select random files (don't select more than available)
    return random.sample(all_files, min(num_files, len(all_files)))

def process_csv_files(input_dir, output_dir, files_to_process=None):
    """
    Process CSV files in the input directory.
    
    Args:
        input_dir (str): Input directory path
        output_dir (str): Output directory path
        files_to_process (list): Optional list of specific files to process
    """
    os.makedirs(output_dir, exist_ok=True)

    # If no specific files provided, process all CSV files
    if files_to_process is None:
        files_to_process = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for filename in files_to_process:
        if filename.endswith('.csv'):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                df = pd.read_csv(input_path)
                df = df.iloc[:-4]
                df = df.iloc[:, :3]

                df.iloc[:, 1] = df.iloc[:, 1].fillna('')
                df.iloc[:, 2] = df.iloc[:, 2].fillna('')

                mask = (df.iloc[:, 1] == '') & (df.iloc[:, 2] != '')
                df.loc[mask, df.columns[1]] = df.loc[mask, df.columns[2]]
                df.loc[mask, df.columns[2]] = ''

                df = df.iloc[:, :2]
                df.to_csv(output_path, index=False)
                print(f"Processed {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")

if __name__ == "__main__":
    input_directory = "/Users/shawnfoster/data/virtual_patients/raw_files"
    output_directory = "/Users/shawnfoster/data/virtual_patients/json_ready"
    num_files_to_process = 100

    files_to_process = select_random_files(input_directory, num_files_to_process)
    print(f"Selected {len(files_to_process)} files to process:")
    for file in files_to_process:
        print(f"- {file}")

    process_csv_files(input_directory, output_directory, files_to_process)