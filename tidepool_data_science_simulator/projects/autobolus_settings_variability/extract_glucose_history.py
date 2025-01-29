___author__ = """Shawn Foster"""

__author__ = """Shawn Foster"""

import os
import pandas as pd
import re


def extract_last_three_rows(input_file, output_file):
    try:
        # Read the CSV file
        df = pd.read_csv(input_file)

        # Select the last 3 rows
        last_three_rows = df.tail(3)

        # Write the last 3 rows to a new CSV file
        last_three_rows.to_csv(output_file, index=False)

        print(f"Extracted last 3 rows from {os.path.basename(input_file)} to {os.path.basename(output_file)}")
    except Exception as e:
        print(f"Error processing {input_file}: {e}")


def process_csv_files(input_directory, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Find all CSV files with 'condition' in their name
    condition_files = [f for f in os.listdir(input_directory) if 'condition' in f and f.endswith('.csv')]

    # Sort the files to ensure consistent ordering
    condition_files.sort()

    # Iterate through condition files first and rename them
    for index, filename in enumerate(condition_files, 1):
        input_path = os.path.join(input_directory, filename)

        # Create new filename
        new_filename = f"icgm_user_{index}.csv"
        new_path = os.path.join(input_directory, new_filename)

        # Rename the file
        os.rename(input_path, new_path)
        print(f"Renamed {filename} to {new_filename}")

    # Now process all CSV files in the directory
    for filename in os.listdir(input_directory):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            input_path = os.path.join(input_directory, filename)

            # Create output filename based on input filename
            output_filename = f"{filename}_glucose_v1.csv"
            output_path = os.path.join(output_directory, output_filename)

            # Extract last 3 rows
            extract_last_three_rows(input_path, output_path)


# Set directories
input_directory = '/Users/shawnfoster/data/virtual_patients/raw_files'
output_directory = '/Users/shawnfoster/data/virtual_patients/glucose_history'

# Run the script
process_csv_files(input_directory, output_directory)