__author__ = "Shawn Foster"

import os
import pandas as pd


def process_csv_files(input_dir, output_dir):
    """
    Process CSV files in the input directory:
    1. Remove last 4 rows
    2. Retain only first 3 columns
    3. Move values from Column C to Column B if Column B is empty
    4. Save processed files to output directory

    Args:
    input_dir (str): Path to directory containing input CSV files
    output_dir (str): Path to directory for saving processed CSV files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.csv'):
            # Construct full file paths
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)

            try:
                # Read the CSV file
                df = pd.read_csv(input_path)

                # Remove last 4 rows
                df = df.iloc[:-4]

                # Retain only first 3 columns
                df = df.iloc[:, :3]

                # Move values from Column C to Column B if Column B is empty
                # First, fill NaN with empty string to handle different types of empty values
                df.iloc[:, 1] = df.iloc[:, 1].fillna('')
                df.iloc[:, 2] = df.iloc[:, 2].fillna('')

                # Create a mask for rows where Column B is empty and Column C has a value
                mask = (df.iloc[:, 1] == '') & (df.iloc[:, 2] != '')

                # Move values from Column C to Column B where the mask is True
                df.loc[mask, df.columns[1]] = df.loc[mask, df.columns[2]]
                df.loc[mask, df.columns[2]] = ''

                # Delete Column C
                df = df.iloc[:, :2]

                # Save processed file
                df.to_csv(output_path, index=False)

                print(f"Processed {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Example usage
if __name__ == "__main__":
    input_directory = "/Users/shawnfoster/data/virtual_patients/raw_files"
    output_directory = "/Users/shawnfoster/data/virtual_patients/json_ready"

    process_csv_files(input_directory, output_directory)