__author__ = "Shawn Foster"

import os
import pandas as pd
from datetime import datetime, timedelta


def process_csv_files(input_directory, output_directory):
    # Create output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)

    # Starting datetime
    start_datetime = datetime(2019, 8, 15, 0, 0, 0)

    # Iterate through all files in the input directory
    for filename in os.listdir(input_directory):
        # Check if the file is a CSV
        if filename.endswith('.csv'):
            input_path = os.path.join(input_directory, filename)

            try:
                # Read the CSV file
                df = pd.read_csv(input_path)

                # Select first 146 columns
                df_truncated = df.iloc[:, :146]

                # Pivot the data
                id_columns = ['setting_name', 'settings']

                # Melt the dataframe to prepare for pivoting
                df_melted = df_truncated.melt(
                    id_vars=id_columns,
                    var_name='timestamp_index',
                    value_name='value'
                )

                # Remove rows with empty values
                df_melted = df_melted.dropna(subset=['value'])

                # Convert timestamp index to datetime
                df_melted['timestamp'] = df_melted['timestamp_index'].apply(
                    lambda x: (start_datetime + timedelta(minutes=5 * (int(x)))).strftime('%m/%d/%Y %H:%M:%S')
                )

                # Drop the original timestamp index
                df_melted = df_melted.drop(columns=['timestamp_index'])

                # Create output filename
                output_filename = f"{filename}_trimmed.csv"
                output_path = os.path.join(output_directory, output_filename)

                # Save the processed file
                df_melted.to_csv(output_path, index=False)

                print(f"Processed {filename} -> {output_filename}")

                # Print first few timestamps for verification
                print("First few timestamps:")
                print(df_melted['timestamp'].head())

            except Exception as e:
                print(f"Error processing {filename}: {e}")


# Set directories
input_directory = '/Users/shawnfoster/data/virtual_patients/glucose_history'
output_directory = '/Users/shawnfoster/data/virtual_patients/glucose_history/processed_files'

# Run the script
process_csv_files(input_directory, output_directory)