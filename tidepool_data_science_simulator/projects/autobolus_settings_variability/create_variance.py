import pandas as pd
import os
import random
from typing import List
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'patient_settings_modification_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)


def modify_patient_file(file_path: str) -> bool:
    """
    Read a patient file and modify specific settings by ±50%.

    Args:
        file_path: Path to the CSV file to modify
    Returns:
        bool: True if modification was successful, False otherwise
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)

        # Settings to modify
        settings_to_modify = [
            'basal_rate_values',
            'carb_ratio_values',
            'sensitivity_ratio_values'
        ]

        modifications_made = False
        # Modify each setting if it exists
        for setting in settings_to_modify:
            # Check if the setting exists in the file
            mask = df['setting_name'] == setting
            if any(mask):
                # Get current value
                current_value = float(df.loc[mask, 'settings'].iloc[0])

                # Randomly increase or decrease by 50%
                modifier = 1.5 if random.random() > 0.5 else 0.5
                new_value = current_value * modifier

                # Update the value
                df.loc[mask, 'settings'] = new_value

                logging.info(f"Modified {setting} in {os.path.basename(file_path)} from {current_value} to {new_value}")
                modifications_made = True

        if modifications_made:
            # Create backup of original file
            backup_path = file_path + '.backup'
            if not os.path.exists(backup_path):
                df.to_csv(backup_path, index=False)
                logging.info(f"Created backup: {backup_path}")

            # Save the modified file
            df.to_csv(file_path, index=False)
            logging.info(f"Successfully modified {file_path}")
            return True
        else:
            logging.warning(f"No relevant settings found to modify in {file_path}")
            return False

    except Exception as e:
        logging.error(f"Error processing {file_path}: {e}")
        return False


def process_directory(directory_path: str) -> None:
    """
    Process all CSV files in the specified directory.

    Args:
        directory_path: Path to directory containing patient files
    """
    try:
        # Ensure directory exists
        if not os.path.exists(directory_path):
            logging.error(f"Directory not found: {directory_path}")
            return

        # Get all CSV files in directory
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

        if not csv_files:
            logging.warning(f"No CSV files found in {directory_path}")
            return

        logging.info(f"Found {len(csv_files)} CSV files to process")

        # Process each file
        successful_modifications = 0
        for file_name in csv_files:
            file_path = os.path.join(directory_path, file_name)
            if modify_patient_file(file_path):
                successful_modifications += 1

        logging.info(
            f"Processing complete. Successfully modified {successful_modifications} out of {len(csv_files)} files")

    except Exception as e:
        logging.error(f"Error processing directory: {e}")


if __name__ == "__main__":
    directory_path = "/Users/shawnfoster/data/virtual_patients/json_ready"
    logging.info(f"Starting processing of directory: {directory_path}")
    process_directory(directory_path)