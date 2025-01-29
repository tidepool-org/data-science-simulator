import os
import random
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional


def get_random_patient_files(directory_path: str, num_files: int) -> List[str]:
    """
    Select random patient files from the specified directory.

    Args:
        directory_path: Path to the directory containing patient files
        num_files: Number of files to randomly select

    Returns:
        List of selected file paths
    """
    # Get all files in directory
    all_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]

    # Select random files
    selected_files = random.sample(all_files, min(num_files, len(all_files)))

    # Return full paths
    return [os.path.join(directory_path, f) for f in selected_files]


def read_patient_data(file_path: str) -> dict:
    """
    Read patient data from CSV file and extract settings.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary containing patient settings with distinct pump and patient model values
    """
    try:
        df = pd.read_csv(file_path)

        # Create a settings dictionary
        settings = {}

        # Look for suspend_threshold if it exists
        suspend_row = df[df['setting_name'] == 'suspend_threshold']
        if not suspend_row.empty:
            try:
                # Convert to integer instead of float
                settings['suspend_threshold'] = int(float(suspend_row.iloc[0]['settings']))
            except ValueError as e:
                print(f"Error converting suspend_threshold: {e}")
                settings['suspend_threshold'] = 67  # default value is already an integer
        else:
            settings['suspend_threshold'] = 67  # default value is already an integer

        # Look for actual_carb_ratios (for patient model)
        actual_carb_row = df[df['setting_name'] == 'actual_carb_ratios']
        if not actual_carb_row.empty:
            try:
                settings['actual_carb_ratios'] = float(actual_carb_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting actual_carb_ratios: {e}")
                settings['actual_carb_ratios'] = 15  # default value
        else:
            print("No actual_carb_ratios found in file")
            settings['actual_carb_ratios'] = 15  # default value

        # Look for basal rate for patient model
        actual_basal_row = df[df['setting_name'] == 'actual_basal_rates']
        if not actual_basal_row.empty:
            try:
                settings['actual_basal_rates'] = float(actual_basal_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting actual_basal_rates: {e}")
                settings['actual_basal_rates'] = 1.33 # default value
            else:
                print("No actual_basal_rates found in file")
                settings['actual_basal_rates'] = 1.33 # default value

        # Look for carb_ratio_values (for pump)
        carb_ratio_row = df[df['setting_name'] == 'carb_ratio_values']
        if not carb_ratio_row.empty:
            try:
                settings['carb_ratio_values'] = float(carb_ratio_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting carb_ratio_values: {e}")
                settings['carb_ratio_values'] = 5.5  # default value
        else:
            print("No carb_ratio_values found in file")
            settings['carb_ratio_values'] = 5.5  # default value

        # Look for pump basal rate
        basal_row = df[df['setting_name'] == 'basal_rate_values']
        if not basal_row.empty:
            try:
                settings['basal_rate_values'] = float(basal_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting basal_rate_values: {e}")
                settings['basal_rate_values'] = 1.33 # default value
            else:
                print("No basal_rate_values found in file")
                settings['basal_rate_values'] = 1.33 # default value

        return settings

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


from typing import Dict, List, Optional  # Add Optional to the imports at the top

def create_basic_scenario_json(patient_id: str, duration_hours: float, settings: Dict[str, Optional[float]]) -> Dict:
    """
    Create a basic scenario JSON structure.

    Args:
        patient_id: Identifier for the patient
        duration_hours: Duration of the scenario in hours
        settings: Dictionary containing patient settings

    Returns:
        Dictionary containing the scenario data
    """
    return {
        "metadata": {
            "risk-id": "TLR-",
            "simulation_id": f"test_simulation_{patient_id}",
            "risk_description": "",
            "config_format_version": "v1.0"
        },
        "base_config": "reusable.simulations.base_median_swift",
        "override_config": [
            {
                "sim_id": f"scenario_{patient_id}",
                "duration_hours": duration_hours,
                "patient": {
                    "pump": {
                        "metabolism_settings": {
                            "carb_insulin_ratio": {
                                "start_times": ["00:00:00"],
                                "values": [settings['carb_ratio_values']]
                    },
                            "basal_rate": {
                                "start_times": ["0:00:00"],
                                "values": [settings['basal_rate_values']]
                            }
                        },
                    "patient_model": {
                        "metabolism_settings": {
                            "carb_insulin_ratio": {
                                "start_times": ["00:00:00"],
                                "values": [settings['actual_carb_ratios']]
                            },
                            "basal_rate": {
                                "start_times": ["0:00:00"],
                                "values": [settings['actual_basal_rates']]
                            }
                        }
                    }
                },
                "controller": {
                    "settings": {
                        "suspend_threshold": settings['suspend_threshold']
                    }
                }
            }
            }
        ]
    }


def save_json_file(data: Dict, output_dir: str, filename: str):
    """
    Save data as JSON file.

    Args:
        data: Dictionary to save as JSON
        output_dir: Directory to save the file
        filename: Name of the file
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Create full file path
    file_path = os.path.join(output_dir, filename)

    # Save JSON file with nice formatting
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f"Saved JSON file: {file_path}")


def main():
    # Configuration
    data_directory = "/Users/shawnfoster/data/virtual_patients/json_ready"
    output_directory = "/Users/shawnfoster/data/virtual_patients/test_scenarios"
    num_patients = 2  # Number of patients to select
    duration_hours = 24.0  # Duration in hours

    # Get random patient files
    selected_files = get_random_patient_files(data_directory, num_patients)

    # Process each file and create JSON
    for file_path in selected_files:
        patient_id = os.path.splitext(os.path.basename(file_path))[0]
        print(f"\nProcessing patient: {patient_id}")

        # Read patient data and extract settings
        settings = read_patient_data(file_path)

        if settings is not None:
            # Create basic scenario JSON with settings
            scenario_data = create_basic_scenario_json(patient_id, duration_hours, settings)

            # Save JSON file
            json_filename = f"{patient_id}_scenario.json"
            save_json_file(scenario_data, output_directory, json_filename)


if __name__ == "__main__":
    main()