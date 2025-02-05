import os
import random
import pandas as pd
import json
from datetime import datetime
from typing import List, Dict, Optional, Any

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import POINTER_OBJ_DIR


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


def count_leaf_nodes(data, parent_path=""):
    """
    Recursively count and print information about leaf nodes in a dictionary structure.

    Args:
        data: Dictionary or value to analyze
        parent_path: String tracking the path to this node

    Returns:
        int: Number of leaf nodes found
    """
    count = 0

    if isinstance(data, dict):
        for key, value in data.items():
            current_path = f"{parent_path}.{key}" if parent_path else key
            if isinstance(value, (dict, list)):
                count += count_leaf_nodes(value, current_path)
            else:
                count += 1
                print(f"Leaf node at {current_path}: {value}")
    elif isinstance(data, list):
        for i, value in enumerate(data):
            current_path = f"{parent_path}[{i}]"
            if isinstance(value, (dict, list)):
                count += count_leaf_nodes(value, current_path)
            else:
                count += 1
                print(f"Leaf node at {current_path}: {value}")
    else:
        count = 1
        print(f"Leaf node at {parent_path}: {data}")

    return count

def read_patient_data(file_path: str) -> dict:
    """
    Read patient data from CSV file and extract settings including meals and doses.

    Args:
        file_path: Path to the CSV file

    Returns:
        Dictionary containing patient settings and meal/dose data
    """
    try:
        df = pd.read_csv(file_path)

        # Initialize settings dictionary
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

        # Look for sensitivity factor for patient model
        actual_sensitivity_row = df[df['setting_name'] == 'actual_sensitivity_ratios']
        if not actual_sensitivity_row.empty:
            try:
                settings['actual_sensitivity_ratios'] = float(actual_sensitivity_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting actual_sensitivity_ratios: {e}")
                settings['actual_sensitivity_ratios'] = 25 # default value
        else:
            print("No actual_sensitivity_ratios found in file")
            settings['actual_sensitivity_ratios'] = 25 # default value

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

        # Look for correction range
        target_low_row = df[df['setting_name'] == 'target_range_low']
        if not target_low_row.empty:
            try:
                settings['target_range_low'] = float(target_low_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting target_range_low: {e}")
                settings['target_range_low'] = 80.0  # default value
        else:
            print("No target_range_low found in file")
            settings['target_range_low'] = 80.0  # default value

        target_high_row = df[df['setting_name'] == 'target_range_high']
        if not target_high_row.empty:
            try:
                settings['target_range_high'] = float(target_high_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting target_range_high: {e}")
                settings['target_range_high'] = 180.0  # default value
        else:
            print("No target_range_high found in file")
            settings['target_range_high'] = 180.0  # default value

        # Look for pump sensitivity factor
        sensitivity_row = df[df['setting_name'] == 'sensitivity_ratio_values']
        if not sensitivity_row.empty:
            try:
                settings['sensitivity_ratio_values'] = float(sensitivity_row.iloc[0]['settings'])
            except ValueError as e:
                print(f"Error converting sensitivity_ratio_values: {e}")
                settings['sensitivity_ratio_values'] = 25 # default value
        else:
            print("No sensitivity_ratio found in file")
            settings['sensitivity_ratio_values'] = 25 # default value

        settings['meals'] = []
        settings['doses'] = []

        # Get all carb dates and values
        carb_dates = df[df['setting_name'] == 'carb_dates']['settings'].tolist()
        print(f"Reading carb dates from CSV: {carb_dates[:2]}")  # Show first 2 dates

        carb_values = df[df['setting_name'] == 'actual_carbs']['settings'].tolist()

        # Validate and pair carb dates with values
        if len(carb_dates) == len(carb_values):
            for date, value in zip(carb_dates, carb_values):
                try:
                    carb_value = float(value)
                    # Don't try to parse and reformat the date - it's already correct
                    settings['meals'].append({
                        "start_time": date,  # Use the date string directly
                        "value": carb_value,
                        "type": "carb"  # Add the type field
                    })
                except ValueError as e:
                    print(f"Error converting carb value {value}: {e}")

        # Get all dose times and values
        dose_times = df[df['setting_name'] == 'dose_start_times']['settings'].tolist()
        dose_values = df[df['setting_name'] == 'dose_values']['settings'].tolist()

        # Validate and pair dose times with values
        if len(dose_times) == len(dose_values):
            for time, value in zip(dose_times, dose_values):
                if value == 'accept_recommendation':
                    settings['doses'].append({
                        "time": time,
                        "value": "accept_recommendation"
                    })
        else:
            print(f"Mismatched dose times ({len(dose_times)}) and values ({len(dose_values)})")

        print("\nDose data:")
        print("Number of doses:", len(settings['doses']))
        if settings['doses']:
            print("Sample dose:", settings['doses'][0])

        return settings

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


from typing import Dict, List, Optional  # Add Optional to the imports at the top


def get_glucose_history_pointer(patient_id: str) -> str:
    """
    Create a pointer reference to the reusable glucose history file.

    Args:
        patient_id: The ID of the patient (e.g., 'icgm_user_477')

    Returns:
        String pointer in the format "reusable.glucose.icgm_virtual_pt.<patient_id>"
    """
    return f"reusable.glucose.{patient_id}"


def create_basic_scenario_json(patient_id: str, duration_hours: float, settings: Dict[str, Any]) -> Dict:
    glucose_history_pointer = f"reusable.glucose.{patient_id}"

    # Create carb entries with type
    modified_meals = []
    for meal in settings.get('meals', []):
        meal_with_type = meal.copy()
        meal_with_type['type'] = 'carb'
        modified_meals.append(meal_with_type)

# Create proper bolus entries
    modified_boluses = []
    for bolus in settings.get('doses', []):
        modified_bolus = {
            'time': bolus['time'],
            'value': bolus['value']
        }
        modified_boluses.append(modified_bolus)

    # Create base JSON structure
    scenario = {
        "metadata": {
            "risk-id": f"TLR-AB_{patient_id}",
            "simulation_id": f"test_simulation_{patient_id}",
            "risk_description": "Autobolus/TBR comparison with incorrect settings",
            "config_format_version": "v1.0"
        },
        "base_config": "reusable.simulations.base_median_swift",
        "override_config": [
            {
                "sim_id": f"scenario_{patient_id}",
                "duration_hours": duration_hours,
                "patient": {
                    "sensor": {
                        "glucose_history": glucose_history_pointer
                    },
                    "pump": {
                        "target_range": {
                            "start_times": ["00:00:00"],
                            "lower_values": [settings['target_range_low']],
                            "upper_values": [settings['target_range_high']]
                            },
                        "metabolism_settings": {
                            "carb_insulin_ratio": {
                                "start_times": ["00:00:00"],
                                "values": [settings['carb_ratio_values']]
                            },
                            "basal_rate": {
                                "start_times": ["0:00:00"],
                                "values": [settings['basal_rate_values']]
                            },
                            "insulin_sensitivity_factor": {
                                "start_times": ["0:00:00"],
                                "values": [settings['sensitivity_ratio_values']]
                            }
                        },
                        "carb_entries": modified_meals,
                    },
                    "patient_model": {
                        "glucose_history": glucose_history_pointer,
                        "metabolism_settings": {
                            "carb_insulin_ratio": {
                                "start_times": ["00:00:00"],
                                "values": [settings['actual_carb_ratios']]
                            },
                            "basal_rate": {
                                "start_times": ["0:00:00"],
                                "values": [settings['actual_basal_rates']]
                            },
                            "insulin_sensitivity_factor": {
                                "start_times": ["0:00:00"],
                                "values": [settings['actual_sensitivity_ratios']]
                            }
                        },
                        "carb_entries": modified_meals,
                        "bolus_entries": [
                            {
                            "time": dose["time"],
                            "value": dose["value"]
                            }
                            for dose in settings.get('doses', [])
                        ]
                    }
                },
                "controller": {
                    "settings": {
                        "suspend_threshold": settings['suspend_threshold'],
                        "partial_application_factor": 0.4,
                        "include_positive_velocity_and_RC": False,
                        "max_bolus": 20
                    }
                }
            }
        ]
    }

    print("\nFinal JSON structure:")
    pump_boluses = scenario['override_config'][0]['patient']['pump'].get('bolus_entries', [])
    patient_boluses = scenario['override_config'][0]['patient']['patient_model']['bolus_entries']
    print("Pump bolus entries:", pump_boluses)
    print("Patient model bolus entries:", patient_boluses)

    for i, override in enumerate(scenario["override_config"]):
        print(f"\nAnalyzing override configuration {i}:")
        leaf_count = count_leaf_nodes(override)
        print(f"Total leaf nodes in override {i}: {leaf_count}")

    return scenario


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
    output_directory = "/Users/shawnfoster/PycharmProjects/data-science-simulator-v2/data-science-simulator/scenario_configs/tidepool_risk_v2/loop_risk_v2_0/loop_risk_compare_ab-tbr/TLR-ab_04"
    num_patients = 100  # Number of patients to select
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
            # Create basic scenario JSON with settings and glucose pointer
            scenario_data = create_basic_scenario_json(
                patient_id,
                duration_hours,
                settings
            )

            # Save JSON file
            json_filename = f"Simulation-configuration-TLR-{patient_id}_scenario.json"
            save_json_file(scenario_data, output_directory, json_filename)

            print(f"Processed {len(settings['doses'])} doses:")
            for dose in settings['doses'][:3]:  # Show first 3 doses
                print(f"  {dose}")

if __name__ == "__main__":
    main()