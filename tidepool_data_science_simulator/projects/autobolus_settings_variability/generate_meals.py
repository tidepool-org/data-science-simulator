import random
from datetime import datetime, timedelta
import pandas as pd
import os
import shutil
import json


def get_duration_from_scenario():
    try:
        with open('icgm_to_scenario.py', 'r') as file:
            content = file.read()
            duration_str = "duration_hours = "
            start_index = content.find(duration_str)
            if start_index != -1:
                end_index = content.find('\n', start_index)
                duration_line = content[start_index:end_index].strip()
                # Extract just the number by splitting on '#' and taking first part
                duration_value = duration_line.split('=')[1].split('#')[0].strip()
                duration = float(duration_value)
                print(f"Found duration: {duration} hours")
                return duration
    except Exception as e:
        print(f"Error reading duration: {e}")
        return 24.0  # Default fallback

    return 24.0  # Default fallback


def generate_random_meals():
    # Set the start date/time and get duration from scenario file
    start_datetime = datetime.strptime("8/15/2019 12:00:00", "%m/%d/%Y %H:%M:%S")
    duration_hours = get_duration_from_scenario()
    end_datetime = start_datetime + timedelta(hours=duration_hours)

    current_date = start_datetime
    all_meal_times = []
    all_meal_sizes = []

    while current_date < end_datetime:
        # Generate 2-4 meals for each day
        num_meals = random.randint(2, 4)

        # For first day, start from current time
        if current_date.date() == start_datetime.date():
            earliest_time = current_date.hour
            earliest_minute = current_date.minute
        else:
            earliest_time = 5  # 5 AM
            earliest_minute = 0

        # Generate meals for this day
        day_meals = 0
        while day_meals < num_meals:
            random_hour = random.randint(earliest_time, 22)  # Between earliest time and 10 PM
            random_minute = random.randint(earliest_minute if random_hour == earliest_time else 0, 59)

            meal_datetime = current_date.replace(hour=random_hour, minute=random_minute)

            if meal_datetime >= start_datetime and meal_datetime < end_datetime:
                meal_datetime_str = meal_datetime.strftime("%m/%d/%Y %H:%M:%S")
                meal_size = random.randint(20, 80)

                all_meal_times.append(meal_datetime_str)
                all_meal_sizes.append(meal_size)
                day_meals += 1

        # Move to next day
        current_date = current_date + timedelta(days=1)
        current_date = current_date.replace(hour=0, minute=0)

    return all_meal_times, all_meal_sizes


def update_csv_with_meals_and_boluses(file_path):
    try:
        # Read the existing CSV file
        df = pd.read_csv(file_path)

        # Generate random meals
        meal_times, meal_sizes = generate_random_meals()

        # Create new rows for the meals and boluses
        new_rows = []

        # Add carb_dates
        for time in meal_times:
            new_rows.append({
                'setting_name': 'carb_dates',
                'settings': time
            })

        # Add actual_carbs
        for value in meal_sizes:
            new_rows.append({
                'setting_name': 'actual_carbs',
                'settings': value
            })

        # Add dose start times (same as meal times)
        for time in meal_times:
            new_rows.append({
                'setting_name': 'dose_start_times',
                'settings': time
            })

        # Add dose end times (same as meal times)
        for time in meal_times:
            new_rows.append({
                'setting_name': 'dose_end_times',
                'settings': time
            })

        # Add dose values (accept_recommendation for each meal)
        for _ in meal_times:
            new_rows.append({
                'setting_name': 'dose_values',
                'settings': 'accept_recommendation'
            })

        # Add dose types (bolus for each meal)
        for _ in meal_times:
            new_rows.append({
                'setting_name': 'dose_types',
                'settings': 'bolus'
            })

        # Remove existing meal and bolus related rows if they exist
        df = df[~df['setting_name'].isin([
            'carb_dates', 'actual_carbs',
            'dose_start_times', 'dose_end_times', 'dose_values', 'dose_types'
        ])]

        # Add new rows
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        df.to_csv(file_path, index=False)
        print(f"Updated file: {file_path}")
        print(f"Total meals generated: {len(meal_times)}")

        # Print sample of meal and bolus data (first 5 entries)
        print("\nSample of new meal and bolus data (first 5 entries):")
        meal_rows = df[df['setting_name'].isin([
            'carb_dates', 'actual_carbs',
            'dose_start_times', 'dose_end_times', 'dose_values', 'dose_types'
        ])]
        print(meal_rows.head())

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")


def create_test_file(source_file, test_directory):
    # Create test directory if it doesn't exist
    os.makedirs(test_directory, exist_ok=True)

    # Create test file name
    test_file = os.path.join(test_directory, 'test_' + os.path.basename(source_file))

    # Copy the source file to create test file
    shutil.copy2(source_file, test_file)
    print(f"Created test file: {test_file}")

    return test_file


def update_csv_with_meals_and_boluses(file_path):
    try:
        # Read the existing CSV file
        df = pd.read_csv(file_path)
        print(f"Initial carb dates: {df[df['setting_name'] == 'carb_dates']['settings'].tolist()}")

        # Generate random meals
        meal_times, meal_sizes = generate_random_meals()
        print(f"New generated meal times: {meal_times[:2]}")  # Show first 2 for brevity

        # Remove existing rows
        df = df[~df['setting_name'].isin([
            'carb_dates', 'actual_carbs',
            'dose_start_times', 'dose_end_times', 'dose_values', 'dose_types'
        ])]
        print(
            f"After removal - any remaining carb dates?: {df[df['setting_name'] == 'carb_dates']['settings'].tolist()}")

        # Create new rows for the meals and boluses
        new_rows = []

        # Add carb_dates
        for time in meal_times:
            new_rows.append({
                'setting_name': 'carb_dates',
                'settings': time
            })

        # Add actual_carbs
        for value in meal_sizes:
            new_rows.append({
                'setting_name': 'actual_carbs',
                'settings': value
            })

        # Add dose start times (same as meal times)
        for time in meal_times:
            new_rows.append({
                'setting_name': 'dose_start_times',
                'settings': time
            })

        # Add dose end times (same as meal times)
        for time in meal_times:
            new_rows.append({
                'setting_name': 'dose_end_times',
                'settings': time
            })

        # Add dose values (accept_recommendation for each meal)
        for _ in meal_times:
            new_rows.append({
                'setting_name': 'dose_values',
                'settings': 'accept_recommendation'
            })

        # Add dose types (bolus for each meal)
        for _ in meal_times:
            new_rows.append({
                'setting_name': 'dose_types',
                'settings': 'bolus'
            })

        # Add new rows
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

        # Save the updated DataFrame back to the CSV
        df.to_csv(file_path, index=False)
        print(f"Updated file: {file_path}")
        print(f"Total meals generated: {len(meal_times)}")

        # Print final carb dates to verify
        print(f"Final carb dates: {df[df['setting_name'] == 'carb_dates']['settings'].tolist()[:2]}")  # Show first 2

    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

def main():
    source_directory = "/Users/shawnfoster/data/virtual_patients/json_ready"
    test_directory = "/Users/shawnfoster/data/virtual_patients/test"

    # Test mode with single file
    TEST_MODE = False  # Set to False to process all files

    if TEST_MODE:
        source_file = os.path.join(source_directory, "icgm_user_666.csv")
        if os.path.exists(source_file):
            test_file = create_test_file(source_file, test_directory)
            print("\nProcessing test file...")
            update_csv_with_meals_and_boluses(test_file)
        else:
            print(f"Error: Source file not found: {source_file}")
    else:
        # Process all files in the directory
        csv_files = [f for f in os.listdir(source_directory) if f.endswith('.csv')]
        for file_name in csv_files:
            file_path = os.path.join(source_directory, file_name)
            print(f"\nProcessing: {file_name}")
            update_csv_with_meals_and_boluses(file_path)


if __name__ == "__main__":
    main()