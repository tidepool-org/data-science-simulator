import os
import pandas as pd
from datetime import datetime, timedelta


def process_csv_files(input_directory, output_directory):
    start_datetime = datetime(2019, 8, 15, 0, 40, 0)
    num_intervals = 137
    interval_minutes = 5

    os.makedirs(output_directory, exist_ok=True)

    for filename in os.listdir(input_directory):
        if filename.endswith('.csv'):
            input_path = os.path.join(input_directory, filename)
            try:
                df = pd.read_csv(input_path)
                glucose_df = df[df['setting_name'] == 'actual_blood_glucose']

                if not glucose_df.empty:
                    # Get numeric columns (excluding 'setting_name' and 'settings')
                    value_columns = [col for col in glucose_df.columns if col.isdigit()]
                    # Convert row to series and round values
                    values = glucose_df[value_columns].iloc[0].astype(float).round().astype(int)
                    values = values.head(num_intervals)

                    # Create timestamps
                    timestamps = [
                        (start_datetime + timedelta(minutes=interval_minutes * i)).strftime('%-m/%d/%Y %H:%M:%S')
                        for i in range(num_intervals)
                    ]

                    # Create output DataFrame
                    result_df = pd.DataFrame({
                        'datetime': timestamps[:len(values)],
                        'value': values,
                        'units': ['mg/dL'] * len(values)
                    })

                    output_path = os.path.join(output_directory, filename.replace('_trimmed.csv', '.csv'))
                    result_df.to_csv(output_path, index=False)
                    print(f"Processed {filename}")

            except Exception as e:
                print(f"Error processing {filename}: {e}")
                import traceback
                print(traceback.format_exc())


if __name__ == "__main__":
    input_directory = '/Users/shawnfoster/data/virtual_patients/glucose_history'
    output_directory = '/Users/shawnfoster/data/virtual_patients/glucose_history/trimmed'
    process_csv_files(input_directory, output_directory)