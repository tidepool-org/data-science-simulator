import pandas as pd
import numpy as np

file_path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/510k_short_run_example_mitigation_RC_momentum/point_metrics.csv'

# Load the CSV file
print(f"Loading data from: {file_path}")
df = pd.read_csv(file_path)

print(f"\nLoaded {len(df)} rows")
print(f"Columns: {df.columns.tolist()}")

# Check if required columns exist
if 'gradthresh' not in df.columns:
    print("\nERROR: 'gradthresh' column not found in dataframe")
    print(f"Available columns: {df.columns.tolist()}")
elif 'time_in_range_70_180' not in df.columns:
    print("\nERROR: 'time_in_range_70_180' column not found in dataframe")
    print(f"Available columns: {df.columns.tolist()}")
else:
    # Calculate mean TIR for each gradthresh value
    mean_tir_by_gradthresh = df.groupby('gradthresh')['time_in_range_70_180'].mean()
    
    print("\n" + "="*50)
    print("Mean TIR by Gradual Transition Threshold:")
    print("="*50)
    for gradthresh, mean_tir in mean_tir_by_gradthresh.items():
        print(f"gradthresh = {gradthresh:.1f} mg/dL/min: Mean TIR = {mean_tir:.2f}%")
    
    # Additional statistics
    print("\n" + "="*50)
    print("Detailed Statistics:")
    print("="*50)
    stats = df.groupby('gradthresh')['time_in_range_70_180'].agg(['count', 'mean', 'std', 'min', 'max'])
    print(stats)
    
    # Calculate count of simulations per gradthresh
    print("\n" + "="*50)
    print("Number of simulations per gradthresh:")
    print("="*50)
    counts = df.groupby('gradthresh').size()
    print(counts)
