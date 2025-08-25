import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

TIMESERIES_PATH = "/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/unannounced_meals/cumulative_sum_insulin.npy"
POINT_METRICS_PATH = "/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/unannounced_meals/point_metrics.csv"

FONT_SIZE = 22
def load_and_compare_paf_posvel_groups(timeseries_path: str, point_metrics_path: str):
    """
    Load point metrics and timeseries data to compare Temp Basal+posvel=true vs PAF=0.4+posvel=false groups.
    
    Args:
        timeseries_path (str): Path to the .npy file containing 2D matrix with dimensions [n_simulation, time]
        point_metrics_path (str): Path to the CSV file containing point metrics with PAF and posvel values
    """
    # Check if files exist
    if not Path(timeseries_path).exists():
        print(f"Timeseries file not found: {timeseries_path}")
        return
        
    if not Path(point_metrics_path).exists():
        print(f"Point metrics file not found: {point_metrics_path}")
        return
    
    try:
        # Load the timeseries data (.npy file)
        timeseries_data = np.load(timeseries_path)
        print(f"Loaded timeseries data with shape: {timeseries_data.shape}")
        
        # Load the point metrics (CSV file)
        point_metrics_df = pd.read_csv(point_metrics_path)
        print(f"Loaded point metrics with shape: {point_metrics_df.shape}")
        
        # Check if required columns exist
        required_columns = ['paf', 'posvel']
        missing_columns = [col for col in required_columns if col not in point_metrics_df.columns]
        if missing_columns:
            print(f"Error: Required columns not found in point metrics: {missing_columns}")
            print(f"Available columns: {list(point_metrics_df.columns)}")
            return
        
        # Validate dimensions match
        if len(point_metrics_df) != timeseries_data.shape[0]:
            print(f"Error: Dimension mismatch between point metrics ({len(point_metrics_df)}) and timeseries ({timeseries_data.shape[0]})")
            return
        
        # Filter data for specific group comparisons
        # Group 1: Temp Basal AND posvel=true
        group1_mask = point_metrics_df['paf'].isna() & (point_metrics_df['posvel'] == True)
        group1_data = timeseries_data[group1_mask]
        
        # Group 2: PAF=0.4 AND posvel=false
        group2_mask = (point_metrics_df['paf'] == 0.4) & (point_metrics_df['posvel'] == False)
        group2_data = timeseries_data[group2_mask]
        
        print(f"\nGroup sizes:")
        print(f"Temp Basal + posvel=true: {group1_data.shape[0]} simulations")
        print(f"PAF=0.4 + posvel=false: {group2_data.shape[0]} simulations")
        
        if group1_data.shape[0] == 0:
            print("Warning: No simulations found with Temp Basal + posvel=true")
            return
            
        if group2_data.shape[0] == 0:
            print("Warning: No simulations found with PAF=0.4 + posvel=false")
            return
        
        # Calculate statistics for each group
        n_timepoints = timeseries_data.shape[1]
        
        # Convert timesteps to time in hours (each timestep = 5 minutes)
        time_minutes = np.arange(n_timepoints) * 5  # Convert to minutes
        time_hours = time_minutes / 60  # Convert to hours
        
        # Calculate statistics for each group
        group1_means = np.nanmean(group1_data, axis=0)
        group1_stds = np.nanstd(group1_data, axis=0)
        
        group2_means = np.nanmean(group2_data, axis=0)
        group2_stds = np.nanstd(group2_data, axis=0)
        
        # Create the comparison plot
        plt.figure(figsize=(14, 8))
        
        # Plot with error bands
        plt.plot(time_hours, group1_means, 'b-', label=f'Temp Basal + posvel=true (n={group1_data.shape[0]})', linewidth=2)
        plt.fill_between(time_hours, group1_means - group1_stds, group1_means + group1_stds, 
                         alpha=0.2, color='blue')
        
        plt.plot(time_hours, group2_means, 'r-', label=f'PAF=0.4 + posvel=false (n={group2_data.shape[0]})', linewidth=2)
        plt.fill_between(time_hours, group2_means - group2_stds, group2_means + group2_stds, 
                         alpha=0.2, color='red')
        
        # Customize the plot
        plt.xlabel('Time (Hours)', fontsize=18)
        plt.ylabel('Cumulative Insulin (Units)', fontsize=18)
        plt.title('Cumulative Insulin Over Time:\nTemp Basal+posvel=true vs PAF=0.4+posvel=false', fontsize=FONT_SIZE, fontweight='bold')
        
        # Set x-ticks at hourly intervals
        max_hours = time_hours[-1]
        hour_ticks = np.arange(0, int(max_hours) + 1, 1)  # Every hour
        plt.xticks(hour_ticks, fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=16)
        
        plt.tight_layout()
        
        # Display statistics
        print(f"\nComparison Statistics:")
        print(f"Temp Basal + posvel=true group:")
        print(f"  - Number of simulations: {group1_data.shape[0]}")
        print(f"  - Mean cumulative insulin at end: {group1_means[-1]:.2f} ± {group1_stds[-1]:.2f} units")
        print(f"  - Maximum mean cumulative insulin: {group1_means.max():.2f} units")
        
        print(f"\nPAF=0.4 + posvel=false group:")
        print(f"  - Number of simulations: {group2_data.shape[0]}")
        print(f"  - Mean cumulative insulin at end: {group2_means[-1]:.2f} ± {group2_stds[-1]:.2f} units")
        print(f"  - Maximum mean cumulative insulin: {group2_means.max():.2f} units")
        
        # Calculate difference
        end_diff = group2_means[-1] - group1_means[-1]
        print(f"\nDifference at end (PAF=0.4+posvel=false - Temp Basal+posvel=true): {end_diff:.2f} units")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return

def main():
    """
    Main function to run the plotting script.
    """
    print("Comparing cumulative insulin metrics: Temp Basal+posvel=true vs PAF=0.4+posvel=false...")
    load_and_compare_paf_posvel_groups(TIMESERIES_PATH, POINT_METRICS_PATH)

if __name__ == "__main__":
    main()
