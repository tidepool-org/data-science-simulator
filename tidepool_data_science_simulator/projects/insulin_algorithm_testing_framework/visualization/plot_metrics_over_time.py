import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import os

DATA_DIR = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/510k_short_run_example_mitigation_RC_momentum'
TIMESERIES_PATH = os.path.join(DATA_DIR, 'cumulative_sum_insulin.npy')
POINT_METRICS_PATH = os.path.join(DATA_DIR, 'point_metrics.csv')

FONT_SIZE = 22

def load_and_compare_gradthresh_groups(timeseries_path: str, point_metrics_path: str):
    """
    Load point metrics and timeseries data to compare different gradual transition threshold (gradthresh) values.
    
    Args:
        timeseries_path (str): Path to the .npy file containing 2D matrix with dimensions [n_simulation, time]
        point_metrics_path (str): Path to the CSV file containing point metrics with gradthresh values
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
        required_columns = ['gradthresh', 'alg']
        missing_columns = [col for col in required_columns if col not in point_metrics_df.columns]
        if missing_columns:
            print(f"Error: Required columns not found in point metrics: {missing_columns}")
            print(f"Available columns: {list(point_metrics_df.columns)}")
            return
        
        # Validate dimensions match
        if len(point_metrics_df) != timeseries_data.shape[0]:
            print(f"Error: Dimension mismatch between point metrics ({len(point_metrics_df)}) and timeseries ({timeseries_data.shape[0]})")
            return
        
        # Get unique algorithm and gradthresh values
        unique_algorithms = sorted(point_metrics_df['alg'].unique())
        unique_gradthresh = sorted(point_metrics_df['gradthresh'].unique())
        print(f"\nUnique algorithms: {unique_algorithms}")
        print(f"Unique gradthresh values: {unique_gradthresh}")
        
        # Create a dictionary to store data for each algorithm-gradthresh combination
        combined_groups = {}
        for alg in unique_algorithms:
            for gradthresh_val in unique_gradthresh:
                mask = (point_metrics_df['alg'] == alg) & (point_metrics_df['gradthresh'] == gradthresh_val)
                group_key = (alg, gradthresh_val)
                combined_groups[group_key] = timeseries_data[mask]
        
        print(f"\nGroup sizes:")
        for (alg, gradthresh_val), data in sorted(combined_groups.items()):
            print(f"alg={alg}, gradthresh={gradthresh_val}: {data.shape[0]} simulations")
        
        # Filter out empty groups
        combined_groups = {k: v for k, v in combined_groups.items() if v.shape[0] > 0}
        
        # Apply filtering: keep tempbasal (all gradthresh) or autobolus with gradthresh=40
        filtered_groups = {}
        for (alg, gradthresh_val), data in combined_groups.items():
            if alg == 'tempbasal' or (alg == 'autobolus' and gradthresh_val == 40.0):
                filtered_groups[(alg, gradthresh_val)] = data
        
        combined_groups = filtered_groups
        
        if not combined_groups:
            print("Warning: No valid simulation groups found after filtering")
            return
        
        # Calculate statistics for each group
        n_timepoints = timeseries_data.shape[1]
        
        # Convert timesteps to time in hours (each timestep = 5 minutes)
        time_minutes = np.arange(n_timepoints) * 5  # Convert to minutes
        time_hours = time_minutes / 60  # Convert to hours
        
        # Calculate means and stds for each algorithm-gradthresh group
        group_stats = {}
        for (alg, gradthresh_val), data in combined_groups.items():
            group_stats[(alg, gradthresh_val)] = {
                'mean': np.nanmean(data, axis=0),
                'std': np.nanstd(data, axis=0),
                'median': np.nanmedian(data, axis=0),
                '25th_percentile': np.nanpercentile(data, 25, axis=0),
                '75th_percentile': np.nanpercentile(data, 75, axis=0),
                'n': data.shape[0],
                'alg': alg,
                'gradthresh': gradthresh_val
            }
        
        # Create the comparison plot
        plt.figure(figsize=(12, 8))
        
        # Define colors for each gradthresh value
        colors = ['#627cfb', '#281b47']
        # Define line styles for each algorithm
        linestyles = {'tempbasal': '-', 'autobolus': '-'}
        
        # Plot with error bands for each algorithm-gradthresh group
        gradthresh_color_map = {gt: colors[i % len(colors)] for i, gt in enumerate(sorted(unique_gradthresh))}
        
        # Hard-coded legend labels
        legend_labels = [
            'Temp Basal',
            'Autobolus'
        ]
        
        # Hard-coded legend labels
        legend_labels = [
            'Temp Basal',
            'Autobolus'
        ]

        for i, ((key, stats), legend_label) in enumerate(zip(sorted(group_stats.items()), legend_labels)):
            alg, gradthresh_val = key
            color = gradthresh_color_map[gradthresh_val]
            linestyle = linestyles.get(alg, '-')
            
            plt.plot(time_hours, stats['median'], linestyle, color=color, label=legend_label, linewidth=2)
            # Ensure lower bound doesn't go below 0 for cumulative insulin
            lower_bound = np.maximum(stats['mean'] - stats['std'], 0)
            # plt.fill_between(time_hours, lower_bound, stats['mean'] + stats['std'], 
            #                alpha=0.15, color=color)
            plt.fill_between(time_hours, stats['25th_percentile'], stats['75th_percentile'], 
                        alpha=0.15, color=color)
        
        # Customize the plot
        plt.xlabel('Time (Hours)', fontsize=18)
        plt.ylabel('Cumulative Insulin (Units)', fontsize=18)
        plt.title('Cumulative Insulin Over Time', fontsize=FONT_SIZE, fontweight='bold')
        
        # Set x-ticks at hourly intervals
        max_hours = time_hours[-1]
        hour_ticks = np.arange(0, int(max_hours) + 1, 1)  # Every hour
        plt.xticks(hour_ticks, fontsize=16)
        plt.yticks(fontsize=16)
        
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=14, loc='best')
        
        plt.tight_layout()
        
        # Display statistics
        print(f"\nComparison Statistics:")
        for (alg, gradthresh_val), stats in sorted(group_stats.items()):
            print(f"\nalg={alg}, gradthresh={gradthresh_val}:")
            print(f"  - Number of simulations: {stats['n']}")
            print(f"  - Mean cumulative insulin at end: {stats['mean'][-1]:.2f} ± {stats['std'][-1]:.2f} units")
            print(f"  - Maximum mean cumulative insulin: {stats['mean'].max():.2f} units")
        
        plt.show()
        
    except Exception as e:
        print(f"Error processing files: {str(e)}")
        return

def main():
    """
    Main function to run the plotting script.
    """
    print("Comparing cumulative insulin metrics across algorithms and gradual transition thresholds...")
    load_and_compare_gradthresh_groups(TIMESERIES_PATH, POINT_METRICS_PATH)

if __name__ == "__main__":
    main()
