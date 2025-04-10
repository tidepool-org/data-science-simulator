import os
import glob
import numpy as np
import pandas as pd
from tidepool_data_science_simulator.evaluation.inspect_results import load_results, load_result, collect_sims_and_results
from tidepool_data_science_simulator.utils import DATA_DIR

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_metrics.glucose.glucose import percent_values_ge_70_le_180, blood_glucose_risk_index


def calculate_cumulative_insulin(df):
    """
    Calculate cumulative insulin delivered from a DataFrame.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing insulin data with columns 'delivered_basal_insulin' and 'true_bolus'.
    
    Returns:
    tuple: Cumulative basal and bolus insulin.
    """
    cumulative_basal = df['delivered_basal_insulin'].sum()
    cumulative_bolus = df['true_bolus'].sum()
    return cumulative_basal + cumulative_bolus

def calculate_metrics(df):
    tir = percent_values_ge_70_le_180(df['bg'])
    cumulative_insulin = calculate_cumulative_insulin(df)
    bgri = blood_glucose_risk_index(df['bg'])[2]

    return tir, cumulative_insulin, bgri

# Directory containing the files
directory = '/Users/mconn/tidepool/repositories/data-science-simulator/tidepool_data_science_simulator/projects/autobolus_application_factor/'

processed_dir = os.path.join(DATA_DIR, "processed/")
result_dir = 'no_meal_announcements_2025_03_17_T_15_09_04/'
result_path = os.path.join(processed_dir, result_dir)

# Get list of files in the directory
files_0_0 = glob.glob(os.path.join(result_path, '*paf=0.0*tsv'))
files_0_4 = glob.glob(os.path.join(result_path, '*paf=0.4*tsv'))

# Create pairs of files with 0.0 and 0.4 in the name
pairs = {}
for file in files_0_0:
    base_name = os.path.basename(file).replace('paf=0.0', '')
    pairs[base_name] = [file, None]

for file in files_0_4:
    base_name = os.path.basename(file).replace('paf=0.4', '')
    if base_name in pairs:
        pairs[base_name][1] = file
    else:
        pairs[base_name] = [None, file]

metrics_all = np.zeros((len(pairs), 3, 2))  # Initialize a numpy array to store metrics

# Loop through each pair and plot the insulin on board
for base_name, (file_0_0, file_0_4) in pairs.items():
    if file_0_0 and file_0_4:
        result_data_0_0 = load_result(file_0_0)
        result_data_0_4 = load_result(file_0_4)

        df_0_0 = result_data_0_0[1]
        df_0_4 = result_data_0_4[1]

        # Extract the vp=X part from the filename
        label_0_0 = os.path.basename(file_0_0).split('_')[0]
        label_0_4 = os.path.basename(file_0_4).split('_')[0]
        
        # Calculate cumulative delivered basal and bolus insulin
        cumulative_basal_0_0 = df_0_0['delivered_basal_insulin'].sum()
        cumulative_bolus_0_0 = df_0_0['true_bolus'].sum()
        cumulative_basal_0_4 = df_0_4['delivered_basal_insulin'].sum()
        cumulative_bolus_0_4 = df_0_4['true_bolus'].sum()

        # Calculate overall cumulative delivered insulin
        overall_cumulative_0_0 = cumulative_basal_0_0 + cumulative_bolus_0_0
        overall_cumulative_0_4 = cumulative_basal_0_4 + cumulative_bolus_0_4

        # Print the cumulative values
        print(f'{label_0_0} paf=0.0 - Cumulative Basal: {cumulative_basal_0_0}, Cumulative Bolus: {cumulative_bolus_0_0}, Overall: {overall_cumulative_0_0}')
        print(f'{label_0_4} paf=0.4 - Cumulative Basal: {cumulative_basal_0_4}, Cumulative Bolus: {cumulative_bolus_0_4}, Overall: {overall_cumulative_0_4}')

        # Prepare the results dictionary for plotting
        all_results = {
            f'{label_0_0} paf=0.0': df_0_0,
            f'{label_0_4} paf=0.4': df_0_4
        }
        tir, cumulative_insulin, bgri = calculate_metrics(df_0_0)

        # Calculate metrics for both data frames
        metrics_0_0 = calculate_metrics(df_0_0)
        metrics_0_4 = calculate_metrics(df_0_4)

        # Store each metric pair in a separate numpy array
        metrics_array = np.array([
            [metrics_0_0[0], metrics_0_4[0]],  # TIR
            [metrics_0_0[1], metrics_0_4[1]],  # Cumulative Insulin
            [metrics_0_0[2], metrics_0_4[2]]  # BGRI
        ])

        metrics_all[list(pairs.keys()).index(base_name)] = metrics_array

        # Plot the results using plot_sim_results
        # plot_sim_results(all_results)
    
# Create box plots comparing each metric between the two conditions
metric_names = ['Time in Range (TIR)', 'Cumulative Insulin', 'Blood Glucose Risk Index (BGRI)']
condition_labels = ['paf=0.0', 'paf=0.4']

# Create a figure for the box plots
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)

for i, metric_name in enumerate(metric_names):
    # Extract the metric values for both conditions
    metric_values_0_0 = metrics_all[:, i, 0]
    metric_values_0_4 = metrics_all[:, i, 1]

    # Create a box plot for the current metric
    axes[i].boxplot([metric_values_0_0, metric_values_0_4], labels=condition_labels)
    axes[i].set_title(metric_name)
    axes[i].set_ylabel('Value')
    axes[i].grid(True, linestyle='--', alpha=0.7)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()