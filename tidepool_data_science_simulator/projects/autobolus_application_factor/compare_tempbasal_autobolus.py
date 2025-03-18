import os
import glob
import pandas as pd
from tidepool_data_science_simulator.evaluation.inspect_results import load_results, load_result, collect_sims_and_results
from tidepool_data_science_simulator.utils import DATA_DIR

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

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

        # Plot the results using plot_sim_results
        plot_sim_results(all_results)