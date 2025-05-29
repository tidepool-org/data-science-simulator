import os
import glob
import numpy as np
import pandas as pd
from tidepool_data_science_simulator.evaluation.inspect_results import load_results, load_result, collect_sims_and_results
from tidepool_data_science_simulator.utils import DATA_DIR

import matplotlib.pyplot as plt
import seaborn as sns

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_metrics.glucose.glucose import percent_values_ge_70_le_180, blood_glucose_risk_index
from scipy.stats import lognorm
from scipy.stats import ttest_ind
from scipy.stats import gaussian_kde


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
    """
    Calculate key metrics from a DataFrame containing blood glucose and insulin data.

    Parameters:
    df (pd.DataFrame): DataFrame containing simulation data with at least the following columns:
                       - 'bg': Blood glucose values.
                       - 'delivered_basal_insulin': Delivered basal insulin values.
                       - 'true_bolus': Delivered bolus insulin values.

    Returns:
    tuple: A tuple containing the following metrics:
           - tir (float): Time in Range (percentage of blood glucose values between 70 and 180 mg/dL).
           - cumulative_insulin (float): Total cumulative insulin delivered (basal + bolus).
           - bgri (float): Blood Glucose Risk Index (a measure of glucose variability and risk).
    """
    # Calculate Time in Range (TIR) as the percentage of blood glucose values within the target range (70-180 mg/dL)
    tir = percent_values_ge_70_le_180(df['bg'])

    # Calculate the total cumulative insulin delivered (sum of basal and bolus insulin)
    cumulative_insulin = calculate_cumulative_insulin(df)

    # Calculate the Blood Glucose Risk Index (BGRI), which quantifies glucose variability and risk
    bgri = blood_glucose_risk_index(df['bg'])[2]

    return tir, cumulative_insulin, bgri

# Directory containing the files
directory = '/Users/mconn/tidepool/repositories/data-science-simulator/tidepool_data_science_simulator/projects/autobolus_application_factor/'

processed_dir = os.path.join(DATA_DIR, "processed/")
result_dir = 'autobolus_tempbasal_comparison_2025_04_10_T_09_52_00/'
result_path = os.path.join(processed_dir, result_dir)

# Get list of files in the directory
files_0_0 = glob.glob(os.path.join(result_path, '*paf=0.0*tsv'))
files_0_4 = glob.glob(os.path.join(result_path, '*paf=0.4*tsv'))

# Group files by user and IBG
pairs = {}
for file in files_0_0 + files_0_4:
    base_name = os.path.basename(file)
    base_name = base_name.replace(".tsv", "")

    # Extract vp, patient_id, and ibg explicitly from the file name by splitting out tags, e.g., "vp=1_patient_id=1_ibg=0.0_paf=0.4"
    parts = [seg for seg in base_name.split("_") if "=" in seg]
    data = dict(seg.split("=", 1) for seg in parts) 

    user_ibg_key = f"vp={data['vp']}_patient_id={data['id']}_ibg={data['ibg']}"
    paf_value = "paf=0.0" if "paf=0.0" in base_name else "paf=0.4"

    if user_ibg_key not in pairs:
        pairs[user_ibg_key] = {"paf=0.0": [], "paf=0.4": []}

    pairs[user_ibg_key][paf_value].append(file)

# Load the histogram data
histogram_file = '/Users/mconn/Downloads/BG_Distribution_Histogram.csv'
histogram_df = pd.read_csv(histogram_file)

# Initialize a numpy array to store metrics
metrics_all = np.zeros((len(pairs), 3, 2)) 

# Initialize an array to store IBG values corresponding to each metric
ibg_values = np.zeros(len(pairs))

# Initialize weights array
weights = np.zeros(len(pairs))

# Loop through each user/IBG group and process files
for idx, (user_ibg_key, paf_files) in enumerate(pairs.items()):
    if paf_files["paf=0.0"] and paf_files["paf=0.4"]:
        dfs_0_0 = [load_result(file)[1] for file in paf_files["paf=0.0"]]
        dfs_0_4 = [load_result(file)[1] for file in paf_files["paf=0.4"]]

        # Combine dataframes for each PAF value
        df_0_0 = pd.concat(dfs_0_0, ignore_index=True)
        df_0_4 = pd.concat(dfs_0_4, ignore_index=True)

        # Calculate metrics for both conditions
        metrics_0_0 = calculate_metrics(df_0_0)
        metrics_0_4 = calculate_metrics(df_0_4)

        # Extract IBG value from the user_ibg_key
        ibg = float(user_ibg_key.split("_ibg=")[1])
        ibg_values[idx] = ibg

        # Get the proportion corresponding to the IBG
        proportion = histogram_df.loc[histogram_df['ibg'] == ibg, 'proportion']
        if proportion.empty:
            raise ValueError(f"No matching proportion found for IBG={ibg} in the histogram file.")
        weights[idx] = proportion.values[0]

        # Store metrics in the numpy array
        metrics_all[idx] = [
            [metrics_0_0[0], metrics_0_4[0]],  # TIR
            [metrics_0_0[1], metrics_0_4[1]],  # Cumulative Insulin
            [metrics_0_0[2], metrics_0_4[2]]   # BGRI
        ]


# Directory to save plots
output_dir = '/Users/mconn/Library/CloudStorage/GoogleDrive-mark.connolly@tidepool.org/My Drive/projects/Sensitivity Analysis/processed_data/compare_tempbasal_autobolus'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Create weighted box plots comparing each metric between the two conditions
metric_names = ['Time in Range (TIR)', 'Cumulative Insulin', 'Blood Glucose Risk Index (BGRI)']
condition_labels = ['Temp Basal', 'Autobolus (paf=0.4)']

# Initialize a dictionary to store statistical details
statistical_details = {}

# Without weighting
for i, metric_name in enumerate(metric_names):
    metric_values_0_0 = metrics_all[:, i, 0]
    metric_values_0_4 = metrics_all[:, i, 1]

    # Perform a t-test between the two conditions
    t_stat, p_value = ttest_ind(metric_values_0_0, metric_values_0_4, equal_var=False)

    # Save statistical details
    statistical_details[metric_name] = {
        "Without Scaling": {
            "Temp Basal": {"mean": np.mean(metric_values_0_0), "std": np.std(metric_values_0_0)},
            "Autobolus (paf=0.4)": {"mean": np.mean(metric_values_0_4), "std": np.std(metric_values_0_4)},
            "t_stat": t_stat,
            "p_value": p_value
        }
    }

    # Save the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    # ax.boxplot([metric_values_0_0, metric_values_0_4], labels=condition_labels)
    x_grid = np.linspace(metric_values_0_0.min() - 1, metric_values_0_0.max() + 1, 500)
    kde_0_0 = gaussian_kde(metric_values_0_0, weights=weights, bw_method=0.2)
    kde_0_4 = gaussian_kde(metric_values_0_4, weights=weights, bw_method=0.2)

    ax.plot(x_grid, kde_0_0(x_grid), label='Temp Basal', color='blue')
    ax.set_title(f"{metric_name} (Without Scaling)\n(p={p_value:.2e})")
    ax.set_ylabel('Value')
    ax.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{metric_name.replace(' ', '_')}_without_scaling.png"))
    plt.close(fig)
    plt.show()

# # With weighting
# for i, metric_name in enumerate(metric_names):
#     metric_values_0_0 = np.repeat(metrics_all[:, i, 0], (weights * 10000).astype(int))
#     metric_values_0_4 = np.repeat(metrics_all[:, i, 1], (weights * 10000).astype(int))

#     # Perform a t-test between the two conditions
#     t_stat, p_value = ttest_ind(metric_values_0_0, metric_values_0_4, equal_var=False)

#     # Save statistical details
#     statistical_details[metric_name]["With Scaling"] = {
#         "Temp Basal": {"mean": np.mean(metric_values_0_0), "std": np.std(metric_values_0_0)},
#         "Autobolus (paf=0.4)": {"mean": np.mean(metric_values_0_4), "std": np.std(metric_values_0_4)},
#         "t_stat": t_stat,
#         "p_value": p_value
#     }

#     # Save the plot
#     fig, ax = plt.subplots(figsize=(5, 5))
#     ax.boxplot([metric_values_0_0, metric_values_0_4], labels=condition_labels)
#     ax.set_title(f"{metric_name} (With Scaling)\n(p={p_value:.2e})")
#     ax.set_ylabel('Value')
#     ax.grid(True, linestyle='--', alpha=0.7)
#     plt.tight_layout()
#     # plt.savefig(os.path.join(output_dir, f"{metric_name.replace(' ', '_')}_with_scaling.png"))
#     # plt.close(fig)
#     plt.show()
# # Output statistical details
# for metric_name, details in statistical_details.items():
#     print(f"Metric: {metric_name}")
#     for scaling, stats in details.items():
#         print(f"  {scaling}:")
#         print(f"    Temp Basal - Mean: {stats['Temp Basal']['mean']:.2f}, Std: {stats['Temp Basal']['std']:.2f}")
#         print(f"    Autobolus (paf=0.4) - Mean: {stats['Autobolus (paf=0.4)']['mean']:.2f}, Std: {stats['Autobolus (paf=0.4)']['std']:.2f}")
#         print(f"    t_stat: {stats['t_stat']:.2f}, p_value: {stats['p_value']:.2e}")
#     print()
