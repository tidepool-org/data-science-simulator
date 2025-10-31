import os
import datetime
from pathlib import Path
import numpy as np
import itertools
import pandas as pd
# import seaborn as sns
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table, get_probability_index

data_dir = ''

data_names = [
    '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=False_original.csv',
    '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=True_original.csv',
    '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=True_maxjump=20_full.csv',
    # '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=True_maxjump=10_merged.csv',
    # '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=True_maxjump=10_wide_merged.csv',
    # '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=True_maxjump=25_merged.csv',
    # '/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_paf=0.4_posrc=True_maxjump=40_merged.csv',
]

# Initialize storage for all data
all_file_data = []
global_vmax_per_risk = {}
global_vmin_per_risk = {}

# FIRST PASS: Load all files and calculate global min/max for each risk level
print("="*80)
print("FIRST PASS: Loading all files and calculating global min/max")
print("="*80)

for i, data_name in enumerate(data_names):
    print(f"\nLoading file {i+1}/{len(data_names)}: {data_name}")
    
    data_path = data_dir + data_name
    try:
        summary_df = pd.read_csv(data_path, sep="\t")
        severity_event_probability_df, (low_icgm_axis, low_true_axis, mean_lbgi_swift_start, joint_prob_swift) = compute_score_risk_table(summary_df, concurrency_table='adult')

        severity_event_probability_df = severity_event_probability_df * 48
        
        risk_index_vals = [get_probability_index(p) for p in severity_event_probability_df[0]]
        risk_index_vals = np.array(risk_index_vals)
        
        # Calculate risk data for this file
        file_risk_data = {}
        for risk_index in range(2, 5):
            risk_lbgi_swift_start = mean_lbgi_swift_start[:, risk_index]
            a = risk_lbgi_swift_start * joint_prob_swift
            file_risk_data[risk_index] = a
            
            # Update global min/max
            if risk_index not in global_vmax_per_risk:
                global_vmax_per_risk[risk_index] = np.max(a)
                global_vmin_per_risk[risk_index] = np.min(a)
            else:
                global_vmax_per_risk[risk_index] = max(global_vmax_per_risk[risk_index], np.max(a))
                global_vmin_per_risk[risk_index] = min(global_vmin_per_risk[risk_index], np.min(a))
        
        # Store all data for this file
        all_file_data.append({
            'name': data_name,
            'severity_event_probability_df': severity_event_probability_df,
            'risk_index_vals': risk_index_vals,
            'low_icgm_axis': low_icgm_axis,
            'low_true_axis': low_true_axis,
            'mean_lbgi_swift_start': mean_lbgi_swift_start,
            'joint_prob_swift': joint_prob_swift,
            'risk_data': file_risk_data
        })
        
        print(f"  ✓ Successfully loaded")
        
    except FileNotFoundError:
        print(f"  ✗ ERROR: File not found: {data_path}")
        continue
    except Exception as e:
        print(f"  ✗ ERROR processing {data_name}: {str(e)}")
        continue

print("\n" + "="*80)
print("Global min/max values for each risk level:")
for risk_index in range(2, 5):
    print(f"  Risk {risk_index+1}: min={global_vmin_per_risk[risk_index]:.6f}, max={global_vmax_per_risk[risk_index]:.6f}")
print("="*80)

# SECOND PASS: Plot all files with shared z-scales
lw = 2
ticks = [40, 61, 81, 121, 161, 201, 251, 301, 351]
ticklabels = ['40-60','61-80','81-120','121-160','161-200','201-250','251-300','301-350','351-400']
rotation = 20

for file_idx, file_data in enumerate(all_file_data):
    print(f"\n{'='*80}")
    print(f"Plotting file {file_idx+1}/{len(all_file_data)}: {file_data['name']}")
    print(f"{'='*80}")
    
    print('Severity Event Probability')
    print(file_data['severity_event_probability_df'])
    print()
    
    print('Risk Scores')
    print(file_data['risk_index_vals'] * np.array([1,2,3,4,5]))
    print()
    
    # Create plots with global z-scales
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    for risk_index in range(2, 5):
        plot_idx = risk_index - 2
        a = file_data['risk_data'][risk_index]
        
        dim = int(np.sqrt(len(file_data['low_icgm_axis'])))
        dims = (dim, dim)
        
        true_grid = np.reshape(file_data['low_true_axis'], dims)
        icgm_grid = np.reshape(file_data['low_icgm_axis'], dims)
        
        # Use GLOBAL vmin and vmax for this risk level
        ax[plot_idx].pcolormesh(true_grid, icgm_grid, np.reshape(a, dims), 
                                vmin=global_vmin_per_risk[risk_index], 
                                vmax=global_vmax_per_risk[risk_index], 
                                edgecolors='k', linewidths=lw)
        ax[plot_idx].invert_yaxis()
        
        ax[plot_idx].set_xlabel("True Blood Glucose")
        ax[plot_idx].set_xticks(ticks)
        ax[plot_idx].set_yticks(ticks)
        ax[plot_idx].set_title('Risk Severity: {}'.format(risk_index+1))
    
    ax[0].set_ylabel("Sensor Blood Glucose")
    
    # Add filename to figure title
    fig.suptitle(f"File: {os.path.basename(file_data['name'])}", fontsize=10, y=1.00)
    
    plt.tight_layout()
plt.show()