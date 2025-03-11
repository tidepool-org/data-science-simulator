import argparse
import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table, get_probability_index
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.float_format', '{:.2e}'.format)
# risk_idx = 2

# Caclulate risk for 2021 Manual Recommendation with positive RC and momentum disabled
# path = '/Users/mconn/data/simulator/processed_data/sim_summary_df_MITIGATED_Aug12_2021.csv'
# summary_df_2021 = pd.read_csv(path, sep=",")
# severity_event_probability_df_2021, (low_icgm_axis_2021, low_true_axis_2021, mean_lbgi_2021_start, mean_lbgi_2021_valid, joint_prob_2021) = compute_score_risk_table(summary_df_2021)

# severity_event_probability_df_2021 = severity_event_probability_df_2021 #* 48
# print('Severity Event Probability')
# print(severity_event_probability_df_2021)

# risk_index = [get_probability_index(p) for p in severity_event_probability_df_2021[0]]
# risk_index = np.array(risk_index)
# print('Risk Scores')
# print(risk_index * np.array([1,2,3,4,5]))

# risk_lbgi_2021 = mean_lbgi_2021_valid[:,risk_idx]

data_dir = '/Users/mconn/Google Drive/My Drive/projects/Sensitivity Analysis/processed_data/'

data_name = 'icgm_sensitivity_analysis_results_AUTOBOLUS_04_2024_11_20_619381e.csv'
data_name = 'icgm_sensitivity_analysis_results_AUTOBOLUS_04_positive_bias_correction_2024_12_12_T_18_32_28_7dd54a9.csv'
data_name = 'icgm_sensitivity_analysis_results_TEMPBASAL_NO_MANUAL_2024_12_20_T_14_04_39_748cf5e.csv'
# data_name = 'icgm_sensitivity_analysis_results_AUTOBOLUS_05_positive_bias_correction_2025_01_13_T_13_21_21_76026e7.csv'
# data_name = 'icgm_sensitivity_analysis_results_AUTOBOLUS_06_positive_bias_correction_2025_01_16_T_03_50_48_eed0c90.csv'
# data_name = 'icgm_sensitivity_analysis_results_MANUAL_2024_11_13_0da14e7.csv'
# data_name = 'icgm_sensitivity_analysis_results_MANUAL_BOLUS_positive_bias_correction_2025_02_03_T_16_12_06_d997c998.csv'

data_path = data_dir + data_name
summary_df = pd.read_csv(data_path, sep="\t")
severity_event_probability_df, (low_icgm_axis, low_true_axis, mean_lbgi_swift_start, joint_prob_swift) = compute_score_risk_table(summary_df, concurrency_table='adult')

severity_event_probability_df = severity_event_probability_df #* 48
print('Severity Event Probability')
print(severity_event_probability_df)
print()

risk_index = [get_probability_index(p) for p in severity_event_probability_df[0]]
risk_index = np.array(risk_index)
print('Risk Scores')
print(risk_index * np.array([1,2,3,4,5]))

lw = 2 
# ticks = [40, 61, 81, 121, 161, 201, 251, 301, 351]
# ticklabels = ['40-60','61-80','81-120','121-160','161-200','201-250','251-300','301-350','351-400']
ticks = range(40, 400, 5)
rotation = 20
vmax = [0.01583,
        0.00170,
        6.01944e-05]
fig, ax = plt.subplots(1,3,figsize=(18, 6))

for risk_index in range(2, 5):

    plot_idx = risk_index-2
    risk_lbgi_swift_start = mean_lbgi_swift_start[:,risk_index]

    a = risk_lbgi_swift_start * joint_prob_swift
    # a = np.concatenate(([0,0],a))
    # a[0:2] = [0, 0]
    vmin = min(a)
    # vmax = max(a)
    print(vmax)                          
    dim = int(np.sqrt(len(low_icgm_axis)))
    dims = (dim, dim)

    true_grid = np.reshape(low_true_axis, dims)
    icgm_grid = np.reshape(low_icgm_axis, dims)
    
    ax[plot_idx].pcolormesh(true_grid, icgm_grid, np.reshape(a, dims), vmin=vmin, vmax=vmax[plot_idx], edgecolors='k', linewidths=lw)
    ax[plot_idx].invert_yaxis()    

    ax[plot_idx].set_xlabel("True Blood Glucose")
    
    
    ax[plot_idx].set_xticks(ticks)
    # ax[plot_idx].set_xticklabels(ticklabels, rotation=rotation)

    ax[plot_idx].set_yticks(ticks)
    # ax[plot_idx].set_yticklabels(ticklabels)
    
    ax[plot_idx].set_title('Risk Severity: {}'.format(risk_index+1))

ax[0].set_ylabel("Sensor Blood  Glucose")
plt.show()

#####
#####
# lw = 2 
# fig = plt.figure()

# axes = [fig.add_subplot(3, 2, i) for i in range(1,7)]

# dim = int(np.sqrt(len(low_icgm_axis)))
# dims = (dim, dim)

# true_grid = np.reshape(low_true_axis, dims)
# icgm_grid = np.reshape(low_icgm_axis, dims)


# axes[0].pcolormesh(true_grid, icgm_grid, np.reshape(risk_lbgi_2021, dims), vmin=0, vmax=1, edgecolors='k', linewidths=lw)
# axes[1].pcolormesh(true_grid, icgm_grid, np.reshape(risk_lbgi_swift*48, dims), vmin=0, vmax=1, edgecolors='k', linewidths=lw)

# axes[2].pcolormesh(true_grid, icgm_grid, np.reshape(joint_prob_swift, dims), edgecolors='k', linewidths=lw)
# axes[3].pcolormesh(true_grid, icgm_grid, np.reshape(joint_prob_swift, dims), edgecolors='k', linewidths=lw)

# a = risk_lbgi_2021*joint_prob_swift
# b = risk_lbgi_swift*joint_prob_swift
# c = np.concatenate((a, b))

# vmin = min(c)
# vmax = max(c)

# axes[4].pcolormesh(true_grid, icgm_grid, np.reshape(a, dims), vmin=vmin, vmax=vmax, edgecolors='k', linewidths=lw)
# axes[5].pcolormesh(true_grid, icgm_grid, np.reshape(b, dims), vmin=vmin, vmax=vmax, edgecolors='k', linewidths=lw)

# axes[0].set_title('2021 Manual Recommendation')
# axes[1].set_title('Swift Temp Basal ')

# ticks = [40, 61, 81, 121, 161, 201, 251, 301, 351]
# ticklabels = ['41-60','61-80','81-120','121-160','161-200','201-250','251-300','301-350','351-400']
# rotation = 20

# for i in range(6):
#     axes[i].invert_yaxis()
#     axes[i].set_ylabel("Sensor Blood  Glucose")
#     axes[i].set_xlabel("True Blood Glucose")
#     # axes[i].set_xticks(ticks)
#     # axes[i].set_xticklabels(ticklabels, rotation=rotation)

#     # axes[i].set_yticks(ticks)
#     # axes[i].set_yticklabels(ticklabels)

# d = joint_prob_swift*(risk_lbgi_swift - risk_lbgi_2021)
# data_grid = np.reshape(d, dims)

# fig2 = plt.figure()
# ax = fig2.add_subplot()

# ax.pcolormesh(true_grid, icgm_grid, data_grid, edgecolors='k', linewidths=lw)

# ax.set_xlabel("True Blood Glucose")
# ax.set_ylabel("Sensor Blood  Glucose")

# ax.invert_yaxis()
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticklabels, rotation=rotation)

# ax.set_yticks(ticks)
# ax.set_yticklabels(ticklabels)

# fig3 = plt.figure()
# ax = fig3.add_subplot()

# ax.pcolormesh(true_grid, icgm_grid, np.reshape(risk_lbgi_swift*joint_prob_swift, dims), vmin=vmin, vmax=vmax, edgecolors='k', linewidths=lw)

# ax.set_xlabel("True Blood Glucose")
# ax.set_ylabel("Sensor Blood  Glucose")

# ax.invert_yaxis()
# ax.set_xticks(ticks)
# ax.set_xticklabels(ticklabels, rotation=rotation)

# ax.set_yticks(ticks)
# ax.set_yticklabels(ticklabels)

# for i in range(5):
#     fig = plt.figure()
#     ax = fig.add_subplot()
#     ax.pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_swift[:,i], dims), vmin=0, vmax=1, edgecolors='k', linewidths=lw)

#     ax.set_xlabel("True Blood Glucose")
#     ax.set_ylabel("Sensor Blood  Glucose")

#     ax.invert_yaxis()
#     ax.set_xticks(ticks)
#     ax.set_xticklabels(ticklabels, rotation=rotation)

#     ax.set_yticks(ticks)
#     ax.set_yticklabels(ticklabels)
plt.show()