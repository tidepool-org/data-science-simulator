import argparse
import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.float_format', '{:.2e}'.format)

path = '/Users/mconn/data/simulator/processed_data/sim_summary_df_MITIGATED_Aug12_2021.csv'
summary_df_2021 = pd.read_csv(path, sep=",")
severity_event_probability_df_2021, (low_icgm_axis_2021, low_true_axis_2021, mean_lbgi_2021, joint_prob_2021) = compute_score_risk_table(summary_df_2021)
print(severity_event_probability_df_2021)


# parser = argparse.ArgumentParser("icgm_analysis_evaluation")
# parser.add_argument("path", help="simulation data directory (process) or summary file path (summarize)", type=str)
# args = parser.parse_args()

path = '/Users/mconn/data/simulator/processed_data/icgm_sensitivity_analysis_results_AUTOBOLUS_04_2024_11_20_619381e.csv'
# path = '/Users/mconn/data/simulator/processed_data/icgm_sensitivity_analysis_results_AUTOBOLUS_06_2024_12_02_8476f12.csv'
# path = '/Users/mconn/data/simulator/processed_data/icgm_sensitivity_analysis_results_MANUAL_2024_11_13_0da14e7.csv'

summary_df = pd.read_csv(path, sep="\t")
severity_event_probability_df, (low_icgm_axis, low_true_axis, mean_lbgi_swift, joint_prob_swift) = compute_score_risk_table(summary_df)
print(severity_event_probability_df)

lw =2 
fig = plt.figure()

axes = [fig.add_subplot(3, 2, i) for i in range(1,7)]

dim = int(np.sqrt(len(low_icgm_axis)))
dims = (dim, dim)

true_grid = np.reshape(low_true_axis_2021, dims)
icgm_grid = np.reshape(low_icgm_axis_2021, dims)

axes[0].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_2021, dims), vmin=0, vmax=1, edgecolors='k', linewidths=lw)
axes[1].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_swift, dims), vmin=0, vmax=1, edgecolors='k', linewidths=lw)

axes[2].pcolormesh(true_grid, icgm_grid, np.reshape(joint_prob_swift, dims), edgecolors='k', linewidths=lw)
axes[3].pcolormesh(true_grid, icgm_grid, np.reshape(joint_prob_swift, dims), edgecolors='k', linewidths=lw)
 
a = np.concatenate((mean_lbgi_2021*joint_prob_swift, mean_lbgi_swift*joint_prob_swift))
vmin = min(a)
vmax = max(a)

axes[4].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_2021*joint_prob_swift, dims), vmin=vmin, vmax=vmax, edgecolors='k', linewidths=lw)
axes[5].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_swift*joint_prob_swift, dims), vmin=vmin, vmax=vmax, edgecolors='k', linewidths=lw)

# axes[4].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_2021*joint_prob_swift, dims),  edgecolors='k', linewidths=lw)
# axes[5].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_swift*joint_prob_swift, dims), edgecolors='k', linewidths=lw)

axes[0].set_title('2021')
axes[4].set_xlabel("Sensor Blood  Glucose")
axes[4].set_ylabel("True Blood Glucose")

ticks = [40, 61, 81, 121, 161, 201, 251, 301, 351]
ticklabels = ['41-60','61-80','81-120','121-160','161-200','201-250','251-300','301-350','351-400']
rotation = 20

for i in range(6):
    axes[i].invert_yaxis()

    axes[i].set_xticks(ticks)
    axes[i].set_xticklabels(ticklabels, rotation=rotation)

    axes[i].set_yticks(ticks)
    axes[i].set_yticklabels(ticklabels)

d = joint_prob_swift*(mean_lbgi_swift - mean_lbgi_2021)
data_grid = np.reshape(d, dims)

fig2 = plt.figure()
ax = fig2.add_subplot()

ax.pcolormesh(true_grid, icgm_grid, data_grid, edgecolors='k', linewidths=lw)

ax.set_xlabel("True Blood Glucose")
ax.set_ylabel("Sensor Blood  Glucose")

ax.invert_yaxis()
ax.set_xticks(ticks)
ax.set_xticklabels(ticklabels, rotation=rotation)

ax.set_yticks(ticks)
ax.set_yticklabels(ticklabels)
plt.show()