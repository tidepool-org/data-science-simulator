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
path = '/Users/mconn/data/simulator/processed_data/icgm_sensitivity_analysis_results_AUTOBOLUS_06_2024_12_02_8476f12.csv'
# path = '/Users/mconn/data/simulator/processed_data/icgm_sensitivity_analysis_results_MANUAL_2024_11_13_0da14e7.csv'

summary_df = pd.read_csv(path, sep="\t")
severity_event_probability_df, (low_icgm_axis, low_true_axis, mean_lbgi_swift, joint_prob_swift) = compute_score_risk_table(summary_df)
print(severity_event_probability_df)

fig = plt.figure()

axes = [fig.add_subplot(3, 2, i) for i in range(1,7)]

dim = int(np.sqrt(len(low_icgm_axis)))
dims = (dim, dim)

true_grid = np.reshape(low_true_axis_2021, dims)
icgm_grid = np.reshape(low_icgm_axis_2021, dims)

axes[0].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_2021, dims), vmin=0, vmax=1)
axes[1].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_swift, dims), vmin=0, vmax=1)

axes[2].pcolormesh(true_grid, icgm_grid, np.reshape(joint_prob_swift, dims))
axes[3].pcolormesh(true_grid, icgm_grid, np.reshape(joint_prob_swift, dims))

axes[4].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_2021*joint_prob_swift, dims))
axes[5].pcolormesh(true_grid, icgm_grid, np.reshape(mean_lbgi_swift*joint_prob_swift, dims))

axes[0].set_xlabel("Sensor Blood  Glucose")
axes[0].set_ylabel("True Blood Glucose")
axes[0].set_title('2021')

for i in range(6):
    axes[i].invert_yaxis()

d = joint_prob_swift*(mean_lbgi_swift - mean_lbgi_2021)
data_grid = np.reshape(d, dims)

fig2 = plt.figure()
ax = fig2.add_subplot()

ax.pcolormesh(true_grid, icgm_grid, data_grid)

ax.set_xlabel("True Blood Glucose")
ax.set_ylabel("Sensor Blood  Glucose")
ax.invert_yaxis()

plt.show()