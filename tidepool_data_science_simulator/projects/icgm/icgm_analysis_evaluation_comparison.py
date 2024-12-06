import argparse
import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.float_format', '{:.2e}'.format)

path = '/Users/mconn/tidepool/repositories/data-science-simulator/sim_summary_df_MITIGATED_Aug12_2021.csv'
summary_df_2021 = pd.read_csv(path, sep=",")
severity_event_probability_df_2021, (low_icgm_axis_2021, low_true_axis_2021, mean_lbgi_2021, joint_prob_2021) = compute_score_risk_table(summary_df_2021)
print(severity_event_probability_df_2021)


# parser = argparse.ArgumentParser("icgm_analysis_evaluation")
# parser.add_argument("path", help="simulation data directory (process) or summary file path (summarize)", type=str)
# args = parser.parse_args()

# path = args.path
path = '/Users/mconn/tidepool/repositories/data-science-simulator/processed_simulation_data_2024-11-08_AUTOBOLUS.csv'
# path = '/Users/mconn/tidepool/repositories/data-science-simulator/processed_simulation_data_2024-11-08_TEMPBASAL.csv'
path = 'tidepool_data_science_simulator/projects/icgm/processed_data/icgm_sensitivity_analysis_results_AUTOBOLUS_06_2024_12_02_8476f12.csv'

summary_df = pd.read_csv(path, sep="\t")
severity_event_probability_df, (low_icgm_axis, low_true_axis, mean_lbgi_swift, joint_prob_swift) = compute_score_risk_table(summary_df)
print(severity_event_probability_df)

fig = plt.figure()

axes = [fig.add_subplot(3, 2, i, projection='3d') for i in range(1,7)]

axes[0].scatter(low_icgm_axis_2021, low_true_axis_2021, mean_lbgi_2021, c=mean_lbgi_2021, cmap='viridis', marker='o')
axes[1].scatter(low_icgm_axis, low_true_axis, mean_lbgi_swift, c=mean_lbgi_swift, cmap='viridis', marker='o')

axes[2].scatter(low_icgm_axis, low_true_axis, joint_prob_swift, c=joint_prob_swift, cmap='viridis', marker='o')
axes[3].scatter(low_icgm_axis, low_true_axis, joint_prob_swift, c=joint_prob_swift, cmap='viridis', marker='o')

axes[4].scatter(low_icgm_axis, low_true_axis, mean_lbgi_2021*joint_prob_swift, c=mean_lbgi_2021*joint_prob_swift, cmap='viridis', marker='o')
axes[5].scatter(low_icgm_axis, low_true_axis, mean_lbgi_swift*joint_prob_swift, c=mean_lbgi_swift*joint_prob_swift, cmap='viridis', marker='o')

axes[0].set_xlabel("Sensor Blood  Glucose")
axes[0].set_ylabel("True Blood Glucose")
axes[0].set_zlabel("Mean LBGI")
axes[0].set_title('2021')

for i in range(6):
    axes[i].set_xlabel("Sensor Blood  Glucose")
    axes[i].set_ylabel("True Blood Glucose")

fig2 = plt.figure()
ax = fig2.add_subplot(projection='3d')
d = joint_prob_swift*(mean_lbgi_swift - mean_lbgi_2021)
ax.scatter(low_icgm_axis, low_true_axis, d, c=d, cmap='viridis', marker='o')
ax.set_xlabel("Sensor Blood  Glucose")
ax.set_ylabel("True Blood Glucose")
plt.show()