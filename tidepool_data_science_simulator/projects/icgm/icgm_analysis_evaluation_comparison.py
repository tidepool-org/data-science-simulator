import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/mconn/Downloads/sim_summary_df_MITIGATED_Aug12_2021.csv'
summary_df_2021 = pd.read_csv(path, sep=",")
severity_event_probability_df_2021, (low_icgm_axis_2021, low_true_axis_2021, mean_lbgi_2021, joint_prob_2021) = compute_score_risk_table(summary_df_2021)


path = '/Users/mconn/tidepool/repositories/data-science-simulator/processed_simulation_data_2024-11-01T15:29:58.673658.csv'
summary_df = pd.read_csv(path, sep="\t")
severity_event_probability_df, (low_icgm_axis, low_true_axis, mean_lbgi_swift, joint_prob_swift) = compute_score_risk_table(summary_df)


fig = plt.figure()
# fig, ax = plt.subplots(1, 2, figsize=(10, 8))

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

# ax2.scatter(low_icgm_axis, low_true_axis, d, c=d, cmap='viridis', marker='o')

# ax2.set_xlabel("Sensor Blood  Glucose")
# ax2.set_ylabel("True Blood Glucose")
# ax2.set_zlabel("Mean LBGI")
# ax2.set_title('Swift Loop')

plt.show()