import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024
import matplotlib.pyplot as plt
import numpy as np

path = '/Users/mconn/Downloads/sim_summary_df_MITIGATED_Aug12_2021.csv'
summary_df_2021 = pd.read_csv(path, sep=",")
severity_event_probability_df_2021, (low_icgm_axis_2021, low_true_axis_2021, d_2021) = compute_score_risk_table(summary_df_2021)


path = '/Users/mconn/tidepool/repositories/data-science-simulator/processed_simulation_data_2024-11-01T15:29:58.673658.csv'
summary_df = pd.read_csv(path, sep="\t")
severity_event_probability_df, (low_icgm_axis, low_true_axis, d) = compute_score_risk_table(summary_df)


fig = plt.figure()
# fig, ax = plt.subplots(1, 2, figsize=(10, 8))

ax1 = fig.add_subplot(1, 2, 1, projection='3d')
ax2 = fig.add_subplot(1, 2, 2, projection='3d')

ax1.scatter(low_icgm_axis_2021, low_true_axis_2021, d_2021, c=d, cmap='viridis', marker='o')

ax2.scatter(low_icgm_axis, low_true_axis, d, c=d, cmap='viridis', marker='o')

ax1.set_xlabel("Sensor Blood  Glucose")
ax1.set_ylabel("True Blood Glucose")
ax1.set_zlabel("Mean LBGI")
ax1.set_title('2021')

ax2.scatter(low_icgm_axis, low_true_axis, d, c=d, cmap='viridis', marker='o')

ax2.set_xlabel("Sensor Blood  Glucose")
ax2.set_ylabel("True Blood Glucose")
ax2.set_zlabel("Mean LBGI")
ax2.set_title('Swift Loop')

plt.show()