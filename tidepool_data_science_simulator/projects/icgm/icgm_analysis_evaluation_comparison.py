import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024
import matplotlib.pyplot as plt
import numpy as np
path = '/Users/mconn/Downloads/sim_summary_df_MITIGATED_Aug12_2021.csv'
summary_df_2021 = pd.read_csv(path, sep=",")

mjc_scores_2021 = compute_score_risk_table(summary_df_2021)
# cas_scores = score_risk_table_CS_Aug_2024(summary_df_2021)


path = '/Users/mconn/tidepool/repositories/data-science-simulator/processed_simulation_data_2024-11-01T15:29:58.673658.csv'
summary_df = pd.read_csv(path, sep="\t")
mjc_scores = compute_score_risk_table(summary_df)

# print(pd.DataFrame(cas_scores.table))
# print(mjc_scores_2021)
# print(mjc_scores)

# print(sum(summary_df_2021['lbgi'])/len(summary_df_2021['lbgi']))
# print(sum(summary_df['lbgi_icgm'])/len(summary_df['lbgi_icgm']))

# bins = np.arange(0,600, .1)
# plt.hist(summary_df_2021['lbgi'], bins=bins)
# plt.hist(summary_df['lbgi_icgm'], bins=bins, alpha=0.5)
# plt.show()