import pandas as pd
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.projects.icgm.icgm_eval_tmp import score_risk_table_CS_Aug_2024


path = '/Users/mconn/Downloads/sim_summary_df_MITIGATED_Aug12_2021.csv'
summary_df = pd.read_csv(path, sep=",")
mjc_scores_2021 = compute_score_risk_table(summary_df)

cas_scores = score_risk_table_CS_Aug_2024(summary_df)

path = '/Users/mconn/tidepool/repositories/data-science-simulator/processed_simulation_data_2024-09-04T20:51:28.795869.csv'
summary_df = pd.read_csv(path, sep="\t")
mjc_scores = compute_score_risk_table(summary_df)


print(mjc_scores_2021)
print(pd.DataFrame(cas_scores.table))

print(mjc_scores)
