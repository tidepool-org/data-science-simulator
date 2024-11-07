import numpy as np
from tidepool_data_science_simulator.evaluation.inspect_results import load_result
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from pathlib import Path
from tidepool_data_science_metrics.glucose import glucose

result_directory = Path("/Users/mconn/data/simulator/processed/icgm-sensitivity-analysis-results-2020-12-11/")
result_directory = Path("/Users/mconn/data/simulator/processed/icgm_sensitivity_analysis_results_COASTAL_AUTOBOLUS_2024_10_30_T_19_10_10/")

result_pattern = "*bg40.siCGM_40.0*.tsv"
result_pattern = "*tbg=40_sbg=40.tsv"

for result_path in result_directory.glob(result_pattern): 
    (sim_id, sim_results_df) = load_result(result_path, ext="tsv")

    bg = np.array(sim_results_df.bg)  
    # true_bolus = np.array(sim_results_df['true_bolus'])
    # true_basal = np.array(sim_results_df['temp_basal'])
    # first_valid_bolus = np.argmax(~np.isnan(true_bolus))
    # first_valid_basal = np.argmax(~np.isnan(true_basal))
    # first_valid_index = min((first_valid_basal, first_valid_bolus))
    bg = np.ones((1,100))*95
    lbgi_icgm, hbgi_icgm, brgi_icgm = glucose.blood_glucose_risk_index(bg)
    print(lbgi_icgm)
    
    # bg = bg[first_valid_index:]
    # lbgi, hgbi, bgi = glucose.blood_glucose_risk_index(bg)
    # print(lbgi)
    # print()
    plot_sim_results({sim_id: sim_results_df})


