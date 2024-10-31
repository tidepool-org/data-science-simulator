__author__ = "Mark Connolly"

import os
import datetime
from tidepool_data_science_simulator.utils import DATA_DIR
import glob
import pandas as pd
from tidepool_data_science_simulator.evaluation.inspect_results import load_results, load_result, collect_sims_and_results
from tidepool_data_science_metrics.glucose.glucose import percent_values_ge_70_le_180, percent_values_lt_70, percent_values_gt_180

import matplotlib.pyplot as plt
import numpy as np

processed_dir = os.path.join(DATA_DIR, "processed/")
result_dir = os.path.join(processed_dir, "autobolus-analysis-results-2024-07-31")

result_files = glob.glob(result_dir + '/patient_id=*.tsv')

result_data = load_results(result_dir, ext='tsv')

sim_info_dict = collect_sims_and_results(result_dir, sim_id_pattern="patient_id.*.json")
paf_arr = []
tir_arr = []
tbr_arr = []
tar_arr = []

for sim in sim_info_dict:
    sim_data = sim_info_dict[sim]
    
    paf_arr.append(sim_data['controller']['config']['partial_application_factor'])
    
    result_path = sim_data['result_path']
    result_data = load_result(result_path)

    tir_arr.append(percent_values_ge_70_le_180(result_data[1]['bg']))
    tbr_arr.append(percent_values_lt_70(result_data[1]['bg']))
    tar_arr.append(percent_values_gt_180(result_data[1]['bg']))


paf_arr = np.array(paf_arr)
fig, ax = plt.subplots(2,1)

x_tir = []
x_tbr = []
x_tar = []
for paf in np.unique(paf_arr):
    index_list = np.where(paf_arr == paf)[0]

    t = [tir_arr[i] for i in index_list]
    x_tir.append(t)

    t = [tbr_arr[i] for i in index_list]
    x_tbr.append(t)

    t = [tar_arr[i] for i in index_list]
    x_tar.append(t)

ax[0].boxplot(x_tir)
ax[0].set_xticklabels([])
ax[0].set_ylabel('Percent Time in Range\n(70-180 mg/dL)')

ax[1].boxplot(x_tar)
ax[1].set_xticklabels(np.unique(paf_arr))
ax[1].set_ylabel('Percent Time Above Range\n(>180 mg/dL)')
ax[1].set_xlabel('Partial Application Factor')

# ax[2].boxplot(x_tbr)
# ax[2].set_xticklabels(np.unique(paf_arr))
# ax[2].set_xlabel('Partial Application Factor')
# ax[2].set_ylabel('Percent Time Below Range (<70 mg/dL)')

plt.show()