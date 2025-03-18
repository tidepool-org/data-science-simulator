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


def get_range_data(result_dir):

    n_trim = 143
    result_path = os.path.join(processed_dir, result_dir)


    result_files = glob.glob(result_path + '*.tsv')
    result_data = load_results(result_path, ext='tsv')

    sim_id_pattern="vp.*.json"
    sim_info_dict = collect_sims_and_results(result_path, sim_id_pattern=sim_id_pattern)

    paf_arr = []
    tir_arr = []
    tbr_arr = []
    tar_arr = []

    insulin_arr = []

    for sim in sim_info_dict:
        sim_data = sim_info_dict[sim]
        
        paf_arr.append(sim_data['controller']['config']['partial_application_factor'])
        
        result_path = sim_data['result_path']
        result_data = load_result(result_path)
        
        bg_data = result_data[1]['bg'][n_trim:]
        true_bolus = result_data[1]['true_bolus'][n_trim:]
        delivered_basal_insulin = result_data[1]['delivered_basal_insulin'][n_trim:]

        tir_arr.append(percent_values_ge_70_le_180(bg_data))
        tbr_arr.append(percent_values_lt_70(bg_data))
        tar_arr.append(percent_values_gt_180(bg_data))

        delivered_insulin = np.nansum(delivered_basal_insulin) + np.nansum(true_bolus)
        insulin_arr.append(delivered_insulin)

    paf_arr = np.array(paf_arr)
    x_paf = np.unique(paf_arr)
    x_tir = []
    x_tbr = []
    x_tar = []
    x_insulin = []

    for paf in x_paf:
        index_list = np.where(paf_arr == paf)[0]

        t = [tir_arr[i] for i in index_list]
        x_tir.append(t)

        t = [tbr_arr[i] for i in index_list]
        x_tbr.append(t)

        t = [tar_arr[i] for i in index_list]
        x_tar.append(t)

        t = [insulin_arr[i] for i in index_list]
        x_insulin.append(t)

    return x_tir, x_tbr, x_tar, x_insulin, x_paf

if __name__ == "__main__":
    processed_dir = os.path.join(DATA_DIR, "processed/")

    result_dir = 'no_meal_announcements_2025_03_17_T_15_09_04/'
    # result_dir_with_ma = 'no_meal_announcements_2025_02_07_WITH_MA_140/'
    # # result_dir_with_ma = 'no_meal_announcements_2025_02_06_WITH_MA/'
    # result_dir_no_ma = 'no_meal_announcements_2025_02_07_NO_MA_140/'
    # # result_dir_no_ma = 'no_meal_announcements_2025_02_06_NO_MA/'

    # # result_dir_with_ma = 'no_meal_announcements_2025_02_07_NO_MA_140_RC_SHORT/'
    x_tir, x_tbr, x_tar, x_insulin, x_paf = get_range_data(result_dir)
    # x_tir_no_ma, x_tbr_no_ma, x_tar_no_ma, paf_arr_no_ma = get_range_data(result_dir_no_ma)


    print(np.mean(x_tir, axis=1))
    print(np.std(x_tir, axis=1))

    positions = np.array(range(len(x_paf)))
    widths = 0.3
    colors = ["#FF9999", "#66B3FF"]  # One for each subgroup

    fig, ax = plt.subplots(2,1)

    box1 = ax[0].boxplot(x_tir, positions=positions, widths=widths, patch_artist=True)
    box2 = ax[1].boxplot(x_insulin, positions=positions, widths=widths, patch_artist=True)

    ax[0].plot(positions, np.mean(x_tir, axis=1), linestyle='None', marker='x', markeredgecolor='Black')
    ax[1].plot(positions, np.mean(x_insulin, axis=1), linestyle='None', marker='x', markeredgecolor='Black')

    # for patch1, patch2 in zip(box1["boxes"], box2["boxes"]):
    #     patch1.set_facecolor(colors[0])
    #     patch2.set_facecolor(colors[1])

    # ax[0].set_xticks(positions)
    ax[0].set_xticklabels(x_paf)

    # ax[0].set_xticks([3.85,4.15])
    # ax[0].set_xticklabels(['MA', 'No MA'])

    # ax[0].set_ylabel('Percent Time in Range\n(70-180 mg/dL)')

    # box1 = ax[1].boxplot(x_tar_with_ma, positions=positions-0.15, widths=widths, patch_artist=True)
    # box2 = ax[1].boxplot(x_tar_no_ma, positions=positions+0.15, widths=widths, patch_artist=True)

    # for patch1, patch2 in zip(box1["boxes"], box2["boxes"]):
    #     patch1.set_facecolor(colors[0])
    #     patch2.set_facecolor(colors[1])

    # ax[1].set_xticks(positions)
    # ax[1].set_xticklabels(np.unique(paf_arr_with_ma))
    # ax[1].set_ylabel('Percent Time Above Range\n(>180 mg/dL)')
    # ax[1].set_xlabel('Partial Application Factor')

    # ax[2].boxplot(x_tbr)
    # ax[2].set_xticklabels(np.unique(paf_arr))
    # ax[2].set_xlabel('Partial Application Factor')
    # ax[2].set_ylabel('Percent Time Below Range (<70 mg/dL)')

    plt.show()