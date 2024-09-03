__author__ = "Cameron Summers"

import os
import re
import json
import glob
import argparse
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import pandas as pd

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.evaluation.jaeb_utils import (
    load_jaeb_issue_report_time_series_data
)

JAEB_DATA_DIR = "../../data/PHI/time-series-data-around-issue-reports-2020-07-28"


def collect_sims_and_results(result_dir, sim_id_pattern="vp.*.json", max_sims=np.inf):
    i=0
    sim_info_dict = dict()
    for root, dirs, files in os.walk(result_dir, topdown=False):
        for file in sorted(files):
            if re.search(sim_id_pattern, file):
                sim_info = json.load(open(os.path.join(root, file), "r"))
                sim_id = sim_info["sim_id"]

                # df_file = [fn for fn in files if sim_id in fn and ".tsv" in fn][0]
                df_file = sim_id + '.tsv'

                df_path = os.path.join(root, df_file)
                sim_info["result_path"] = df_path
                sim_info_dict[sim_id] = sim_info
                
                if i % 1000 == 0:
                    print(i)
                i+=1
                
                if len(sim_info_dict) > max_sims:
                    break

    return sim_info_dict

def collect_sims_and_results_generator(result_dir, sim_id_pattern="vp.*.json", max_sims=np.inf):
    i = 0
    sim_info_dict = dict()
    for root, dirs, files in os.walk(result_dir, topdown=False):
        for file in sorted(files):
            if re.search(sim_id_pattern, file):
                sim_info = json.load(open(os.path.join(root, file), "r"))
                sim_id = sim_info["sim_id"]

                # df_file = [fn for fn in files if sim_id in fn and ".tsv" in fn][0]
                df_file = sim_id + '.tsv'

                df_path = os.path.join(root, df_file)
                sim_info["result_path"] = df_path
                sim_info_dict[sim_id] = sim_info
                
                if len(sim_info_dict) > max_sims:
                    break

                yield sim_id, sim_info


def collect_sims_and_results_parallel(result_dir, sim_id_pattern="vp.*.json", max_sims=np.inf, number_processes=1):
    
    files =glob.glob(result_dir + '/*', recursive=True)
    pool = multiprocessing.Pool(14)
    
    # sim_info_dict = *pool.map()
    return sim_info_dict


def load_results(save_dir, ext="tsv", max_dfs=10):

    all_results = {}
    for root, dirs, files in os.walk(save_dir, topdown=False):
        for i, file in enumerate(sorted(files)):
            if re.search(".*.{}".format(ext), file):

                filepath = os.path.join(root, file)
                sim_id, result_df = load_result(filepath)

                yield sim_id, result_df

                if len(all_results) >= max_dfs:
                    break


def load_result(result_filepath, ext="tsv"):

    if ext == "csv":
        sep = ","
    elif ext == "tsv":
        sep = "\t"

    all_results = {}

    df = pd.read_csv(result_filepath, sep=sep)
    df.set_index("time", inplace=True)
    df.index = pd.to_datetime(df.index)
    path, file = os.path.split(result_filepath)
    all_results[file] = df

    return file, df
    # return all_results


def get_sim_population_results(result_dir, num_vps=10, vp_list=None):

    print("Loading sim population results...")

    population_results = {}
    sim_info_dict = collect_sims_and_results(result_dir)

    all_vps = list(set([sim_info["patient"]["name"] for sim_id, sim_info in sim_info_dict.items()]))
    if vp_list is None:
        vp_list = all_vps[:num_vps]

    for i, vp in enumerate(vp_list):
        vp_sim_info_dict = {sim_id: sim_info for sim_id, sim_info in sim_info_dict.items() if
                            sim_info["patient"]["name"] == vp}

        for sim_id, sim_info in vp_sim_info_dict.items():
            result_path = sim_info["result_path"]

            results_df = pd.read_csv(result_path, sep="\t")

            population_results[sim_id] = results_df

        if vp_list is None and i == num_vps:
            break

    return population_results


def get_sim_population_cgm_data(result_dir, num_vps=10):
    population_data = []
    sim_info_dict = collect_sims_and_results(result_dir)

    vp_list = set([sim_info["patient"]["name"] for sim_id, sim_info in sim_info_dict.items()])

    for vp in vp_list:
        vp_sim_info_dict = {sim_id: sim_info for sim_id, sim_info in sim_info_dict.items() if
                            sim_info["patient"]["name"] == vp}

        for sim_id, sim_info in vp_sim_info_dict.items():
            result_path = sim_info["result_path"]

            results_df = pd.read_csv(result_path, sep="\t")

            population_data.append(results_df)

        if len(population_data) > num_vps:
            break

    population_cgm_data = pd.concat([df["bg_sensor"] for df in population_data], axis=0)
    return population_cgm_data


def compare_bg_delta_distributions(sim_population_data):

    fig, ax = plt.subplots(2, 1)

    print("Stacking bg deltas...")
    sim_population_bg_delta = np.concatenate([get_sim_bg_deltas(result_df) for result_df in sim_population_data.values()], axis=0)

    jaeb_population_data = load_jaeb_issue_report_time_series_data(JAEB_DATA_DIR, num_reports=100)
    jaeb_population_bg_delta = np.concatenate([get_jaeb_cgm_deltas(result_df) for result_df in jaeb_population_data], axis=0)

    print("Plotting...")
    ax[0].hist(sim_population_bg_delta, bins=50)
    ax[0].set_title(
        "Simulation mu={:.1f} sigma={:.1f}".format(np.nanmean(sim_population_bg_delta), np.nanstd(sim_population_bg_delta)))
    ax[0].set_xlim(-20, 20)
    ax[1].hist(jaeb_population_bg_delta, bins=50)
    ax[1].set_title(
        "Jaeb mu={:.1f} sigma={:.1f}".format(np.nanmean(jaeb_population_bg_delta), np.nanstd(jaeb_population_bg_delta)))
    ax[1].set_xlim(-20, 20)
    plt.show()


def compare_bg_distributions(sim_population_data):

    fig, ax = plt.subplots(2, 1)

    sim_population_bg = np.concatenate([result_df["bg_sensor"] for result_df in sim_population_data.values()], axis=0)

    jaeb_population_data = load_jaeb_issue_report_time_series_data(JAEB_DATA_DIR, num_reports=100)
    jaeb_population_cgm_data = np.concatenate([result_df["cgm"] for result_df in jaeb_population_data], axis=0)

    ax[0].hist(sim_population_bg)
    ax[0].set_title("Simulation mu={:.1f} sigma={:.1f}".format(np.nanmean(sim_population_bg), np.nanstd(sim_population_bg)))
    ax[0].set_xlim(0, 500)
    ax[1].hist(jaeb_population_cgm_data)
    ax[1].set_title("Jaeb mu={:.1f} sigma={:.1f}".format(np.nanmean(jaeb_population_cgm_data), np.nanstd(jaeb_population_cgm_data)))
    ax[1].set_xlim(0, 500)
    plt.show()


def get_jaeb_cgm_deltas(result_df):

    bg_delta = result_df["cgm"].diff().iloc[1:]

    result_df["rounded_local_time"] = pd.to_datetime(result_df['rounded_local_time'])
    time_delta_minutes = result_df["rounded_local_time"].diff().iloc[1:]

    bg_delta_per_minute = bg_delta / [dt.seconds / 60 for dt in time_delta_minutes]

    return bg_delta_per_minute


def get_sim_bg_deltas(result_df, sensor=True):

    col = "bg"
    if sensor:
        col = "bg_sensor"

    bg_delta = np.diff(result_df[col])
    bg_delta = bg_delta[~np.isnan(bg_delta)]
    bg_delta /= 5.0  # mg/dL / minute

    return bg_delta


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--result_dir", "-d", default="../../data/results/simulations/physio_results_75vps_2020_09_28_19_58_58")
    parser.add_argument("--num_virtual_patients", "-n", default=30)

    args = parser.parse_args()

    result_dir = args.result_dir
    num_vps = args.num_virtual_patients

    # Select Virtual Patient
    input_str = "Enter VPs Separated by Commas. Press Enter for All.\n"
    sim_info_dict = collect_sims_and_results(result_dir)
    input_id_vp_id_map = {}
    for i, vp_id in enumerate(sorted(set([d["patient"]["name"] for d in sim_info_dict.values()]))):
        input_str += "{}: {}\n".format(i, vp_id)
        input_id_vp_id_map[i] = vp_id
    input_ids_to_inspect = input(input_str) or ",".join(map(str, input_id_vp_id_map.keys()))
    input_ids_to_inspect = list(map(int, input_ids_to_inspect.split(",")))
    vp_id_list = [input_id_vp_id_map[iid] for iid in input_ids_to_inspect]

    sim_population_results = get_sim_population_results(result_dir=result_dir, num_vps=num_vps, vp_list=vp_id_list)

    # Select Simulations
    input_str = "Enter Ids Separated by Commas. Press Enter for All.\n"
    input_id_sim_id_map = {}
    for i, sim_id in enumerate(sorted(sim_population_results.keys())):
        input_str += "{}: {}\n".format(i, sim_id)
        input_id_sim_id_map[i] = sim_id

    input_ids_to_inspect = input(input_str) or ",".join(map(str, input_id_sim_id_map.keys()))

    input_ids_to_inspect = list(map(int, input_ids_to_inspect.split(",")))

    # Select Times
    input_str = "Enter Start Day Index. Press Enter for 0."
    start_day = int(input(input_str) or "0")

    input_str = "Enter Number of Days. Press Enter for All."
    n_days = int(input(input_str) or str("10000000"))

    start_idx = start_day * 288
    end_idx = start_idx + (n_days * 288)

    sim_population_results = {input_id_sim_id_map[input_id]: sim_population_results[input_id_sim_id_map[input_id]].iloc[start_idx: end_idx]
                              for input_id in input_ids_to_inspect}

    plot_sim_results(sim_population_results)
    # compare_bg_delta_distributions(sim_population_results)
    # compare_bg_distributions(sim_population_results)





