__author__ = "Cameron Summers"

import re
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gmean, gstd

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

import seaborn as sns

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, percent_values_ge_70_le_180, percent_values_lt_54
from tidepool_data_science_simulator.evaluation.inspect_results import collect_sims_and_results


def quadratic_loss_estimated(bg_trace, gc):
    """
    Log loss computed using the just the geometric mean, geometric standard deviation.

    Parameters
    ----------
    bg_trace: array
    gc: int

    Returns
    -------
    float: loss
    """
    geo_mean = gmean(bg_trace)
    geo_std = gstd(bg_trace)
    loss = np.power(np.log(geo_std), 2) + np.power(np.log(geo_mean) - np.log(gc), 2)
    return loss


def quadratic_loss(bg_trace, gc):
    """
    Log loss computed as the mean squared error between the log of each bg and the log
    of the set point, gc.

    Parameters
    ----------
    bg_trace: array
    gc: int

    Returns
    -------
    float: loss
    """
    loss = np.mean(np.power(np.log(bg_trace) - np.log(gc), 2))
    return loss


def quadratic_loss_low(bg_trace, gc):
    """
    Log loss computed as the mean squared error between the log of each bg and the log
    of the set point, gc.

    Parameters
    ----------
    bg_trace: array
    gc: int

    Returns
    -------
    float: loss
    """
    lower_bg = bg_trace[bg_trace < gc]
    lower_loss = np.mean(np.power(np.log(lower_bg) - np.log(gc), 2))
    return lower_loss


def quadratic_loss_high(bg_trace, gc):
    """
    Log loss computed as the mean squared error between the log of each bg and the log
    of the set point, gc.

    Parameters
    ----------
    bg_trace: array
    gc: int

    Returns
    -------
    float: loss
    """
    higher_bg = bg_trace[bg_trace > gc]
    higher_loss = np.mean(np.power(np.log(higher_bg) - np.log(gc), 2))
    return higher_loss


def compile_summary_results(result_dir):

    results = []
    sim_info_dict = collect_sims_and_results(result_dir)

    vp_list = sorted(list(set([sim_info["patient"]["name"] for sim_id, sim_info in sim_info_dict.items()])))

    # vp_list = ["VP-10"]
    for vp in vp_list:
        vp_sim_info_dict = {sim_id: sim_info for sim_id, sim_info in sim_info_dict.items() if
                            sim_info["patient"]["name"] == vp}

        for sim_id, sim_info in vp_sim_info_dict.items():
            result_path = sim_info["result_path"]

            results_df = pd.read_csv(result_path, sep="\t")
            results_df.loc[results_df['bg'] > 400.0, "bg"] = 400.0
            results_df.loc[results_df['bg'] < 1.0, "bg"] = 1.0
            lbgi, hbgi, brgi = blood_glucose_risk_index(results_df["bg"])
            tir = percent_values_ge_70_le_180(results_df["bg"])
            pb54 = percent_values_lt_54(results_df["bg"])
            qloss142 = quadratic_loss(results_df["bg"], gc=142)
            qloss142_lower = quadratic_loss_low(results_df["bg"], gc=142)
            qloss142_higher = quadratic_loss_high(results_df["bg"], gc=142)
            qloss142_est = quadratic_loss_estimated(results_df["bg"], gc=142)

            qloss120 = quadratic_loss(results_df["bg"], gc=120)
            qloss120_lower = quadratic_loss_low(results_df["bg"], gc=120)
            qloss120_higher = quadratic_loss_high(results_df["bg"], gc=120)
            qloss120_est = quadratic_loss_estimated(results_df["bg"], gc=120)

            rate = sim_info["controller"]["config"].get("max_physiologic_slope")
            if rate > 1e3:
                rate = 12.0

            print(vp, rate)

            sensor_noise = sim_info["patient"]["sensor"]["standard_deviation"]
            if re.search("Clean", sim_id):
                sensor_noise = 0.0

            row = {
                "vp": vp,
                "sensor_noise": sensor_noise,
                "correct_bolus_bg_threshold": sim_info["patient"]["correct_bolus_bg_threshold"],
                "correct_carb_bg_threshold": sim_info["patient"]["correct_carb_bg_threshold"],
                "carb_count_noise_percentage": sim_info["patient"]["carb_count_noise_percentage"],
                "rate": rate,
                "lbgi": lbgi,
                "hbgi": hbgi,
                "brgi": brgi,
                "tir": tir,
                "qloss142": qloss142,
                "qloss142_lower": qloss142_lower,
                "qloss142_higher": qloss142_higher,
                "qloss120": qloss120,
                "qloss120_lower": qloss120_lower,
                "qloss120_higher": qloss120_higher,
                "qloss142_est": qloss142_est,
                "qloss120_est": qloss120_est,
                "percent_below_54": pb54
            }
            results.append(row)

    results_df = pd.DataFrame(results)
    return results_df


def plot_vp_rate_diff(summary_df, result_dir):

    vp_groups = summary_df.sort_values("rate", ascending=False).groupby("vp")

    metrics = [
        "percent_below_54",
        "lbgi",
        "hbgi",
        "brgi",
        "tir",
        "qloss120",
        "qloss120_lower",
        "qloss120_higher",
        "qloss142",
        "qloss142_lower",
        "qloss142_higher"
    ]

    all_vp_dfs = []
    for name, group in vp_groups:
        group_noisy_baseline_mask = group["sensor_noise"] != 0.0

        vp_df = pd.DataFrame()
        X2_rates = group[group_noisy_baseline_mask]["rate"]
        vp_df["Rate Caps"] = X2_rates
        vp_df["VP"] = name

        for i, metric in enumerate(metrics):

            clean_baseline = group[(group["rate"] == 12.0) & (group["sensor_noise"] == 0.0)][metric].values[0]
            noisy_baseline = group[(group["rate"] == 12.0) & (group["sensor_noise"] != 0.0)][metric].values[0]
            metric_delta_introduced = noisy_baseline - clean_baseline
            metric_delta_introduced_ratio = noisy_baseline / clean_baseline * 100

            print(name, metric, "Clean -> Noisy: {:.2f} -> {:.2f}. Diff {:.2f}".format(clean_baseline, noisy_baseline, noisy_baseline - clean_baseline))

            metric_delta = group[group_noisy_baseline_mask][metric] - noisy_baseline

            metric_delta_ratio_percent_norm = (group[group_noisy_baseline_mask][metric] - clean_baseline) / (noisy_baseline - clean_baseline) * 100.0
            metric_delta_ratio_percent = group[group_noisy_baseline_mask][metric] / noisy_baseline * 100.0

            vp_df["{}_delta_ratio_norm".format(metric)] = list(metric_delta_ratio_percent_norm)
            vp_df["{}_delta_ratio".format(metric)] = list(metric_delta_ratio_percent)
            vp_df["{}_delta".format(metric)] = list(metric_delta)
            vp_df["{}_introduced".format(metric)] = metric_delta_introduced
            vp_df["{}_clean".format(metric)] = clean_baseline
            vp_df["{}_introduced_ratio".format(metric)] = metric_delta_introduced_ratio

            vp_df[metric] = group[group_noisy_baseline_mask][metric]

            if metric in ["percent_below_54"] and (metric_delta_ratio_percent > 150).any() and (noisy_baseline / clean_baseline) > 1.1:
                print(name, metric, vp_df)

        all_vp_dfs.append(vp_df)

    all_vp_dfs = pd.concat(all_vp_dfs, axis=0)
    all_vp_dfs.loc[all_vp_dfs["Rate Caps"] == 12.0, "Rate Caps"] = None

    for metric in metrics:
        plt.figure()
        g = sns.boxplot(x="Rate Caps", y="{}_delta".format(metric), data=all_vp_dfs)
        plt.savefig(os.path.join(result_dir, "figures", "Rate vs {}_delta.png".format(metric)))
        plt.figure()
        g = sns.boxplot(x="Rate Caps", y="{}_delta_ratio".format(metric), data=all_vp_dfs)
        plt.savefig(os.path.join(result_dir, "figures", "Rate vs {}_delta_ratio.png".format(metric)))
        plt.figure()
        g = sns.boxplot(y="{}_introduced".format(metric), data=all_vp_dfs)
        plt.savefig(os.path.join(result_dir, "figures", "{}_introduced.png".format(metric)))
        plt.figure()
        g = sns.boxplot(y="{}_clean".format(metric), data=all_vp_dfs)
        plt.savefig(os.path.join(result_dir, "figures", "{}_clean.png".format(metric)))

    # plt.legend()
    # plt.show()


if __name__ == "__main__":

    sim_result_dir = "../data/results/simulations/physio_results_75vps_2020_09_28_19_58_58/"

    results_df = compile_summary_results(sim_result_dir)
    plot_vp_rate_diff(results_df, sim_result_dir)







