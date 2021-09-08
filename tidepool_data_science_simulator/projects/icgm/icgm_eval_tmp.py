__author__ = "Cameron Summers"

import re
import os
import logging
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, norm

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, percent_values_ge_70_le_180
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score

from tidepool_data_science_simulator.models.sensor_icgm import (
    DexcomG6ValueModel, iCGM_THRESHOLDS, iCGMState, iCGMStateV2
)

from tidepool_data_science_simulator.evaluation.icgm_eval import iCGMEvaluator, compute_bionomial_95_LB_CI_moments

from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results, load_result
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results, plot_sim_icgm_paired
from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser


logger = logging.getLogger(__name__)


TOTAL_SOP_TRIALS = 288 * 365 * 100000


def get_positive_bias_errors(all_results):

    tbgs = []
    sbgs = []
    loop_pred_bgs = []
    sensor_error_size = []
    loop_pred_error_size = []
    for sim_id, result_df in all_results.items():
        start_row_mask = result_df.index == datetime.datetime.strptime("8/15/2019 12:00:00", "%m/%d/%Y %H:%M:%S")
        tbg = result_df[start_row_mask]["bg"].values[0]
        sbg = result_df[start_row_mask]["bg_sensor"].values[0]
        loop_pred = result_df[start_row_mask]["loop_final_glucose_pred"].values[0]

        tbgs.append(tbg)
        sbgs.append(sbg)
        loop_pred_bgs.append(loop_pred)
        sensor_error_size.append(sbg - tbg)
        loop_pred_error_size.append(loop_pred - tbg)

    return tbgs, sbgs, loop_pred_bgs, sensor_error_size, loop_pred_error_size


def plot_sensor_error_vs_loop_prediction_error(result_dirs):

    max_dfs = np.inf
    for label, result_dir in result_dirs:

        all_results = load_results(result_dir, ext = "tsv", max_dfs = max_dfs)
        logger.info("Processing {}".format(label))
        tbgs, sbgs, loop_pred_bgs, sensor_error_size, loop_pred_error_size = get_positive_bias_errors(all_results)
        plt.scatter(sensor_error_size, loop_pred_error_size, label=label)

    x_range = np.arange(min(sensor_error_size), max(sensor_error_size))
    plt.plot(x_range, x_range, label="Linear", color="grey", linestyle="--")
    plt.title("Loop Prediction Sensitivity to Spurious Sensor Errors")
    plt.xlabel("Sensor Error Size (mg/dL)")
    plt.ylabel("Loop Prediction Error Size (mg/dL)")
    plt.legend()
    plt.show()


def get_cgm_value_prob_linear(sbg):
    """
    Probability of a given bg value with a linear model
    """

    cgm_range_prob = None
    if sbg < 70:
        num_values = ((70 - 40) / 5)
        cgm_range_prob = 0.065 / num_values
    elif 70 <= sbg <= 180:
        num_values = ((180 - 70) / 5)
        cgm_range_prob = 0.54 / num_values
    elif sbg > 180:
        num_values = ((400 - 180) / 5)
        cgm_range_prob = 0.388 / num_values

    return cgm_range_prob


def get_cgm_range_prob_linear(sbg):
    """
    Probability of a bg range with linear model.
    """
    cgm_range_prob = None
    if sbg < 70:
        cgm_range_prob = 0.065
    elif 70 <= sbg <= 180:
        cgm_range_prob = 0.54
    elif sbg > 180:
        cgm_range_prob = 0.388

    return cgm_range_prob


def get_cgm_value_prob_lognormal(cgm_val, do_fit=False):

    fitted_params = (0.4017623645167624, 2.6904256164008924, 141.4181660742487)

    if do_fit:

        path_to_tp_data = "/Users/csummers/data/Loop Velocity Cap/PHI-2019-07-17-cgm-distributions-6500-6666.csv"
        data = pd.read_csv(path_to_tp_data)
        bg = data["mg/dL"]

        fitted_params = lognorm.fit(bg)
        logger.info(fitted_params)

        plt.hist(bg)
        plt.figure()
        x = np.linspace(40, 400, 400)
        pdf_fitted = lognorm.pdf(x, *fitted_params)

        plt.plot(x, pdf_fitted)
        plt.show()

    x = np.linspace(cgm_val - 2.5, cgm_val + 2.5, 5)
    prob = np.sum(lognorm.pdf(x, *fitted_params))

    return prob


def get_p_error_given_range_normal(sp_letter, error_abs, error_percentage):

    if sp_letter in ["A", "D", "AD_complement"]:
        mu, sigma = 0, 10.3
        # mu, sigma = 0, 14.9
        error = error_abs
    elif sp_letter in ["B", "E", "BE_complement"]:
        mu, sigma = 0, 14.3
        # mu, sigma = 0, 14.9
        error = error_percentage * 100
    elif sp_letter in ["C", "F", "CF_complement"]:
        mu, sigma = 0, 11.7
        # mu, sigma = 0, 12.2
        error = error_percentage * 100

    prob = np.sum(norm.pdf(range(int(round(error)), int(round(error + 5))), mu, sigma)) * 2

    return prob


def plot_special_controls_dist(error_model, bg_model):
    overall_dist = []

    ad_dist = []
    be_dist = []
    cf_dist = []

    num_samples = 100000

    error_models = ["uniform", "normal", "linear", "pareto"]
    if error_model not in error_models:
        raise Exception("{} not recognized error model. Choices {}".format(error_model, error_models))

    bg_models = ["uniform", "lognormal"]
    if bg_model not in bg_models:
        raise Exception("{} not recognized range model. Choices {}".format(bg_model, bg_models))

    HIGH_ERROR_BOUND = 70

    def get_error_size_bounds(size):
        if size == "small":
            low, high = 0, 15
        elif size == "med":
            low, high = 15, 40
        elif size == "large":
            low, high = 40, HIGH_ERROR_BOUND
        return low, high

    sbg_values = range(40, 405, 5)
    if bg_model == "uniform":
        sbg_probs = [1 for _ in sbg_values]
    elif bg_model == "lognormal":
        sbg_probs = [get_cgm_value_prob_lognormal(sbg) for sbg in sbg_values]
    sbg_probs = [v / np.sum(sbg_probs) for v in sbg_probs]

    for i in range(num_samples):

        sbg = np.random.choice(sbg_values, p=sbg_probs)

        error_sizes = ["small", "med", "large"]
        if sbg < 70:
            abs_error_range = np.random.choice(error_sizes, p=[0.85, 0.13, 0.02])
            low_bound, high_bound = get_error_size_bounds(abs_error_range)

            if error_model == "uniform":
                error = np.random.choice([-1, 1]) * np.random.uniform(low_bound, high_bound)
            elif error_model == "normal":
                error = np.random.choice([-1, 1]) * np.random.normal(0, 10.3)
            elif error_model == "linear":
                linear_weights = np.arange(high_bound, low_bound, -1)
                linear_weights = linear_weights / sum(linear_weights)
                error = np.random.choice([-1, 1]) * np.random.choice(range(low_bound, high_bound), p=linear_weights)
            elif error_model == "pareto":
                val = np.random.pareto(0.72)
                while val > HIGH_ERROR_BOUND:
                    val = np.random.pareto(0.72)
                error = np.random.choice([-1, 1]) * val

            true_value = max(sbg - error, 1)
            percent_error = error / true_value

            if true_value > 180:  # Special Control H
                continue

            ad_dist.append(error)
        elif 70 <= sbg <= 180:
            percent_error_range = np.random.choice(error_sizes, p=[0.70, 0.29, 0.01])
            low_bound, high_bound = get_error_size_bounds(percent_error_range)
            if error_model == "uniform":
                percent_error = np.random.choice([-1, 1]) * np.random.uniform(low_bound, high_bound)
            elif error_model == "normal":
                percent_error = np.random.choice([-1, 1]) * np.random.normal(0, 14.3)
            elif error_model == "linear":
                linear_weights = np.arange(high_bound, low_bound, -1)
                linear_weights = linear_weights / sum(linear_weights)
                percent_error = np.random.choice([-1, 1]) * np.random.choice(range(low_bound, high_bound), p=linear_weights)
            elif error_model == "pareto":
                val = np.random.pareto(1.0)
                while val > HIGH_ERROR_BOUND:
                    val = np.random.pareto(1.0)
                percent_error = np.random.choice([-1, 1]) * val

            true_value = sbg / (1 + percent_error/100)
            error = sbg - true_value

            be_dist.append(percent_error)
        elif sbg > 180:
            percent_error_range = np.random.choice(error_sizes, p=[0.80, 0.19, 0.01])
            low_bound, high_bound = get_error_size_bounds(percent_error_range)

            if error_model == "uniform":
                percent_error = np.random.choice([-1, 1]) * np.random.uniform(low_bound, high_bound)
            elif error_model == "normal":
                percent_error = np.random.choice([-1, 1]) * np.random.normal(0, 11.7)
            elif error_model == "linear":
                linear_weights = np.arange(high_bound, low_bound, -1)
                linear_weights = linear_weights / sum(linear_weights)
                percent_error = np.random.choice([-1, 1]) * np.random.choice(range(low_bound, high_bound), p=linear_weights)
            elif error_model == "pareto":
                val = np.random.pareto(1.0)
                while val > HIGH_ERROR_BOUND:
                    val = np.random.pareto(1.0)
                percent_error = np.random.choice([-1, 1]) * val

            true_value = sbg / (1 + percent_error/100)
            error = sbg - true_value

            if true_value < 70:  # Special Control I
                continue

            cf_dist.append(percent_error)
        else:
            raise Exception()

        overall_dist.append(percent_error)

    plot_error_distributions(ad_dist, be_dist, cf_dist, overall_dist, HIGH_ERROR_BOUND, error_model, bg_model)


def score_distributions(ad_dist, be_dist, cf_dist, overall_dist):

    a_score = len([v for v in ad_dist if np.abs(v) < 15]) / len(ad_dist)
    d_score = len([v for v in ad_dist if np.abs(v) < 40]) / len(ad_dist)
    ad_compl_score = len([v for v in ad_dist if np.abs(v) >= 40]) / len(ad_dist)

    b_score = len([v for v in be_dist if np.abs(v) < 15]) / len(be_dist)
    e_score = len([v for v in be_dist if np.abs(v) < 40]) / len(be_dist)
    be_compl_score = len([v for v in be_dist if np.abs(v) >= 40]) / len(be_dist)

    c_score = len([v for v in cf_dist if np.abs(v) < 15]) / len(cf_dist)
    f_score = len([v for v in cf_dist if np.abs(v) < 40]) / len(cf_dist)
    cf_compl_score = len([v for v in cf_dist if np.abs(v) >= 40]) / len(cf_dist)

    g_score = len([v for v in overall_dist if np.abs(v) < 20]) / len(overall_dist)

    return {
        "a_score": a_score,
        "d_score": d_score,
        "ad_complement_score": ad_compl_score,
        "b_score": b_score,
        "e_score": e_score,
        "be_complement_score": be_compl_score,
        "c_score": c_score,
        "f_score": f_score,
        "cf_complement_score": cf_compl_score,
        "g_score": g_score,
    }


def plot_error_distributions(ad_dist, be_dist, cf_dist, overall_dist,
                             n_bins,
                             error_model,
                             bg_model,
                             verbose=False,
                             save_dir=None,
                             description=None):

    scores = score_distributions(ad_dist, be_dist, cf_dist, overall_dist)

    a_score = scores["a_score"]
    d_score = scores["d_score"]
    ad_compl_score = scores["ad_complement_score"]

    b_score = scores["b_score"]
    e_score = scores["e_score"]
    be_compl_score = scores["be_complement_score"]

    c_score = scores["c_score"]
    f_score = scores["f_score"]
    cf_compl_score = scores["cf_complement_score"]

    g_score = scores["g_score"]

    if verbose:
        print("<70 percentage", len(ad_dist) / len(overall_dist))
        print("70-180 percentage", len(be_dist) / len(overall_dist))
        print(">180 percentage", len(cf_dist) / len(overall_dist))

        print("A", a_score)
        print("D", d_score)
        print("AD_complement", ad_compl_score)
        print("AD mu={}. sigma={}".format(np.mean(ad_dist), np.std(ad_dist)))

        print("B", b_score)
        print("E", e_score)
        print("BE_complement", be_compl_score)
        print("BE mu={}. sigma={}".format(np.mean(be_dist), np.std(be_dist)))

        print("C", c_score)
        print("F", f_score)
        print("CF_complement", cf_compl_score)
        print("CF mu={}. sigma={}".format(np.mean(cf_dist), np.std(cf_dist)))

        print("Overall +/- 20%: {:.2f}".format(g_score))

    fig, ax = plt.subplots(1, 4, figsize=(18, 10))
    fig.suptitle("Histograms of {} Errors Sampled under iCGM Special Controls with {} Error Density and P(R) {}".format(len(overall_dist), error_model.capitalize(), bg_model.capitalize()))

    ax[0].hist(ad_dist, density=True, bins=n_bins)
    ax[0].set_title("iCGM<70 Error Distribution, {} Samples\nSp.Ctrls A & D\n<15={:.1f}%. <40={:.1f}%. >40={:.1f}%".format(len(ad_dist), a_score*100, d_score*100, ad_compl_score*100))
    ax[0].set_xlabel("Abs Error")
    ax[0].set_ylabel("Normalized Count")

    ax[1].hist(be_dist, density=True, bins=n_bins)
    ax[1].set_title("70 <= iCGM <= 180 Error Distribution, {} Samples\nSp.Ctrls B & E\n<15%={:.1f}%. <40%={:.1f}%. >40%={:.1f}%".format(len(be_dist), b_score * 100, e_score * 100, be_compl_score * 100))
    ax[1].set_xlabel("Percent Error")

    ax[2].hist(cf_dist, density=True, bins=n_bins)
    ax[2].set_title("iCGM>180 Error Distribution, {} Samples\nSp.Ctrls C & F\n<15%={:.1f}%. <40%={:.1f}%. >40%={:.1f}%".format(len(cf_dist), c_score*100, f_score*100, cf_compl_score*100))
    ax[2].set_xlabel("Percent Error")

    ax[3].hist(overall_dist, density=True, bins=n_bins)
    ax[3].set_xlabel("Percent Error")
    ax[3].set_title("Overall Distribution, Sp. Ctrls G\n<20%={:.1f}%".format(g_score*100))

    plt.legend()

    # if save_dir:
    #     plt.savefig(os.path.join(save_dir, "iCGM_Sample_Error_Distributions_errors={}_bgs={}_{}.png".format(error_model, bg_model, description)))
    # else:
    plt.show()


def get_uniform_dist_from_special_controls(errors, special_controls_boundaries):
    """
    Get the joint uniform probability of given errors and boundary probabilities from special controls.
    """
    num_lt_15 = sum(errors < 15)
    num_gte_15_lt_40 = sum((errors >= 15) & (errors < 40))
    num_gte_40 = sum(errors >= 40)
    uniform_joint_dist = []
    for error in errors:
        if error < 15:
            joint_dist_prob = special_controls_boundaries[0] * 1 / num_lt_15
        elif 15 <= error < 40:
            joint_dist_prob = special_controls_boundaries[1] * 1 / num_gte_15_lt_40
        elif error >= 40:
            joint_dist_prob = special_controls_boundaries[2] * 1 / num_gte_40
        else:
            raise Exception()

        uniform_joint_dist.append(joint_dist_prob)

    total = np.sum(uniform_joint_dist)
    assert abs(total - 1.0) < 0.05
    uniform_joint_dist = [v / total for v in uniform_joint_dist]

    return uniform_joint_dist


def get_p_error_given_cgm_value(sbg_value, tbg_values, error_model="uniform", bg_model="uniform"):
    """
    Compute P(tbg | sbg=sbg_i)
    """

    errors_abs = np.array([abs(tbg - sbg_value) for tbg in tbg_values])
    errors = np.array([tbg - sbg_value for tbg in tbg_values])
    errors_percent_abs = np.array([abs(error / tbg) * 100 for error, tbg in zip(errors_abs, tbg_values)])
    errors_percent = np.array([error / tbg * 100 for error, tbg in zip(errors, tbg_values)])

    if sbg_value < 70:
        if error_model == "uniform":
            tbg_prob_dist = get_uniform_dist_from_special_controls(errors_abs, special_controls_boundaries=[0.85, 0.13, 0.02])
        elif error_model == "normal":
            if bg_model == "uniform":
                tbg_prob_dist = norm.pdf(errors_abs, 0, 9.5)
                tbg_prob_dist = norm.pdf(errors, 0, 10.3)
            elif bg_model == "lognormal":
                tbg_prob_dist = norm.pdf(errors_abs, 0, 9.2)
    elif 70 <= sbg_value <= 180:
        if error_model == "uniform":
            tbg_prob_dist = get_uniform_dist_from_special_controls(errors_percent_abs, special_controls_boundaries=[0.70, 0.29, 0.01])
        elif error_model == "normal":
            if bg_model == "uniform":
                tbg_prob_dist = norm.pdf(errors_percent_abs, 0, 13.4)
            elif bg_model == "lognormal":
                tbg_prob_dist = norm.pdf(errors_percent_abs, 0, 13.4)
    elif sbg_value > 180:
        if error_model == "uniform":
            tbg_prob_dist = get_uniform_dist_from_special_controls(errors_percent_abs, special_controls_boundaries=[0.80, 0.19, 0.01])
        elif error_model == "normal":
            if bg_model == "uniform":
                tbg_prob_dist = norm.pdf(errors_percent_abs, 0, 12.4)
            elif bg_model == "lognormal":
                tbg_prob_dist = norm.pdf(errors_percent_abs, 0, 11.4)
    else:
        raise Exception()

    total = np.sum(tbg_prob_dist)
    tbg_prob_dist = np.array([v / total for v in tbg_prob_dist])

    # Plot the error distribution
    if 0:
    # if 1:
    # if sbg_value == 150:
        ys = [
            # ("Abs Error (mg/dL)", errors_abs),
            # ("Error (mg/dL)", errors),
            ("Abs % Error", errors_percent_abs),
            # ("% Error", errors_percent),
            ("P(tbg | sbg={})", tbg_prob_dist)
        ]
        fig, ax = plt.subplots(len(ys), 1, figsize=(10, 10))

        fig.suptitle("sbg={}".format(sbg_value))
        for i, (descr, y) in enumerate(ys):
            ax[i].scatter(tbg_values, y)
            ax[i].set_ylabel(descr.format(sbg_value))
            ax[i].set_xlabel("True BG")
        plt.show()

        plt.figure()
        plt.scatter(errors_abs, tbg_prob_dist)
        plt.show()

    return tbg_prob_dist


def get_p_error_given_range_linear(sp_letter, source="fda"):

    if source == "fda":
        point_thresholds = iCGM_THRESHOLDS
    elif source == "dexcomG6":
        point_thresholds = G6_THRESHOLDS_DE_NOVO
    else:
        raise Exception

    if sp_letter == "D":
        p_error = iCGM_THRESHOLDS["D"]- iCGM_THRESHOLDS["A"]
    elif sp_letter == "E":
        p_error = iCGM_THRESHOLDS["E"] - iCGM_THRESHOLDS["B"]
    elif sp_letter == "F":
        p_error = iCGM_THRESHOLDS["F"]- iCGM_THRESHOLDS["C"]
    elif sp_letter == "AD_complement":
        p_error = 1 - point_thresholds["D"]
    elif sp_letter == "BE_complement":
        p_error = 1 - point_thresholds["E"]
    elif sp_letter == "CF_complement":
        p_error = 1 - point_thresholds["F"]
    else:
        p_error = point_thresholds[sp_letter]

    return p_error


def remove_H_I_special_controls_sims(sim_summary_df):
    """
    Remove simulations from df that are disallowed by special controls H and I
    """
    logger.info("removing H & I sims...")

    # Remove simulations that are disallowed by special controls H & I
    sp_H_mask = (sim_summary_df["sbg"] < 70) & (sim_summary_df["tbg"] > 180)
    sp_I_mask = (sim_summary_df["sbg"] > 180) & (sim_summary_df["tbg"] < 70)
    sim_summary_df = sim_summary_df[~sp_H_mask]
    sim_summary_df = sim_summary_df[~sp_I_mask]
    return sim_summary_df


def plot_tbg_given_sbg(tbg_given_sbg_dist, error_model):
    """
    Plot entire tbg-sbg space with colored probabilities.
    """
    from operator import itemgetter
    data = []
    for sbg, (tbg, probs) in sorted(tbg_given_sbg_dist.items()):
        for tbg, prob in sorted(zip(tbg, probs), key=itemgetter(0)):
            data.append({
                "sbg": sbg,
                "tbg": tbg,
                "p": prob
            })

    df = pd.DataFrame(data)
    plt.scatter(df["tbg"], df["sbg"], c=df["p"],
                        cmap="Reds")#, vmin=0, vmax=4)
    plt.title("P(tbg | sbg) {} Errors".format(error_model))
    plt.xlabel("True BG (mg/dL)")
    plt.ylabel("Sensor BG (mg/dL)")
    cbar = plt.colorbar()
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel("P(tbg | sbg)", rotation=270)
    plt.show()


def compute_risk_results_sampling(sim_summary_df, save_dir, p_bolus,
                                  error_model="uniform", bg_model="uniform",
                                  description=None):

    random_state = np.random.RandomState(0)
    icgm_state = iCGMState(None, 1, 1, iCGM_THRESHOLDS, iCGM_THRESHOLDS, False, 0, random_state)

    ad_dist = []
    ad_dist_signed = []
    be_dist = []
    be_dist_signed = []
    cf_dist = []
    overall_dist = []

    user_df = sim_summary_df[sim_summary_df["sim_id"].str.contains("07d808c00b707b2dc65962ebff546b7317104516f352b4180d40b4ecdfed8b99")]
    risk_counts_per_sbg_tbg = sim_summary_df[["sbg", "tbg", "risk_score"]].groupby(["sbg", "tbg", "risk_score"]).size().unstack(fill_value=0)
    sbg_values = risk_counts_per_sbg_tbg.index.get_level_values(0).unique()

    # sbg_values = range(70, 185, 5)
    # sbg_values = [40, 65]

    if bg_model == "uniform":
        sbg_probs = [1 for _ in sbg_values]
    elif bg_model == "lognormal":
        sbg_probs = [get_cgm_value_prob_lognormal(sbg) for sbg in sbg_values]
    sbg_probs = [v / np.sum(sbg_probs) for v in sbg_probs]

    tbg_given_sbg_dist = dict()
    for sbg in sbg_values:
        tbg_values = user_df[user_df["sbg"] == sbg]["tbg"].values
        tbg_error_dist = get_p_error_given_cgm_value(sbg, tbg_values, error_model=error_model, bg_model=bg_model)
        tbg_given_sbg_dist[sbg] = (tbg_values, tbg_error_dist)

    # plot_tbg_given_sbg(tbg_given_sbg_dist, error_model)

    num_samples = int(1e5)
    p_severity = defaultdict(float)
    sbg_samples = []

    for i in range(num_samples):

        sbg = np.random.choice(sbg_values, p=sbg_probs)
        tbg = np.random.choice(tbg_given_sbg_dist[sbg][0], p=tbg_given_sbg_dist[sbg][1])

        error_percentage_abs = icgm_state.get_bg_error_pecentage(tbg, sbg) * 100
        error_abs = icgm_state.get_bg_abs_error(tbg, sbg)

        error = (sbg - tbg)
        error_percentage = error / tbg * 100

        if sbg < 70:
            ad_dist.append(error_abs)
            ad_dist_signed.append(error)
        elif 70 <= sbg <= 180:
            be_dist.append(error_percentage_abs)
            be_dist_signed.append(error_percentage)
        elif sbg > 180:
            cf_dist.append(error_percentage_abs)

        overall_dist.append(error_percentage_abs)
        sbg_samples.append(sbg)

        try:
            risk_counts_dict = risk_counts_per_sbg_tbg.loc[sbg, tbg].to_dict()
            for rs, num_sims in risk_counts_dict.items():
                p_severity[rs] = num_sims + p_severity[rs]
        except KeyError:
            continue

    total_counts = sum(p_severity.values())
    p_severity_norm = {k: v / total_counts for k, v in p_severity.items()}
    num_events = {k: int(v * TOTAL_SOP_TRIALS * p_bolus) for k, v in p_severity_norm.items()}
    logger.info("Num Events:", num_events)

    scores = score_distributions(ad_dist, be_dist, cf_dist, overall_dist)

    if 1:
        bins = range(0, 75, 5)
        fig, ax = plt.subplots(2)
        ax[0].hist(be_dist, bins=bins)
        ax[1].hist(be_dist_signed, bins=range(-70, 75, 5))
        plt.xlabel("Errors (mg/dL)")
        plt.ylabel("Count")
        plt.show()

        plt.figure()
        plt.hist(sbg_samples, bins=bins)
        plt.title("Sampled Sensor BG Distribution")
        plt.xlabel("BG (mg/dL)")
        plt.ylabel("Count")
        # plt.savefig(os.path.join(save_dir, "Sensor_BG_Dist_errors={}_bgs={}_{}".format(error_model, bg_model, description)))
        plot_error_distributions(ad_dist, be_dist, cf_dist, overall_dist,
                                 n_bins=bins,
                                 error_model=error_model,
                                 bg_model=bg_model,
                                 save_dir=save_dir,
                                 description=description)

    return num_events, scores


def plot_lognormal_cgm_dist():
    """
    Plot fitted lognormal cgm distribution
    """
    bg_choices = range(40, 405, 5)
    p = [get_cgm_value_prob_lognormal(bg) for bg in bg_choices]
    p = [v / sum(p) for v in p]
    logger.debug(sum(p))
    bg_dist = []
    for i in range(10000):
        sbg = np.random.choice(bg_choices, p=p)
        bg_dist.append(sbg)

    plt.hist(bg_dist, bins=36)
    plt.title("")
    plt.show()


def plot_individual_user_risk(sim_summary_df):
    sim_summary_df["user_id"] = sim_summary_df["sim_id"].apply(lambda x: re.search("vp_(.+)_tbg", x).groups()[0])
    df_user_risk = sim_summary_df[["user_id", "risk_score"]].groupby(["user_id", "risk_score"]).size().reset_index(
        name='counts')
    json_base_configs = transform_icgm_json_to_v2_parser()

    risk_metric = []
    ages = []
    brs = []
    isfs = []
    settings_factor = []
    for c in json_base_configs:
        isf = c["patient"]["patient_model"]["metabolism_settings"]["insulin_sensitivity_factor"]["values"][0]
        br = c["patient"]["patient_model"]["metabolism_settings"]["basal_rate"]["values"][0]

        brs.append(br)
        isfs.append(isf)
        settings_factor.append(isf * br)
        ages.append(c["patient"]["age"])
        cnt_risk5 = \
        df_user_risk[(df_user_risk["user_id"] == c["patient_id"]) & (df_user_risk["risk_score"] == 4)]["counts"].values[
            0]
        risk_metric.append(cnt_risk5)

    factors = [
        ("Basal Rate", brs),
        ("ISF", isfs),
        ("ISF * Basal Rate", settings_factor),
        ("Age", ages)
    ]

    for factor_description, factor_data in factors:
        plt.figure()
        plt.scatter(factor_data, risk_metric)
        plt.title("Risk Factors for Patient Population (Bolus Only)")
        plt.ylabel("Count of Risk=5")
        plt.xlabel(factor_description)

    plt.show()


if __name__ == "__main__":

    # get_p_error_given_range_normal("A", 0, do_fit=True)
    # plot_special_controls_dist(error_model="uniform", bg_model="uniform")
    # plot_lognormal_cgm_dist()

    is_aws_env = False

    test_patient_result_dir = "/Users/csummers/data/simulator/processed/test_patient_jun24"
    test_patietn_no_RC_result_dir = "/Users/csummers/data/simulator/processed/icgm-sensitivity-analysis-results-2021-06-24/"

    # plot_sensor_error_vs_loop_prediction_error([
    #     ("with_RC", test_patient_result_dir),
    #     ("no_RC", test_patietn_no_RC_result_dir)
    # ])

    save_dir = "/Users/csummers/data/simulator/icgm/"
    results_dir = test_patient_result_dir
    if is_aws_env:
        save_dir = "/mnt/cameronsummers/data/simulator/icgm/"
        results_dir = "/mnt/cameronsummers/data/simulator/processed/icgm-sensitivity-analysis-results-2021-06-23/"

    # sim_summary_df = get_icgm_sim_summary_df(results_dir, save_dir=save_dir)

    sim_result_collections = [
        ("Pre-mitigation", "sim_summary_df_Jun6_2021"),
        # ("Post-mitigation", "sim_summary_df_MITIGATED_Aug12_2021")
    ]
    error_models = [
        # "uniform",
        "normal",
    ]
    bg_models = [
        "uniform",
        # "lognormal"
    ]

    p_bolus = 0.02
    data = []
    for description, sim_summary_csv_name in sim_result_collections:
        sim_summary_csv_path = os.path.join(save_dir, sim_summary_csv_name + ".csv")
        # sim_summary_csv_path = os.path.join(save_dir, "sim_summary_df_MITIGATED_Aug12_2021.csv")
        sim_summary_df = pd.read_csv(sim_summary_csv_path)
        sim_summary_df = remove_H_I_special_controls_sims(sim_summary_df)

        plot_individual_user_risk(sim_summary_df)

        for error_model in error_models:
            for bg_model in bg_models:
                logger.info("Running {}. error_model={}, bg_model={}".format(sim_summary_csv_path, error_model, bg_model))
                num_events, scores = compute_risk_results_sampling(sim_summary_df, save_dir,
                                              p_bolus=p_bolus,
                                              error_model=error_model,
                                              bg_model=bg_model,
                                              description=description)

                num_events.update({
                    "description": description,
                    "error_model": error_model,
                    "sbg_model": bg_model,
                })
                num_events.update(scores)
                data.append(num_events)

    df = pd.DataFrame(data)
    df.to_csv(os.path.join(save_dir, "Expected_Events_p_bolus={}.csv".format(p_bolus)))
    logger.info(df)

