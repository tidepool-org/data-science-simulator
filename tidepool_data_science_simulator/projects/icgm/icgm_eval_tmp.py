__author__ = "Cameron Summers"

import re
import os
import logging
import datetime
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, percent_values_ge_70_le_180
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score

from tidepool_data_science_simulator.models.sensor_icgm import (
    DexcomG6ValueModel, iCGM_THRESHOLDS, iCGMState, iCGMStateV2
)
from tidepool_data_science_simulator.projects.icgm.icgm_sensitivity_analysis_ai_letter_June2021 import get_initial_offset_sensor

from tidepool_data_science_simulator.evaluation.icgm_eval import iCGMEvaluator, compute_bionomial_95_LB_CI_moments

from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results, load_result
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results, plot_sim_icgm_paired


logger = logging.getLogger(__name__)


def compute_sim_summary_stats(result_dir):

    sim_results = collect_sims_and_results(result_dir, sim_id_pattern="vp.*bg.*.json", max_sims=1e12)

    summary_data = []
    for sim_id, sim_json_info in sim_results.items():
        if "Ideal" in sim_id:
            continue

        for sim_id_match, sim_json_info_match in sim_results.items():
            non_sensor_id = re.sub(r"\.s.*\.", "", sim_id)
            non_sensr_id_match = re.sub(r"\.sIdealSensor\.", "", sim_id_match)
            if non_sensor_id == non_sensr_id_match:
                sim_json_info_match = sim_json_info_match
                break

        filename_icgm, df_results = load_result(sim_json_info["result_path"])
        # filename_icgm, df_results = list(df_results_dict.items())[0]
        true_bg = df_results['bg']
        true_bg[true_bg < 1] = 1
        lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(true_bg)
        dkai_icgm = dka_index(df_results['iob'], df_results["sbr"])

        filename_ideal, df_results_ideal = load_result(sim_json_info_match["result_path"])
        # filename_ideal, df_results_ideal = list(df_results_ideal_dict.items())[0]
        true_bg = df_results_ideal["bg"]
        true_bg[true_bg < 1] = 1
        lbgi_ideal, hbgi_ideal, brgi_ideal = blood_glucose_risk_index(true_bg)
        dkai_ideal = dka_index(df_results_ideal['iob'], df_results_ideal["sbr"])

        bg_cond = int(re.search(r"bg=(\d)", sim_id).groups()[0])

        true_bg_start = sim_json_info["patient"]["sensor"]["true_start_bg"]
        sensor_bg_start = sim_json_info["patient"]["sensor"]["start_bg_with_offset"]
        
        target_bg = 110
        isf = df_results["isf"].values[0]
        max_bolus_delivered = df_results["true_bolus"].max()
        traditional_bolus_delivered = max(0, (sensor_bg_start - target_bg) / isf)

        row = {
            "sim_id": sim_id,
            "lbgi_icgm": lbgi_icgm,
            "lbgi_ideal": lbgi_ideal,
            "lbgi_diff": lbgi_icgm - lbgi_ideal,
            "dkai_icgm": dkai_icgm,
            "dkai_ideal": dkai_ideal,
            "dkai_diff": dkai_icgm - dkai_ideal,
            "bg_condition": bg_cond,
            "true_start_bg": true_bg_start,
            "start_bg_with_offset": sensor_bg_start,
            "sbr": df_results["sbr"].values[0],
            "isf": isf,
            "cir": df_results["cir"].values[0],
            "ylw": sim_json_info["controller"]["config"]["ylw"],
            "age": sim_json_info["controller"]["config"]["age"],
            "max_bolus_delivered": max_bolus_delivered,
            "traditional_bolus_delivered": traditional_bolus_delivered,
            "bolus_diff": max_bolus_delivered - traditional_bolus_delivered
        }

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    summary_result_filepath = "./result_summary_{}.csv".format(datetime.datetime.now().isoformat())
    summary_df.to_csv(summary_result_filepath, sep="\t")
    logger.info("Saved summary results to", summary_result_filepath)

    return summary_result_filepath


def compute_risk_stats(summary_df):

    # compute_dka_risk_tp_icgm(summary_df)
    # compute_lbgi_risk_tp_icgm_negative_bias(summary_df)

    score_risk_table(summary_df)


def compute_dka_risk_tp_icgm(summary_df):

    initially_ok_mask = summary_df["dkai_ideal"] == 0.0
    logger.info(summary_df[initially_ok_mask]["dkai_icgm"].describe())


def compute_lbgi_risk_tp_icgm_negative_bias(summary_df):

    initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
    logger.info(summary_df[initially_ok_mask]["lbgi_icgm"].describe())
    logger.info("2.5 LBGI percentile",
          summary_df[initially_ok_mask]["lbgi_icgm"].values.searchsorted(2.5)/len(summary_df[initially_ok_mask])*100)


def sim_id_without_sensor(sim_id):
    if sim_id is None:
        return

    return re.sub(r"\.s.*\.", "", sim_id)


def plot_icgm_results(result_dir, sim_inspect_id=None):

    all_results = load_results(result_dir, ext="tsv", max_dfs=np.inf)

    sim_groups_to_plot = defaultdict(dict)
    ideal_sims = defaultdict(dict)
    for sim_id, result_df in all_results.items():
        sensor_group_id = sim_id_without_sensor(sim_id)
        if "Ideal" in sim_id:
            ideal_sims[sensor_group_id][sim_id] = result_df
        else:
            sim_groups_to_plot[sensor_group_id][sim_id] = result_df

    sim_inspect_group_id = sim_id_without_sensor(sim_inspect_id)
    for sim_group_id, sim_group_results in sim_groups_to_plot.items():

        if sim_inspect_id is not None and sim_inspect_group_id != sim_group_id:
            continue

        plot_sim_results(sim_group_results)

        # sim_group_results = {sim_id: results_df for i, (sim_id, results_df) in enumerate(sim_group_results.items()) if i == 10}
        # sim_group_results.update({sim_id: result_df for sim_id, result_df in ideal_sims[sim_group_id].items()})
        # plot_sim_icgm_paired(sim_group_results)


class TPRiskTableRev7(object):

    def __init__(self):
        self.table = np.zeros(shape=(5, 5))

        self.table_severity_indices = {
            (0.0, 2.5): 0,
            (2.5, 5.0): 1,
            (5.0, 10.0): 2,
            (10.0, 20.0): 3,
            (20.0, np.inf): 4
        }

        self.table_probability_indices = {
            (14600000, np.inf): 0,
            (1460000, 14599999): 1,
            (14600, 1459999): 2,
            (146, 14599): 3,
            (0, 146): 4,
        }

        self.acceptability_regions = {
            (0, 0): "Yellow",
            (0, 1): "Red",
            (0, 2): "Red",
            (0, 3): "Red",
            (0, 4): "Red",
            (1, 0): "Green",
            (1, 1): "Yellow",
            (1, 2): "Red",
            (1, 3): "Red",
            (1, 4): "Red",
            (2, 0): "Green",
            (2, 1): "Yellow",
            (2, 2): "Yellow",
            (2, 3): "Red",
            (2, 4): "Red",
            (3, 0): "Green",
            (3, 1): "Green",
            (3, 2): "Yellow",
            (3, 3): "Red",
            (3, 4): "Red",
            (4, 0): "Green",
            (4, 1): "Green",
            (4, 2): "Green",
            (4, 3): "Yellow",
            (4, 4): "Yellow",
        }

    def get_probability_index(self, num_events_per_100k_person_years):

        for bounds in self.table_probability_indices.keys():
            if bounds[0] <= num_events_per_100k_person_years < bounds[1]:
                return self.table_probability_indices[bounds]

        raise Exception("Probability not in indices.")

    def get_severity_index(self, severity_lbgi):

        for bounds in self.table_severity_indices.keys():
            if bounds[0] <= severity_lbgi < bounds[1]:
                return self.table_severity_indices[bounds]

        raise Exception("Severity not in indices.")

    def add(self, severity_lbgi, num_events_per_100k_person_years):

        severity_idx = self.get_severity_index(severity_lbgi)
        prob_idx = self.get_probability_index(num_events_per_100k_person_years)

        self.table[prob_idx, severity_idx] += 1

    def is_problematic(self, severity_lbgi, num_events_per_100k_person_years):

        severity_idx = self.get_severity_index(severity_lbgi)
        prob_idx = self.get_probability_index(num_events_per_100k_person_years)
        problematic = False
        if self.acceptability_regions[(prob_idx, severity_idx)] == "Red":
            problematic = True

        return problematic

    def print(self):
        print(pd.DataFrame(self.table))


class PositiveBiasiCGMRequirements():

    def __init__(self, risk_table=TPRiskTableRev7(), true_ranges=None):

        self.p_corr_bolus_given_error = 3.0 / 288.0
        self.num_cgm_per_100k_person_years = 288 * 365 * 100000

        self.true_ranges = true_ranges
        if true_ranges is None:
            self.true_ranges = [
                (0, 40),
                (40, 60),
                (61, 80),
                (81, 120),
                (121, 160),
                (161, 200),
                (201, 250),
                (251, 300),
                (301, 350),
                (351, 400),
            ]

        self.dexcom_pediatric_value_model = DexcomG6ValueModel(concurrency_table="pediatric")
        total_data_points = np.sum(self.dexcom_pediatric_value_model.comparator_totals)
        self.p_true_pediatric = np.array([v / total_data_points for v in self.dexcom_pediatric_value_model.comparator_totals])

        self.risk_table = risk_table

    def fit_positive_bias_prob(self, summary_df):

        for i, (low_true, high_true) in enumerate(self.true_ranges):

            for (low_icgm, high_icgm) in self.true_ranges[i:]:

                initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
                true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)

                icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (
                            summary_df["start_bg_with_offset"] <= high_icgm)

                concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask
                sub_df = summary_df[concurrency_square_mask]

                p_error_max = self.fit_error_probability(sub_df)
                p_requirements = self.dexcom_pediatric_value_model.get_joint_probability(low_true, low_icgm)

                logger.info(low_true, high_true, low_icgm, high_icgm, p_error_max, p_requirements)

    def fit_positive_bias_range_and_prob(self, summary_df):

        initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
        # initially_ok_mask = summary_df["lbgi_ideal"] < 0.5

        for i, (low_true, high_true) in enumerate(self.true_ranges):

            true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)

            for (low_icgm, high_icgm) in self.true_ranges[i:]:

                test_high_icgms = [high_icgm - i for i in range(0, 40, 1)]
                mitigation_probs = []

                for test_high_icgm in test_high_icgms:

                    icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (
                            summary_df["start_bg_with_offset"] <= test_high_icgm)

                    concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask
                    sub_df = summary_df[concurrency_square_mask]

                    # plt.hist(sub_df["true_start_bg"], alpha=0.5, label="True")
                    # plt.hist(sub_df["start_bg_with_offset"], alpha=0.5, label="iCGM")
                    # plt.legend()
                    # plt.show()

                    p_error_max = self.fit_error_probability(sub_df)

                    mitigation_probs.append(p_error_max)
                    p_requirements = self.dexcom_pediatric_value_model.get_joint_probability(low_true, low_icgm)

                    logger.info(low_true, high_true, low_icgm, test_high_icgm, p_error_max, p_requirements)
                    logger.info("Num sims", np.sum(concurrency_square_mask))

                    if p_error_max is not None and p_error_max < p_requirements and (test_high_icgm - low_icgm) < 5:
                        a = 1

                plt.plot(test_high_icgms, mitigation_probs, label="Max P(True, iCGM)")
                plt.axhline(p_requirements, label="Dexcom P(True, iCGM)", linestyle="--")
                plt.legend()
                plt.show()

    def fit_error_probability(self, df, max_iters=20):

        high_bound = 1.0
        low_bound = 0.0

        lbgi_data = df["lbgi_icgm"]

        if len(lbgi_data) == 0:
            return None

        # Check if ok initially
        if self.is_mitigated(lbgi_data, high_bound):
            return high_bound

        num_iters = 0
        num_iters_not_mitigated = 0
        test_bounds = []
        while True:

            test_bound = (high_bound - low_bound) / 2.0

            if num_iters >= max_iters:
                # plt.plot(test_bounds)
                # plt.show()
                # print(num_iters_not_mitigated)
                return test_bound

            test_bounds.append(test_bound)

            if not self.is_mitigated(lbgi_data, test_bound):
                high_bound = test_bound
            else:
                low_bound = test_bound
                num_iters_not_mitigated += 1

            num_iters += 1

    def is_mitigated(self, lbgi_data, region_probability):

        num_total_sims = len(lbgi_data)

        for s_idx, severity_band in enumerate([(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)], 1):

            severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])
            num_sims_in_severity_band = len(lbgi_data[severity_mask])
            severity_prob = num_sims_in_severity_band / num_total_sims
            risk_prob_sim = severity_prob * self.p_corr_bolus_given_error * region_probability
            num_risk_events_sim = risk_prob_sim * self.num_cgm_per_100k_person_years

            if self.risk_table.is_problematic(severity_band[0], num_risk_events_sim):
                return False

        return True


def score_risk_table(summary_df):

    dexcom_value_model = DexcomG6ValueModel(concurrency_table="TP_iCGM")

    summary_df["vp_id"] = summary_df["sim_id"].apply(lambda sim_id: re.search(r"(vp.*).bg=\d", sim_id).groups()[0])

    risk_table_per_error_bin_patient_prob = TPRiskTableRev7()
    risk_table_per_error_bin_sim_prob = TPRiskTableRev7()
    risk_table_per_sim = TPRiskTableRev7()

    patient_percentages = []
    lbgi_band = []

    total_sims = 0
    for (low_true, high_true), (low_icgm, high_icgm) in [
        ((40, 60), (40, 60)),
        ((40, 60), (61, 80)),
        ((40, 60), (81, 120)),

        ((61, 80), (61, 80)),
        ((61, 80), (81, 120)),

        ((81, 120), (81, 120)),
        ((81, 120), (121, 160)),
        ((81, 120), (161, 200)),

        ((121, 160), (121, 160)),
        ((121, 160), (161, 200)),
        ((121, 160), (201, 250)),

        ((161, 200), (161, 200)),
        ((161, 200), (201, 250)),
        ((161, 200), (251, 300)),

        ((201, 250), (201, 250)),
        ((201, 250), (251, 300)),
        ((201, 250), (301, 350)),

        ((251, 300), (251, 300)),
        ((251, 300), (301, 350)),
        ((251, 300), (351, 400)),

        ((301, 350), (301, 350)),
        ((301, 350), (351, 400)),
        ((351, 400), (351, 450)),
    ]:
        true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)
        icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (summary_df["start_bg_with_offset"] <= high_icgm)
        initially_ok_mask = summary_df["lbgi_ideal"] == 0.0

        concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask

        p_error = dexcom_value_model.get_joint_probability(low_true, low_icgm)
        p_corr_bolus_given_error = 3 / 288
        num_cgm_per_100k_person_years = 288 * 365 * 100000

        num_initially_ok = np.sum(initially_ok_mask)
        num_range_mask = np.sum(true_mask & icgm_mask)
        num_total_sims = max(1, len(summary_df[concurrency_square_mask]))

        lbgi_data = summary_df[concurrency_square_mask]["lbgi_icgm"]

        num_total_patients = max(1, len(summary_df[concurrency_square_mask]["vp_id"].unique()))
        for s_idx, severity_band in enumerate([(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)], 1):
            severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])

            num_patients_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask]["vp_id"].unique())
            num_sims_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask])

            patient_prob = num_patients_in_severity_band / num_total_patients
            sim_prob = num_sims_in_severity_band / num_total_sims

            risk_prob_patient = patient_prob * p_corr_bolus_given_error * p_error
            risk_prob_sim = sim_prob * p_corr_bolus_given_error * p_error

            num_risk_events_patient = risk_prob_patient * num_cgm_per_100k_person_years
            num_risk_events_sim = risk_prob_sim * num_cgm_per_100k_person_years

            patient_percentages.append(patient_prob)
            lbgi_band.append(s_idx)

            if not np.isnan(num_risk_events_patient) and num_risk_events_patient > 0.0:
                risk_table_per_error_bin_patient_prob.add(severity_band[0], num_risk_events_patient)
                # print(num_risk_events_patient)

            if not np.isnan(num_risk_events_sim) and num_risk_events_sim > 0.0:
                risk_table_per_error_bin_sim_prob.add(severity_band[0], num_risk_events_sim)
                # print(num_risk_events_sim)

        # p_settings = 1/99.0
        # for i, row in summary_df[concurrency_square_mask].iterrows():
        #
        #     sim_severity = row["lbgi_icgm"]
        #
        #     # bg_error = max(10, row["start_bg_with_offset"] - row["true_start_bg"])
        #     # p_corr_bolus_given_error /= (bg_error / 10)
        #     # print(p_corr_bolus_given_error)
        #
        #     sim_prob = p_error * p_corr_bolus_given_error * p_settings
        #     num_events_per_100k_person_years = sim_prob * num_cgm_per_100k_person_years
        #     risk_table_per_sim.add(sim_severity, num_events_per_100k_person_years)
        #
        #     if risk_table_per_sim.is_problematic(sim_severity, num_events_per_100k_person_years):
        #         print(num_events_per_100k_person_years)

        # print("Num sims excluded", num_sims_excluded)

        total_sims += num_total_sims

        # print(low_true, high_true, low_icgm, high_icgm, num_total)
        # print("Severity:", severity, "P(true range, icgm range)", p_error, "\n")

    # risk_table_per_error_bin_patient_prob.print()
    risk_table_per_error_bin_sim_prob.print()
    # risk_table_per_sim.print()

    logger.info("Total sims", total_sims)

    # print(risk_severities)
    # print(risk_severities.values())
    # print([val / sum(list(risk_severities.values())) for val in risk_severities.values()])

    # plt.scatter(lbgi_band, patient_percentages)
    # plt.show()


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


def get_icgm_sim_summary_df(result_dir, save_dir):

    # evaluator = iCGMEvaluator(iCGM_THRESHOLDS, bootstrap_num_values=5000)
    # evaluator.bootstrap_95_lower_confidence_bound([0] * 5 + [99] * 95)

    random_state = np.random.RandomState(0)
    icgm_state = iCGMState(None, 1, 1, iCGM_THRESHOLDS, iCGM_THRESHOLDS, False, 0, random_state)
    # icgm_state = iCGMStateV2(None, 1, 1, iCGM_THRESHOLDS, iCGM_THRESHOLDS, False, 0, random_state)
    risk_table = TPRiskTableRev7()

    max_dfs = np.inf
    logger.info("Loading sims generator...")
    result_generator = load_results(result_dir, ext="tsv", max_dfs=max_dfs)

    sims_by_special_control = {special_control_letter: defaultdict(int) for special_control_letter, prob in iCGM_THRESHOLDS.items()}
    sims_risk_scores = {}

    num_criteria_met = 0
    sim_summary_df = []
    logger.info("Processing sims...")
    num_sims_processed = 1
    for sim_id, result_df in result_generator:
        if num_sims_processed % 1000 == 0:
            logger.info("Num sims processed {}.".format(num_sims_processed))
        start_row_mask = result_df.index == datetime.datetime.strptime("8/15/2019 12:00:00", "%m/%d/%Y %H:%M:%S")
        tbg = result_df[start_row_mask]["bg"].values[0]
        sbg = result_df[start_row_mask]["bg_sensor"].values[0]

        active_sim_mask = result_df["active"] == 1
        true_bg = result_df[active_sim_mask]['bg']
        true_bg[true_bg < 1] = 1
        lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(true_bg)

        range_key = icgm_state.get_bg_range_key(sbg)
        error_key = icgm_state.get_bg_range_error_key(tbg, sbg)

        for special_control_letter, criteria in icgm_state.criteria_to_key_map["range"].items():

            if (range_key, error_key) == criteria:
                risk_score = risk_table.get_severity_index(lbgi_icgm)
                sims_by_special_control[special_control_letter][risk_score] += 1
                sims_risk_scores[sim_id] = (risk_score, tbg, sbg)
                num_criteria_met += 1

                sim_summary_df.append({
                    "sim_id": sim_id,
                    "special_control_letter": special_control_letter,
                    "sbg": sbg,
                    "tbg": tbg,
                    "risk_score": risk_score,
                    "range_key": range_key,
                    "error_key": error_key,
                    "lbgi": lbgi_icgm
                })

        num_sims_processed += 1

    sim_summary_df = pd.DataFrame(sim_summary_df)
    sim_summary_df.to_csv(os.path.join(save_dir, "sim_summary_df.csv"))

    return sim_summary_df


def compute_risk_results(sim_summary_df, save_dir):

    def get_cgm_range_prob(sbg):
        cgm_range_prob = None
        if sbg < 70:
            cgm_range_prob = 0.065
        elif 70 <= sbg <= 180:
            cgm_range_prob = 0.54
        elif sbg > 180:
            cgm_range_prob = 0.388

        return cgm_range_prob

    def get_p_error_given_range(sp_letter, lcb_95):
        p_error = None
        if sp_letter == "D":
            p_error = prob - iCGM_THRESHOLDS["A"]
        elif sp_letter == "E":
            p_error = prob - iCGM_THRESHOLDS["B"]
        elif sp_letter == "F":
            p_error = prob - iCGM_THRESHOLDS["C"]
        else:
            p_error = lcb_95

        return p_error

    p_corr_bolus = 0.01
    num_events_per_severity = defaultdict(int)
    risk_results_df = []
    for sp_control, prob in iCGM_THRESHOLDS.items():

        # if sp_letter in ["A", "B", "C"]:
        #     continue

        # p_error_given_range = prob
        p_error_given_range = get_p_error_given_range(sp_control, prob)

        dexcom_g6_de_novo_sensor_N = [164, 159, 164+159]
        # mu_for_N = compute_bionomial_95_LB_CI_moments(0.99, N_candidates=dexcom_g6_de_novo_sensor_N)

        # With N, and mu, solve for std to get errors prob distribution
        # Then integrate risk over error distribution to get p(severity=5, error=True)

        letter_mask = sim_summary_df["special_control_letter"] == sp_control
        summary_df_letter = sim_summary_df[letter_mask]

        # Plot range, error sampling space with outcomes
        if 0:
            plt.title("Special Control {}".format(sp_control))

            plt.scatter(summary_df_letter["tbg"], summary_df_letter["sbg"], c=summary_df_letter["risk_score"],
                        cmap="Reds", vmin=0, vmax=4)
            plt.xlabel("True BG")
            plt.ylabel("Sensor BG")
            plt.colorbar()
            plt.show()

        total_sims = len(summary_df_letter)
        if total_sims == 0:
            # raise Exception()
            continue

        sample_sbg = summary_df_letter["sbg"].values[0]
        for risk_score in range(5):
            count = sum(summary_df_letter["risk_score"] == risk_score)
            p_severity = count / total_sims

            # if sp_control in ["AD_complement", "BE_complement", "CF_complement"]:
            #     p_severity = 0.0

            p_range = get_cgm_range_prob(sample_sbg)
            prob = p_severity * p_corr_bolus * p_error_given_range * p_range
            n = 100000 * 365 * 288
            num_events = int(prob * n)
            print("\n")
            print(sp_control)
            print("Total Sims: {}. P_severity: {} P_error_g_range: {}. Final Prob: {}".format(total_sims, p_severity, p_error_given_range, prob))
            print("Risk Score: {}. Num Events {}".format(risk_score, num_events))

            risk_results_df.append({
                "sp_control": sp_control,
                "total_sims": total_sims,
                "p_severity": p_severity,
                "p_corr_bolus": p_corr_bolus,
                "p_error_given_range": p_error_given_range,
                "p_range": p_range,
                "risk_score": risk_score,
                "num_events": num_events
            })

            num_events_per_severity[risk_score] += num_events
    # assert num_criteria_met == len(result_generator)

    risk_results_df = pd.DataFrame(risk_results_df)
    risk_results_df.to_csv(os.path.join(save_dir, "risk_results.csv"))

    # a = 1
    # print(num_events_per_severity)
    # for risk_score, count in num_events_per_severity.items():
    #     print(risk_score, risk_table.get_probability_index(count))


if __name__ == "__main__":

    # test_patient_result_dir = "/Users/mconn/data/simulator/processed/test_patient_jun24"
    test_patient_result_dir = "/Users/mconn/data/simulator/processed/icgm-sensitivity-analysis-results-2024-08-13/"

    # plot_sensor_error_vs_loop_prediction_error([
    #     ("with_RC", test_patient_result_dir),
    #     ("no_RC", test_patietn_no_RC_result_dir)
    # ])

    summary_result_filepath = compute_sim_summary_stats(test_patient_result_dir)
    # summary_path = '/Users/mconn/tidepool/repositories/data-science-simulator/result_summary_2024-08-13T14:35:19.834082.csv'
    summary_df = pd.read_csv(summary_result_filepath, sep="\t")
    compute_risk_stats(summary_df)

    # sim_summary_df = get_icgm_sim_summary_df(test_patient_result_dir, save_dir="/Users/mconn/data/simulator/icgm/")
    # compute_risk_results(sim_summary_df, save_dir="/Users/mconn/data/simulator/icgm/")
    # score_risk_table(sim_summary_df)