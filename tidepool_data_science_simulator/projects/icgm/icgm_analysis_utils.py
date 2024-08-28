from collections import defaultdict
import datetime
import logging
import re

from matplotlib import pyplot as plt
import numpy as np
from tidepool_data_science_simulator.evaluation.inspect_results import load_results
from tidepool_data_science_simulator.models.sensor_icgm import DexcomG6ValueModel
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import TPRiskTableRev7, compute_score_risk_table
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

logger = logging.getLogger(__name__)

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

def compute_risk_stats(summary_df):

    # compute_dka_risk_tp_icgm(summary_df)
    # compute_lbgi_risk_tp_icgm_negative_bias(summary_df)

    compute_score_risk_table(summary_df)