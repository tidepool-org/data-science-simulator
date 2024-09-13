from collections import defaultdict
import datetime
import logging
import re

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from tidepool_data_science_simulator.evaluation.inspect_results import load_results
from tidepool_data_science_simulator.models.sensor_icgm import DexcomG6ValueModel
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import TPRiskTableRev7, compute_score_risk_table
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

logger = logging.getLogger(__name__)

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


def compute_risk_results(sim_summary_df):

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