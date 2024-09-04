__author__ = "Mark Connolly"

import argparse
import re
import os
import logging
import datetime
import warnings

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

from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results_generator, collect_sim_result, load_result
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results, plot_sim_icgm_paired


logger = logging.getLogger(__name__)


def process_simulation_data(result_dir):
    # Get rid of unnecessary warnings for low/high BG
    warnings.filterwarnings('ignore')
    
    sim_id_pattern="vp.*bg.*.json" 

    sim_results = collect_sims_and_results_generator(
        result_dir, 
        sim_id_pattern=sim_id_pattern, 
        max_sims=1e12
    )
    
    summary_data = []
    i = 0
    for sim_id, sim_json_info in sim_results:
        
        i += 1
        if i % 1000 == 0:
            logger.info("%s",i)
        
        sim_results_match = re.search(r"tbg=(\d+)", sim_id)
        ideal_sbg = sim_results_match.group(1)
        ideal_sbg_string = "sbg=" + ideal_sbg + ".json"

        ideal_sbg_file = re.sub(r"sbg=(\d+)", ideal_sbg_string, sim_id)
   
        sim_json_info_match = collect_sim_result(result_dir, ideal_sbg_file)

        _, df_results = load_result(sim_json_info["result_path"])
        true_bg = np.array(df_results['bg'])
        true_bg[true_bg < 1] = 1
        lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(true_bg)
        dkai_icgm = dka_index(df_results['iob'], df_results["sbr"])

        _, df_results_ideal = load_result(sim_json_info_match["result_path"])
        true_bg = np.array(df_results_ideal["bg"])
        true_bg[true_bg < 1] = 1
        lbgi_ideal, hbgi_ideal, brgi_ideal = blood_glucose_risk_index(true_bg)
        dkai_ideal = dka_index(df_results_ideal['iob'], df_results_ideal["sbr"])

        bg_cond = int(re.search(r"bg=(\d)", sim_id).groups()[0])
        
        true_bg_start = sim_json_info["patient"]["sensor"].get("true_start_bg")

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

    summary_result_filepath = "./processed_simulation_data_{}.csv".format(datetime.datetime.now().isoformat())
    summary_df.to_csv(summary_result_filepath, sep="\t")
    logger.info("Saved summary results to %s", summary_result_filepath)

    return summary_result_filepath

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


def compute_score_risk_table(summary_df):

    dexcom_value_model = DexcomG6ValueModel(concurrency_table="Coastal")

    summary_df["vp_id"] = summary_df["sim_id"].apply(lambda sim_id: re.search(r"(vp.*).bg=\d", sim_id).groups()[0])

    total_sims = 0
    bg_ranges = ((0, 40),(40, 60),(61, 80), (81, 120), 
                 (121, 160), (161, 200), (201, 250), (251, 300), 
                 (301, 350), (351, 400), (400, 50000))        
    
    bg_range_pairs = [(true_range,icgm_range) for true_range in bg_ranges for icgm_range in bg_ranges]
    severity_bands = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]

    severity_event_probability = [0,0,0,0,0]
    for (low_true, high_true), (low_icgm, high_icgm) in bg_range_pairs:

        true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)
        icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (summary_df["start_bg_with_offset"] <= high_icgm)
        initially_ok_mask = summary_df["lbgi_ideal"] == 0.0

        # concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask
        concurrency_square_mask = true_mask & icgm_mask 

        p_error = dexcom_value_model.get_joint_probability(low_true, low_icgm)
        p_corr_bolus_given_error = 3 / 288
        num_cgm_per_100k_person_years = 288 * 365 * 100000

        num_initially_ok = np.sum(initially_ok_mask)
        num_range_mask = np.sum(true_mask & icgm_mask)
        num_total_sims = max(1, len(summary_df[concurrency_square_mask]))

        lbgi_data = summary_df[concurrency_square_mask]["lbgi_icgm"]        
        
        for s_idx, severity_band in enumerate(severity_bands, 1):
            severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])

            num_sims_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask])
            sim_prob = num_sims_in_severity_band / num_total_sims
            risk_prob_sim = sim_prob * p_corr_bolus_given_error * p_error
            num_risk_events_sim = risk_prob_sim * num_cgm_per_100k_person_years

            severity_event_probability[s_idx-1] += num_risk_events_sim

    severity_event_probability_df = pd.DataFrame(severity_event_probability)
    severity_event_probability_df.to_csv('severity_event_probability.csv')

    print(severity_event_probability_df)
    return

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
    # risk_results_df.to_csv(os.path.join(save_dir, "risk_results.csv"))

    # a = 1
    # print(num_events_per_severity)
    # for risk_score, count in num_events_per_severity.items():
    #     print(risk_score, risk_table.get_probability_index(count))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("icgm_analysis_evaluation")
    parser.add_argument("mode", help="process or summarize", type=str)
    parser.add_argument("path", help="simulation data directory (process) or summary file path (summarize)", type=str)
    args = parser.parse_args()

    mode = args.mode
    path = args.path
    
    match mode:
        case 'process': 
            summary_result_filepath = process_simulation_data(path)
       
        case 'summarize': 
            summary_df = pd.read_csv(path, sep="\t")
            compute_score_risk_table(summary_df)
