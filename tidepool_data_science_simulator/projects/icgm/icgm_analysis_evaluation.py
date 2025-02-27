__author__ = "Mark Connolly"

import argparse
import re
import logging
import datetime
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index
from tidepool_data_science_metrics.insulin.insulin import dka_index

from tidepool_data_science_simulator.models.sensor_icgm import  DexcomG6ValueModel

from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results_generator, collect_sim_result, load_result

logger = logging.getLogger(__name__)

def process_simulation_data(result_dir):
    # Get rid of unnecessary warnings for low/high BG
    warnings.filterwarnings('ignore')
    
    sim_id_pattern_regex="vp.*bg.*.json" 

    sim_results = collect_sims_and_results_generator(
        result_dir, 
        sim_id_pattern=sim_id_pattern_regex, 
        max_sims=1e12
    )
    
    summary_data = []
    i = 0
    for sim_id, sim_json_info in sim_results:
        
        i += 1
        if i % 1000 == 0:
            logger.info("%s",i)
        
        # Load data and calculate risk metrics
        _, df_results = load_result(sim_json_info["result_path"])
        true_bg = np.array(df_results['bg'])
        true_bg[true_bg < 1] = 1
        lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(true_bg)
        dkai_icgm = dka_index(df_results['iob'], df_results["sbr"])

        # Find the simulation where the true and sensor blood glucose match
        # Could also run with the ideal sensor class
        sim_results_match = re.search(r"tbg=(\d+)", sim_id)
        ideal_sbg = sim_results_match.group(1)
        ideal_sbg_string = "sbg=" + ideal_sbg + ".json"

        ideal_sbg_file = re.sub(r"sbg=(\d+)", ideal_sbg_string, sim_id)   
        sim_json_info_ideal = collect_sim_result(result_dir, ideal_sbg_file)

        # Load data with ideal sensor and calculate risk metrics
        # Will be used to filter our cases where the risk was already high
        _, df_results_ideal = load_result(sim_json_info_ideal["result_path"])
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


def compute_score_risk_table(summary_df):

    dexcom_value_model = DexcomG6ValueModel(concurrency_table="adult")

    bg_ranges = [(40, 60),(61, 80), (81, 120), (121, 160), (161, 200), 
                 (201, 250), (251, 300), (301, 350), (351, 400)]  
    
    bg_range_pairs = [(true_range,icgm_range) for true_range in bg_ranges for icgm_range in bg_ranges]
    severity_bands = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]

    severity_event_count = [0,0,0,0,0]

    for (low_true, high_true), (low_icgm, high_icgm) in bg_range_pairs:
        
        # Backward compatibility with old versions of the results file. 
        if "true_start_bg" in summary_df:
            true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)
            icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (summary_df["start_bg_with_offset"] <= high_icgm)

        elif "tbg" in summary_df:
            true_mask = (summary_df["tbg"] >= low_true) & (summary_df["tbg"] <= high_true)
            icgm_mask = (summary_df["sbg"] >= low_icgm) & (summary_df["sbg"] <= high_icgm)

        else:
            return

        concurrency_square_mask = true_mask & icgm_mask 

        if "lbgi_icgm" in summary_df:
            lbgi_data = summary_df[concurrency_square_mask]["lbgi_icgm"]
        elif "lbgi" in summary_df:
            lbgi_data = summary_df[concurrency_square_mask]["lbgi"]        
        else:
            return
        # End backward compatibility

        p_error = dexcom_value_model.get_joint_probability(low_true, low_icgm)
        p_corr_bolus_given_error = 6 / 288
        num_cgm_per_100k_person_years = 288 * 365 * 100000

        num_total_sims = max(1, len(summary_df[concurrency_square_mask]))

        for s_idx, severity_band in enumerate(severity_bands, 0):
            severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])

            num_sims_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask])
            sim_prob = num_sims_in_severity_band / num_total_sims
            risk_prob_sim = sim_prob * p_corr_bolus_given_error * p_error
            num_risk_events_sim = risk_prob_sim * num_cgm_per_100k_person_years

            severity_event_count[s_idx] += num_risk_events_sim

    severity_event_count_df = pd.DataFrame(severity_event_count)
    severity_event_probability_df = severity_event_count_df / num_cgm_per_100k_person_years 

    severity_event_probability_df.to_csv('severity_event_probability.csv')

    return severity_event_probability_df


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
            print(compute_score_risk_table(summary_df))
