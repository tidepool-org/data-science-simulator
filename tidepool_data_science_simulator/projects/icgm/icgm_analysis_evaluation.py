__author__ = "Mark Connolly"

import argparse
import re
import logging
import datetime
import warnings
import os
import itertools
import time
import glob
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index
from tidepool_data_science_metrics.insulin.insulin import dka_index

from tidepool_data_science_simulator.models.sensor_icgm import  DexcomG6ValueModel

from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results_generator, collect_sim_result, load_result

logger = logging.getLogger(__name__)

table_probability_indices = {
    (0, 1e-6): 1,
    (1e-6, 1e-4): 2,
    (1e-4, 1e-2): 3,
    (1e-2, 1e-1): 4,
    (.1, 1): 5,
}

def get_probability_index(event_probability):

    for bounds in table_probability_indices.keys():
        if bounds[0] <= event_probability < bounds[1]:
            return table_probability_indices[bounds]

    raise Exception("Probability not in indices.")


def _process_single_simulation(args):
    """
    Worker function to process a single simulation.
    
    This function is designed to be called in parallel via multiprocessing.
    It must be a top-level function (not nested) to be picklable.
    
    Args:
        args: Tuple of (sim_id, sim_json_info)
    
    Returns:
        dict: Summary row for this simulation, or None if processing failed
    """
    sim_id, sim_json_info = args
    
    try:
        # Suppress warnings in worker processes
        warnings.filterwarnings('ignore')
        
        # Load data and calculate risk metrics
        _, df_results = load_result(sim_json_info["result_path"])
        true_bg = np.array(df_results['bg'])        
        true_bg[true_bg < 1] = 1

        # Calculate LBGI based on the default start
        start_index = 137
        
        bg_from_start = true_bg[start_index:]
        lbgi_icgm_start, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(bg_from_start)
        
        # Calculate LBGI based on the first action of Loop
        # for bolus...        
        true_bolus = np.array(df_results['true_bolus'])
        true_bolus = np.where(true_bolus == None, 0.0, true_bolus)
            
        first_valid_bolus = len(true_bolus)
        if np.any(true_bolus > 0):
            first_valid_bolus = np.argmax(true_bolus > 0)
        
        # ... and basal
        true_basal = np.array(df_results['temp_basal'])

        first_valid_basal = len(true_basal)
        if np.any(true_basal > 0):
            first_valid_basal = np.argmax(true_basal > 0)                        

        first_valid_index = min((first_valid_basal, first_valid_bolus))
        
        bg_valid = true_bg[first_valid_index:]
        lbgi_icgm_valid, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(bg_valid)

        # Collect the rest of the information from the run
        bg_cond = int(re.search(r"bg=(\d)", sim_id).groups()[0])
        true_bg_start = sim_json_info["patient"]["sensor"].get("true_start_bg")
        sensor_bg_start = sim_json_info["patient"]["sensor"]["start_bg_with_offset"]
        
        target_bg = 110
        isf = df_results["isf"].values[0]
        max_bolus_delivered = df_results["true_bolus"].max()
        traditional_bolus_delivered = max(0, (sensor_bg_start - target_bg) / isf)

        # Create table row
        row = {
            "sim_id": sim_id,
            "lbgi_icgm_start": lbgi_icgm_start,
            "lbgi_icgm_valid": lbgi_icgm_valid,
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
        
        return row
        
    except Exception as e:
        logger.error(f"Error processing simulation {sim_id}: {str(e)}")
        return None


def process_simulation_data(result_dir, num_workers=None, batch_size=1000):
    """
    Process simulation data with parallel processing using batches to manage memory.
    
    Args:
        result_dir: Directory containing simulation results
        num_workers: Number of parallel workers (default: CPU count)
        batch_size: Number of simulations to process per batch (default: 1000)
        
    
    Returns:
        str: Path to the saved summary CSV file
    """

    # Get rid of unnecessary warnings for low/high BG
    warnings.filterwarnings('ignore')
    
    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count() - 1 # Leave one CPU free
    
    logger.info(f"Processing simulations with {num_workers} parallel workers in batches of {batch_size}")
    
    sim_id_pattern_regex = "vp.*bg.*.json"
    sim_id_pattern_glob = "*vp*bg*.json"

    # Count total files for progress tracking
    pattern = os.path.join(result_dir, sim_id_pattern_glob)
    matching_files = glob.glob(pattern)
    total_files = len(matching_files)
    expected_batches = (total_files + batch_size - 1) // batch_size  # Ceiling division
    
    logger.info(f"Found {total_files} simulation files to process")
    logger.info(f"Expected batches: {expected_batches}")

    sim_results = collect_sims_and_results_generator(
        result_dir, 
        sim_id_pattern=sim_id_pattern_regex, 
        max_sims=1e12
    )
    
    # Process simulations in batches to avoid memory issues
    summary_data = []
    total_processed = 0
    batch_num = 0
    batch_times = []
    overall_start_time = time.time()
    
    # Process each batch with a fresh pool
    while True:
        # Get next batch of simulations
        batch = list(itertools.islice(sim_results, batch_size))
        if not batch:
            break  # No more simulations
        
        batch_num += 1
        batch_start = total_processed + 1
        batch_end = total_processed + len(batch)
        
        logger.info(f"Processing batch {batch_num}/{expected_batches}: simulations {batch_start}-{batch_end}")
        
        # Process this batch with a fresh pool and time it
        batch_start_time = time.time()
        with Pool(processes=num_workers) as pool:
            batch_results = pool.map(_process_single_simulation, batch, chunksize=10)
        batch_duration = time.time() - batch_start_time
        batch_times.append(batch_duration)
        
        # Collect results from this batch
        batch_data = [r for r in batch_results if r is not None]
        summary_data.extend(batch_data)
        
        total_processed += len(batch)
        
        # Calculate progress and ETA
        progress_pct = (batch_num / expected_batches) * 100
        avg_batch_time = sum(batch_times) / len(batch_times)
        remaining_batches = expected_batches - batch_num
        estimated_remaining_sec = remaining_batches * avg_batch_time
        estimated_remaining_min = estimated_remaining_sec / 60
        elapsed_min = (time.time() - overall_start_time) / 60
        
        logger.info(f"Batch {batch_num}/{expected_batches} complete: {len(batch_data)}/{len(batch)} successful | "
                   f"Progress: {progress_pct:.1f}% | "
                   f"Batch time: {batch_duration:.1f}s | "
                   f"Avg: {avg_batch_time:.1f}s | "
                   f"ETA: {estimated_remaining_min:.1f} min | "
                   f"Elapsed: {elapsed_min:.1f} min")
    
    total_elapsed_min = (time.time() - overall_start_time) / 60
    logger.info(f"Completed processing all {total_processed} simulations across {batch_num} batches in {total_elapsed_min:.1f} minutes")

    summary_df = pd.DataFrame(summary_data)
    
    summary_result_filepath = result_dir + '.csv'
    summary_df.to_csv(summary_result_filepath, sep="\t")
    logger.info("Saved summary results to %s", summary_result_filepath)

    return summary_result_filepath


def compute_score_risk_table(summary_df, concurrency_table=None):

    dexcom_value_model = DexcomG6ValueModel(concurrency_table=concurrency_table)

    bg_ranges = [(40, 60),(61, 80), (81, 120), (121, 160), (161, 200), 
                 (201, 250), (251, 300), (301, 350), (351, 400)]  
    
    bg_range_pairs = [(true_range,icgm_range) for true_range in bg_ranges for icgm_range in bg_ranges]
    
    severity_bands = [(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)]

    severity_event_count = [0,0,0,0,0]
    low_true_axis = []
    low_icgm_axis = []
    mean_lbgi_start = []
    mean_lbgi_valid = []
    joint_prob = []

    # Go through each square in the concurrency table 
    for (low_true, high_true), (low_icgm, high_icgm) in bg_range_pairs:
        low_true_axis.append(low_true)
        low_icgm_axis.append(low_icgm)

        # Backward compatibility with old versions of the results file. 
        if "true_start_bg" in summary_df:
            # Current version
            true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)
            icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (summary_df["start_bg_with_offset"] <= high_icgm)

        elif "tbg" in summary_df:
            # 2021 version
            true_mask = (summary_df["tbg"] >= low_true) & (summary_df["tbg"] <= high_true)
            icgm_mask = (summary_df["sbg"] >= low_icgm) & (summary_df["sbg"] <= high_icgm)

        else:
            return

        concurrency_square_mask = true_mask & icgm_mask
        lbgi_data = []
        lbgi_data_valid = []

        if "lbgi_icgm" in summary_df:
            lbgi_data = summary_df[concurrency_square_mask]["lbgi_icgm"]
            
        elif "lbgi" in summary_df:
            lbgi_data = summary_df[concurrency_square_mask]["lbgi"]        
        elif "lbgi_icgm_valid" in summary_df:
            lbgi_data = summary_df[concurrency_square_mask]["lbgi_icgm_start"]        
            lbgi_data_valid = summary_df[concurrency_square_mask]["lbgi_icgm_valid"]        
        else:
            return
        # End backward compatibility

        p_error = dexcom_value_model.get_joint_probability(low_true, low_icgm)

        joint_prob.append(p_error)

        p_corr_bolus_given_error = 6 / 288 # = 1/48
        num_cgm_per_100k_person_years = 288 * 365 * 100000
        num_sims_in_concurrency_square = max(1, len(summary_df[concurrency_square_mask]))
        
        sim_prob_start = []
        sim_prob_valid = []

        if low_true == 40 and (low_icgm == 40 or low_icgm == 61):
            mean_lbgi_start.append(np.zeros(5))

        else:
            for s_idx, severity_band in enumerate(severity_bands, 0):
                severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])
                num_sims_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask])
                sim_prob_start.append(num_sims_in_severity_band / num_sims_in_concurrency_square)
                
                risk_prob_sim = sim_prob_start[s_idx] * p_corr_bolus_given_error * p_error
                num_risk_events_sim = risk_prob_sim * num_cgm_per_100k_person_years

                severity_event_count[s_idx] += num_risk_events_sim
                ####
                # if "lbgi_icgm_valid" in summary_df:
                #     severity_mask = (lbgi_data_valid >= severity_band[0]) & (lbgi_data_valid < severity_band[1])
                #     num_sims_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask])
                #     sim_prob_valid.append(num_sims_in_severity_band / num_sims_in_concurrency_square)
                    
                #     risk_prob_sim = sim_prob_valid[s_idx] * p_corr_bolus_given_error * p_error
                #     num_risk_events_sim = risk_prob_sim * num_cgm_per_100k_person_years

                #     severity_event_count[s_idx] += num_risk_events_sim

       
        
            mean_lbgi_start.append(sim_prob_start)
        # mean_lbgi_valid.append(sim_prob_valid)


    severity_event_count_df = pd.DataFrame(severity_event_count)
    severity_event_probability_df = severity_event_count_df / num_cgm_per_100k_person_years 

    return severity_event_probability_df, (low_icgm_axis, low_true_axis, np.array(mean_lbgi_start), np.array(joint_prob))


if __name__ == "__main__":
    # parser = argparse.ArgumentParser("icgm_analysis_evaluation")
    # parser.add_argument("mode", help="process or summarize", type=str)
    # parser.add_argument("path", help="simulation data directory (process) or summary file path (summarize)", type=str)
    # args = parser.parse_args()

    # mode = args.mode
    # path = args.path

    mode = 'process'
    path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/icgm_sensitivity_analysis_paf=0.4_posrc=False_2025_07_23_T_13_56_44_ae0a0c7d'
    path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/icgm_sensitivity_analysis_paf=0.2_posrc=False_2025_07_23_T_19_49_26_0f59469a'
    path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/icgm_sensitivity_analysis_paf=0.6_posrc=False_2025_07_24_T_14_00_27_658d0e12'
    # path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/icgm_sensitivity_analysis_paf=0.8_posrc=False_2025_07_24_T_20_07_52_14d7f7d4'
    # mode = 'summarize'
    # path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/icgm_sensitivity_analysis_paf=0.4_posrc=False_2025_07_23_T_13_56_44_ae0a0c7d.csv'
    # # path = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/icgm_sensitivity_analysis_paf=0.2_posrc=False_2025_07_23_T_19_49_26_0f59469a.csv'

    match mode:
        case 'process': 
            summary_result_filepath = process_simulation_data(path)
       
        case 'summarize': 
            summary_df = pd.read_csv(path, sep="\t")
            print(compute_score_risk_table(summary_df, 'coastal'))
