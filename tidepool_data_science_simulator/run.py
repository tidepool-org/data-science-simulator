__author__ = "Cameron Summers"

import logging
import time
import json
import os
import subprocess
import math
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from tidepool_data_science_simulator.utils import timing, save_df, StreamingParquetWriter
from tidepool_data_science_metrics.glucose.glucose import (
    blood_glucose_risk_index, percent_values_ge_70_le_180, percent_values_lt_40, percent_values_lt_54,
    percent_values_gt_180, percent_values_gt_250, lbgi_risk_score,
)
from tidepool_data_science_metrics.insulin.insulin import (
    dka_risk_score, dka_index
)

# Setup Logging
logger = logging.getLogger(__name__)


def _format_time(seconds):
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        mins = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{mins}m {secs}s"
    else:
        hours = int(seconds // 3600)
        mins = int((seconds % 3600) // 60)
        return f"{hours}h {mins}m"


def _create_progress_bar(percent, width=40):
    """Create a text-based progress bar."""
    filled = int(width * percent / 100)
    bar = "█" * filled + "░" * (width - filled)
    return f"[{bar}]"


@timing
def run_simulations(sims, save_dir,
                    save_results=True,
                    compute_summary_metrics=True,
                    num_procs=1,
                    return_full_results=True,
                    num_sims=None,
                    save_format='tsv'):
    """
    Run the simulations passed as argument and optionally process, save, or plot the results.

    Parameters
    ----------
    sims: dict or generator
        Dict of sim_id to simulation object to run, OR a generator yielding (sim_id, sim) tuples.
        When using a generator, provide num_sims for progress tracking.

    save_dir: str
        Path to save results

    save_results: bool
        If True save results

    plot_results: bool
        If True plot results

    compute_summary_metrics: bool
        If True compute summary metrics on simulations at run time

    num_procs: int
        Number of processes for multiprocessing
    
    return_full_results: bool
        If True, return full_results dictionary. Set to False to save memory when results are saved to disk.
    
    num_sims: int, optional
        Total number of simulations. Required for progress tracking with generators.
        If not provided and sims is a dict, will use len(sims).
    
    save_format: str
        Output format for individual simulation results. Options:
        - 'tsv' (default): Save as TSV files with separate JSON metadata files (backward compatible)
        - 'parquet': Save as Parquet files with embedded JSON metadata (~70% smaller, faster reads)
        - 'both': Save in both formats
    """
    current_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")

    # Create save directory if it doesn't exist
    if save_results and save_dir:
        os.makedirs(save_dir, exist_ok=True)

    # Handle both dict and generator inputs (backward compatible)
    if hasattr(sims, 'items'):
        # It's a dict - backward compatible behavior
        sim_iterator = sims.items()
        if num_sims is None:
            num_sims = len(sims)
    else:
        # It's a generator/iterator yielding (sim_id, sim) tuples
        sim_iterator = sims

    # Calculate batch information for progress tracking
    total_batches = math.ceil(num_sims / num_procs) if num_sims else None
    
    # ==================== STARTUP LOGGING ====================
    logger.info("=" * 80)
    logger.info("SIMULATION RUN STARTED")
    logger.info("=" * 80)
    if num_sims:
        logger.info(f"Total Simulations: {num_sims} | Batch Size: {num_procs} | Total Batches: {total_batches}")
    else:
        logger.info(f"Total Simulations: Unknown (generator) | Batch Size: {num_procs}")
    logger.info(f"Output Directory: {save_dir}")
    logger.info(f"Save Format: {save_format} | Save Results: {save_results}")
    logger.info(f"Git Commit: {current_commit}")
    logger.info("-" * 80)

    sim_ctr = 1
    batch_ctr = 0
    running_sims = {}
    run_start_time = time.time()

    full_results = dict()
    summary_results = []
    
    # Progress tracking variables
    batch_times = []  # Track batch execution times for ETA calculation
    sims_completed = 0
    
    # Initialize streaming parquet writer if needed (writes incrementally to avoid memory issues)
    parquet_writer = None
    if save_results and save_format in ('parquet', 'both'):
        parquet_writer = StreamingParquetWriter(save_dir)

    # Process sims in batches of num_procs
    for sim_id, sim in sim_iterator:

        sim.start()
        running_sims[sim_id] = sim

        batch_start_time = time.time()
        # Batch condition: process when batch is full, or at end if num_sims known
        batch_full = len(running_sims) >= num_procs
        at_end = num_sims is not None and sim_ctr >= num_sims
        if batch_full or at_end:
            batch_ctr += 1
            batch_size = len(running_sims)
            
            # Log batch start
            if num_sims and total_batches:
                sim_range_start = sims_completed + 1
                sim_range_end = sims_completed + batch_size
                logger.info(f"\n[Batch {batch_ctr}/{total_batches}] Processing simulations {sim_range_start}-{sim_range_end}...")
            else:
                logger.info(f"\n[Batch {batch_ctr}] Processing {batch_size} simulations...")

            # Gather results from sim queues
            batch_results = {id: sim.queue.get() for id, sim in running_sims.items()}
            [sim.join() for id, sim in running_sims.items()]
            
            # Calculate batch timing
            batch_elapsed = time.time() - batch_start_time
            batch_times.append(batch_elapsed)
            sims_completed += batch_size
            avg_time_per_sim = batch_elapsed / batch_size if batch_size > 0 else 0
            
            # Calculate running average across all batches
            total_avg_per_sim = sum(batch_times) / sims_completed if sims_completed > 0 else 0

            # Collect simulation info for saving (used by both formats)
            sim_info_dict = {}
            if save_results:
                for id, sim in running_sims.items():
                    sim_info_dict[id] = sim.get_info_stateless()
                    
                    # Save JSON separately only for TSV format (backward compatible)
                    if save_format in ('tsv', 'both'):
                        json.dump(sim_info_dict[id], open(os.path.join(save_dir, "{}.json".format(id)), "w"), indent=4)

            running_sims = {}  # reset for next batch

            # ==================== BATCH COMPLETION LOGGING ====================
            elapsed_total = time.time() - run_start_time
            
            if num_sims and total_batches:
                percent_complete = (sims_completed / num_sims) * 100
                sims_remaining = num_sims - sims_completed
                
                # Calculate ETA using exponential moving average of recent batch times
                if len(batch_times) >= 3:
                    # Use weighted average of recent batches (more weight on recent)
                    recent_times = batch_times[-5:]  # Last 5 batches
                    weights = list(range(1, len(recent_times) + 1))
                    weighted_avg = sum(t * w for t, w in zip(recent_times, weights)) / sum(weights)
                    avg_per_sim_for_eta = weighted_avg / num_procs
                else:
                    avg_per_sim_for_eta = total_avg_per_sim
                
                eta_seconds = sims_remaining * avg_per_sim_for_eta
                eta_str = _format_time(eta_seconds)
                completion_time = datetime.now() + timedelta(seconds=eta_seconds)
                completion_str = completion_time.strftime("%H:%M:%S")
                
                progress_bar = _create_progress_bar(percent_complete)
                
                logger.info(f"[Batch {batch_ctr}/{total_batches}] Completed in {_format_time(batch_elapsed)} | "
                           f"Avg: {avg_time_per_sim:.2f}s/sim")
                logger.info(f"Progress: {progress_bar} {percent_complete:.1f}% ({sims_completed}/{num_sims})")
                logger.info(f"Elapsed: {_format_time(elapsed_total)} | ETA: {eta_str} | Est. Completion: {completion_str}")
            else:
                # Unknown total - show what we can
                logger.info(f"[Batch {batch_ctr}] Completed in {_format_time(batch_elapsed)} | "
                           f"Avg: {avg_time_per_sim:.2f}s/sim")
                logger.info(f"Simulations completed: {sims_completed} | Elapsed: {_format_time(elapsed_total)}")
            
            logger.debug("Batch run time: {:.2f}m".format(batch_elapsed / 60.0))
            logger.debug("Total run time: {:.2f}m".format(elapsed_total / 60.0))

            # Summarize, save, or plot results
            for sim_id, results_df in batch_results.items():

                metrics_df = results_df[results_df["active"] == 1]
                if compute_summary_metrics:
                    try:
                        true_bg_trace_clipped = np.array([min(401, max(1, val)) for val in metrics_df['bg']])
                        lbgi, hbgi, brgi = blood_glucose_risk_index(true_bg_trace_clipped)
                        dka_index_value = dka_index(metrics_df["iob"], metrics_df["sbr"].values[0])
                        basal_delivered = metrics_df["delivered_basal_insulin"].sum()
                        bolus_delivered = metrics_df["reported_bolus"].sum()
                        total_delivered = basal_delivered + bolus_delivered
                        summary_str = "Sim {}. \n\tMean BG: {} LBGI: {} HBGI: {} BRGI: {}\n\t Basal {}. Bolus {}. Total {}".format(sim_id, np.mean(true_bg_trace_clipped), lbgi, hbgi, brgi, basal_delivered, bolus_delivered, total_delivered)
                        logger.debug(summary_str)

                        sensor_mard = np.mean(np.abs(metrics_df["bg"] - metrics_df["bg_sensor"]) / metrics_df["bg"])
                        sensor_mbe = np.mean(metrics_df["bg_sensor"] - metrics_df["bg"])
                        logger.debug("Sensor Stats: MBE: {}. MARD: {}".format(sensor_mbe, sensor_mard))

                        summary_results.append({
                            "sim_id": sim_id,
                            "total_basal_delivered": basal_delivered,
                            "total_bolus_delivered": bolus_delivered,
                            "total_insulin_delivered": total_delivered,
                            "sensor_mard": sensor_mard,
                            "sensor_mbe": sensor_mbe,
                            "lbgi": lbgi,
                            "hbgi": hbgi,
                            "brgi": brgi,
                            "lbgi_risk_score": lbgi_risk_score(lbgi),
                            "dka_index": dka_index_value,
                            "dka_risk_score": dka_risk_score(dka_index_value),
                            "percent_cgm_lt_40": percent_values_lt_40(true_bg_trace_clipped),
                            "percent_cgm_lt_54": percent_values_lt_54(true_bg_trace_clipped),
                            "percent_cgm_gt_180": percent_values_gt_180(true_bg_trace_clipped),
                            "percent_cgm_gt_250": percent_values_gt_250(true_bg_trace_clipped),
                            "percent_values_ge_70_le_180": percent_values_ge_70_le_180(true_bg_trace_clipped)
                        })
                    except Exception as e:
                        logger.debug("Exception occurred in computed summary metrics. {}".format(e))
                        summary_results.append({
                            "sim_id": sim_id
                        })

                # Sanity debugging random stream sync
                logger.debug("Final Random Int: {}".format(results_df.iloc[-1]["randint"]))

                if save_results:
                    if save_format in ('tsv', 'both'):
                        save_df(results_df, sim_id, save_dir)
                    if save_format in ('parquet', 'both') and parquet_writer:
                        # Stream results to parquet file incrementally (no memory accumulation)
                        metadata = sim_info_dict.get(sim_id)
                        parquet_writer.write_batch(results_df, sim_id, metadata)

                if return_full_results:
                    full_results[sim_id] = results_df

        sim_ctr += 1

    # ==================== COMPLETION LOGGING ====================
    total_runtime = time.time() - run_start_time
    avg_time_per_sim_final = total_runtime / sims_completed if sims_completed > 0 else 0
    
    logger.info("\n" + "=" * 80)
    logger.info("SIMULATION RUN COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total Simulations Completed: {sims_completed}")
    logger.info(f"Total Batches: {batch_ctr}")
    logger.info(f"Total Runtime: {_format_time(total_runtime)}")
    logger.info(f"Average Time per Simulation: {avg_time_per_sim_final:.2f}s")
    
    # Finalize streaming parquet file
    if parquet_writer:
        parquet_writer.close()
        logger.info(f"Parquet Output: {os.path.join(save_dir, 'combined_results.parquet')}")

    if compute_summary_metrics:
        summary_results_df = pd.DataFrame(summary_results)
        summary_results_df.set_index("sim_id", inplace=True)

        if save_results:
            summary_path = os.path.join(save_dir, "summary_results_{}.csv".format(time.time()))
            summary_results_df.to_csv(summary_path)
            logger.info(f"Summary Results: {summary_path}")

    else:
        summary_results_df = None

    if save_results:
        logger.info(f"Output Directory: {save_dir}")
    logger.info("=" * 80)
    
    logger.debug("Full run time: {:.2f}m".format(total_runtime / 60.0))

    return full_results, summary_results_df
