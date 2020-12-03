__author__ = "Cameron Summers"

import logging
import time
import json
import os
import subprocess

import numpy as np

import pdb

# Setup Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
LOG_FILENAME = "sim.log"
filehandler = logging.FileHandler(LOG_FILENAME)
logger.addHandler(filehandler)

from tidepool_data_science_simulator.utils import timing, save_df

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, percent_values_ge_70_le_180


@timing
def run_simulations(sims, save_dir,
                    save_results=True,
                    compute_summary_metrics=True,
                    num_procs=1):
    """
    Run the simulations passed as argument and optionally process, save, or plot the results.

    Parameters
    ----------
    sims: dict
        Dict of sim_id to simulation object to run

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
    """
    current_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")

    logger.debug("Results Directory: {}".format(save_dir))
    logger.debug("Current Code Commit: {}".format(current_commit))

    num_sims = len(sims)
    sim_ctr = 1
    running_sims = {}
    run_start_time = time.time()

    all_results = {}
    # Process sims in batches of num_procs
    for sim_id, sim in sims.items():
        logger.debug("Running: {}. {} of {}".format(sim_id, sim_ctr, num_sims))
        sim.start()
        running_sims[sim_id] = sim

        batch_start_time = time.time()
        if len(running_sims) >= num_procs or sim_ctr >= num_sims:  # Batch condition

            # Gather results from sim queues
            batch_results = {id: sim.queue.get() for id, sim in running_sims.items()}
            [sim.join() for id, sim in running_sims.items()]

            # Save stateless info
            if save_results:
                for id, sim in running_sims.items():
                    info = sim.get_info_stateless()
                    json.dump(info, open(os.path.join(save_dir, "{}.json".format(id)), "w"), indent=4)

            running_sims = {}  # reset for next batch

            logger.debug("Batch run time: {:.2f}m".format((time.time() - batch_start_time) / 60.0))
            logger.debug("Total run time: {:.2f}m".format((time.time() - run_start_time) / 60.0))

            # Summarize, save, or plot results
            for sim_id, results_df in batch_results.items():

                if compute_summary_metrics:
                    try:
                        lbgi, hbgi, brgi = blood_glucose_risk_index(results_df['bg'])
                        basal_delivered = results_df["delivered_basal_insulin"].sum()
                        bolus_delivered = results_df["reported_bolus"].sum()
                        total_delivered = basal_delivered + bolus_delivered
                        summary_str = "Sim {}. \n\tLBGI: {} HBGI: {} BRGI: {}\n\t Basal {}. Bolus {}. Total {}".format(sim_id, lbgi, hbgi, brgi, basal_delivered, bolus_delivered, total_delivered)
                        logger.debug(summary_str)

                        sensor_mard = np.mean(np.abs(results_df["bg"] - results_df["bg_sensor"]) / results_df["bg"])
                        sensor_mbe = np.mean(results_df["bg_sensor"] - results_df["bg"])
                        logger.debug("Sensor Stats: MBE: {}. MARD: {}".format(sensor_mbe, sensor_mard))
                    except Exception as e:
                        logger.debug("Exception occurred in computed summary metrics")

                # Sanity debugging random stream sync
                logger.debug("Final Random Int: {}".format(results_df.iloc[-1]["randint"]))

                if save_results:
                    save_df(results_df, sim_id, save_dir)

                all_results[sim_id] = results_df

        sim_ctr += 1

    logger.debug("Full run time: {:.2f}m".format((time.time() - run_start_time) / 60.0))

    #os.rename(LOG_FILENAME, os.path.join(save_dir, LOG_FILENAME))

    return all_results
