__author__ = "Cameron Summers"

import logging
import time
import json
import os
import subprocess

import numpy as np
import pandas as pd

import pdb

# Setup Logging
logger = logging.getLogger(__name__)

from tidepool_data_science_simulator.utils import timing, save_df
from tidepool_data_science_metrics.glucose.glucose import (
    blood_glucose_risk_index, percent_values_ge_70_le_180, percent_values_lt_40, percent_values_lt_54,
    percent_values_gt_180, percent_values_gt_250, lbgi_risk_score,
)
from tidepool_data_science_metrics.insulin.insulin import (
    dka_risk_score, dka_index
)


@timing
def run_simulations(sims, save_dir,
                    save_results=True,
                    compute_summary_metrics=True,
                    num_procs=1, name=''):
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

    full_results = dict()
    summary_results = []

    # Process sims in batches of num_procs
    for sim_id, sim in sims.items():
        logger.debug("Running: {}. {} of {}".format(sim_id, sim_ctr, num_sims))
        sim.start()
        running_sims[sim_id] = sim

        batch_start_time = time.time()
        if len(running_sims) >= num_procs or sim_ctr >= num_sims:  # Batch condition

            # Gather results from sim queues
            batch_results = {id: sim.queue.get() for id, sim in running_sims.items()}
            # [sim.join() for id, sim in running_sims.items()]
            for sim_id, sim in running_sims.items():
                sim.join()
                sim.close()


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
                    save_df(results_df, sim_id, save_dir)

                full_results[sim_id] = results_df

        sim_ctr += 1

    logger.debug("Full run time: {:.2f}m".format((time.time() - run_start_time) / 60.0))

    summary_results_df = pd.DataFrame(summary_results)
    summary_results_df.set_index("sim_id", inplace=True)

    if save_results:
        # summary_results_df.to_csv(os.path.join(save_dir, "summary_results_{}.csv".format(time.time())))
        summary_results_df.to_csv(os.path.join(save_dir, f"summary_results_{name}.csv"))

    return full_results, summary_results_df
