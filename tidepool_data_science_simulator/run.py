__author__ = "Cameron Summers"

import logging
import time
import json
import os
import subprocess
import traceback

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


class SimulationRunError(Exception):
    """Raised by run_simulations, after every sim has finished, when at least one sim failed.

    Attributes
    ----------
    failures: list of dict
        One record per failed sim with keys sim_id, stage ('run', 'metrics' or 'save'),
        exception_type, message and traceback (the formatted traceback string).
    full_results: dict
        sim_id to results DataFrame for the sims that succeeded.
    summary_results: pandas.DataFrame
        The summary metrics for the sims that succeeded, indexed by sim_id. Failed sims
        have no row.
    """

    def __init__(self, failures, full_results, summary_results):
        self.failures = failures
        self.full_results = full_results
        self.summary_results = summary_results
        super().__init__(
            "{} simulation(s) failed: {}".format(
                len(failures),
                "; ".join(
                    "{} [{}] {}: {}".format(f["sim_id"], f["stage"], f["exception_type"], f["message"])
                    for f in failures
                ),
            )
        )


def _failure_record(sim_id, stage, exc):
    """Describe an exception caught while handling one sim and log it at WARNING."""
    record = {
        "sim_id": sim_id,
        "stage": stage,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }
    logger.warning("Sim {} failed at stage '{}': {}: {}".format(sim_id, stage, record["exception_type"], exc))
    return record


@timing
def run_simulations(sims, save_dir,
                    save_results=True,
                    compute_summary_metrics=True,
                    num_procs=1, name=''):
    """
    Run the simulations passed as argument and optionally process, save, or plot the results.

    Every sim is handled on its own: a failure in one is recorded and the rest still run.
    After all sims have finished and every child process is joined, a SimulationRunError
    is raised if any sim failed. A failed sim has no summary row. The summary CSV is
    written only when no sim failed, so an incomplete one is never left on disk; the
    exception carries the summary instead.

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

    Raises
    ------
    SimulationRunError
        If at least one sim failed, after all sims have finished.
    """
    # Provenance logging must not crash a run: git describe fails (exit 128) when
    # the cwd is not a git repo -- e.g. a packaged/non-editable install run from a
    # bundle dir. Degrade to "unknown" instead of raising.
    try:
        current_commit = subprocess.check_output(
            ["git", "describe", "--always"], stderr=subprocess.DEVNULL
        ).strip().decode("utf-8")
    except (subprocess.CalledProcessError, FileNotFoundError):
        current_commit = "unknown"

    logger.debug("Results Directory: {}".format(save_dir))
    logger.debug("Current Code Commit: {}".format(current_commit))

    num_sims = len(sims)
    sim_ctr = 1
    running_sims = {}
    run_start_time = time.time()

    full_results = dict()
    summary_results = []
    failures = []

    # Process sims in batches of num_procs
    for sim_id, sim in sims.items():
        logger.debug("Running: {}. {} of {}".format(sim_id, sim_ctr, num_sims))
        try:
            sim.start()
            running_sims[sim_id] = sim
        except Exception as e:
            failures.append(_failure_record(sim_id, "run", e))

        batch_start_time = time.time()
        if running_sims and (len(running_sims) >= num_procs or sim_ctr >= num_sims):  # Batch condition

            # Gather results from sim queues
            batch_results = {}
            for id, sim in running_sims.items():
                try:
                    batch_results[id] = sim.queue.get()
                except Exception as e:
                    failures.append(_failure_record(id, "run", e))

            for id, sim in running_sims.items():
                try:
                    sim.join()
                    sim.close()
                except Exception as e:
                    failures.append(_failure_record(id, "run", e))

            logger.debug("Batch run time: {:.2f}m".format((time.time() - batch_start_time) / 60.0))
            logger.debug("Total run time: {:.2f}m".format((time.time() - run_start_time) / 60.0))

            # Each sim is saved, summarized, and recorded on its own; it is only
            # added to the results once every step has succeeded.
            for id, results_df in batch_results.items():
                sim = running_sims[id]
                stage = "save"
                try:
                    # Save stateless info
                    if save_results:
                        with open(os.path.join(save_dir, "{}.json".format(id)), "w") as info_file:
                            json.dump(sim.get_info_stateless(), info_file, indent=4)

                    stage = "metrics"
                    sim_summary = None
                    metrics_df = results_df[results_df["active"] == 1]
                    if compute_summary_metrics:
                        true_bg_trace_clipped = np.array([min(401, max(1, val)) for val in metrics_df['bg']])
                        lbgi, hbgi, brgi = blood_glucose_risk_index(true_bg_trace_clipped)
                        dka_index_value = dka_index(metrics_df["iob"], metrics_df["sbr"].values[0])
                        basal_delivered = metrics_df["delivered_basal_insulin"].sum()
                        bolus_delivered = metrics_df["reported_bolus"].sum()
                        total_delivered = basal_delivered + bolus_delivered
                        summary_str = "Sim {}. \n\tMean BG: {} LBGI: {} HBGI: {} BRGI: {}\n\t Basal {}. Bolus {}. Total {}".format(id, np.mean(true_bg_trace_clipped), lbgi, hbgi, brgi, basal_delivered, bolus_delivered, total_delivered)
                        logger.debug(summary_str)

                        sensor_mard = np.mean(np.abs(metrics_df["bg"] - metrics_df["bg_sensor"]) / metrics_df["bg"])
                        sensor_mbe = np.mean(metrics_df["bg_sensor"] - metrics_df["bg"])
                        logger.debug("Sensor Stats: MBE: {}. MARD: {}".format(sensor_mbe, sensor_mard))

                        sim_summary = {
                            "sim_id": id,
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
                        }

                    # Sanity debugging random stream sync
                    logger.debug("Final Random Int: {}".format(results_df.iloc[-1]["randint"]))

                    stage = "save"
                    if save_results:
                        save_df(results_df, id, save_dir, format_for_file=True)
                except Exception as e:
                    failures.append(_failure_record(id, stage, e))
                    continue

                if sim_summary is not None:
                    summary_results.append(sim_summary)
                full_results[id] = results_df

            running_sims = {}  # reset for next batch

        sim_ctr += 1

    logger.debug("Full run time: {:.2f}m".format((time.time() - run_start_time) / 60.0))

    summary_results_df = pd.DataFrame(summary_results, columns=None if summary_results else ["sim_id"])
    summary_results_df.set_index("sim_id", inplace=True)

    if failures:
        raise SimulationRunError(failures, full_results, summary_results_df)

    if save_results:
        # summary_results_df.to_csv(os.path.join(save_dir, "summary_results_{}.csv".format(time.time())))
        summary_results_df.to_csv(os.path.join(save_dir, f"summary_results_{name}.csv"))

    return full_results, summary_results_df
