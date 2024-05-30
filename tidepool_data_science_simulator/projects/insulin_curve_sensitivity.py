__author__ = "Cameron Summers"


import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tidepool_data_science_simulator.models.sensor import NoisySensor
from tidepool_data_science_simulator.models.patient import VirtualPatientModel, VirtualPatientCarbBolusAccept
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.evaluation.inspect_results import load_results
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_patient import (
    get_canonical_risk_pump_config,
    get_canonical_virtual_patient_model_config,
    get_canonical_sensor_config,
)
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config


from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.run import run_simulations


def build_insulin_curve_sensitivity_sims():
    """
    Look at resulting bgs from settings that are correct/incorrect for analysis.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file

    param_grid: list of dicts
        Parameters to vary
    """
    bg_values_history = [200] * 137
    t0, patient_config = get_canonical_virtual_patient_model_config()
    t0, pump_config = get_canonical_risk_pump_config(t0)
    t0, sensor_config = get_canonical_sensor_config(t0)

    t0, controller_config = get_canonical_controller_config()

    insulin_curve_param_grid = [
        {
            "insulin_delay": insulin_delay,
            "model": insulin_model,
        }
        for insulin_delay in [0, 10, 20, 30, 40, 50, 60]
        # for insulin_delay in [10]
        # for insulin_model in [[360.0, 55], [360.0, 65], [360.0, 75]]
        for insulin_model in [[360.0, 55]]
    ]

    sims = {}
    for pgrid in insulin_curve_param_grid:

        controller_config.controller_settings["insulin_delay"] = pgrid["insulin_delay"]
        controller_config.controller_settings["model"] = pgrid["model"]

        t0, sim = get_canonical_simulation(
            patient_config=patient_config,
            patient_class=VirtualPatientCarbBolusAccept,
            sensor_config=sensor_config,
            sensor_class=NoisySensor,
            pump_config=pump_config,
            pump_class=ContinuousInsulinPump,
            controller_class=LoopController,
            controller_config=controller_config,
            multiprocess=True,
            duration_hrs=8,
        )

        sim_id = pgrid.__str__()
        sims[sim_id] = sim

    return sims


def plot_insulin_changes(all_results):

    br_change = []
    basal = []
    bolus = []
    total_insulin = []

    cgm_mean = []

    fig, ax = plt.subplots(2, 1, sharex=True)
    for sim_id, results_df in all_results.items():
        total_basal_delivered = results_df["delivered_basal_insulin"].sum()
        total_bolus_delivered = results_df["reported_bolus"].sum()

        basal_change = float(re.search("br_change.*(\d+\.\d+).*isf_change", sim_id).groups()[0])
        br_change.append(basal_change)
        basal.append(total_basal_delivered)
        bolus.append(total_bolus_delivered)
        total_insulin.append(total_basal_delivered + total_bolus_delivered)

        cgm_mean.append(results_df["bg_sensor"].mean())

    ax[0].plot(br_change, basal, label="basal")
    ax[0].plot(br_change, bolus, label="bolus")
    ax[0].plot(br_change, total_insulin, label="total")
    plt.legend()
    ax[1].plot(br_change, cgm_mean)
    plt.show()


if __name__ == "__main__":

    sims = build_insulin_curve_sensitivity_sims()

    save_dir = "../../data/results/simulations/insulin_curve_sensitivity_analysis"
    all_results = run_simulations(sims,
                    save_dir=save_dir,
                    save_results=True,
                    # save_results=False,
                    num_procs=8)

    # all_results = load_results(save_dir)

    plot_sim_results(all_results, n_sims_max_legend=10)
    # plot_insulin_changes(all_results)

