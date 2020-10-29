
import os
import re
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tidepool_data_science_simulator.models.sensor import NoisySensor
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.evaluation.inspect_results import load_results
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_patient import (
    get_canonical_risk_pump_config,
    get_canonical_virtual_patient_model_config,
    get_canonical_sensor_config
)

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.run import run_simulations


def build_metabolic_sensitivity_sims():
    """
    Look at resulting bgs from settings that are correct/incorrect for analysis.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file

    param_grid: list of dicts
        Parameters to vary
    """
    t0, patient_config = get_canonical_virtual_patient_model_config()
    t0, pump_config = get_canonical_risk_pump_config(t0)
    t0, sensor_config = get_canonical_sensor_config(t0)

    patient_param_grid = [
        # {"patient_change": -0.1},
        {
            "patient_change": 0.0,
            "recommendation_accept_prob": accept_prob
        }
        # {"patient_change": 0.1},
        for accept_prob in [1.0]
    ]

    pump_param_grid = [
        {
            "br_change": 0,
            "isf_change": 0,
            "cir_change": 0
        },
        {
            "br_change": 0.05,
            "isf_change": 0,
            "cir_change": 0
         },
        {
            "br_change": 0,
            "isf_change": -0.05,
            "cir_change": 0
        },
        {
            "br_change": 0,
            "isf_change": 0,
            "cir_change": -0.05
        },
    ]

    pump_param_grid = [
        {
            "br_change": br_change,
            "isf_change": isf_change,
            "cir_change": 0
        }
        for br_change in [0.0]
        for isf_change in [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ]

    sims = {}
    for patient_pgrid in patient_param_grid:
        for pump_pgrid in pump_param_grid:
            patient_config.basal_schedule.set_override(patient_pgrid["patient_change"])
            patient_config.insulin_sensitivity_schedule.set_override(patient_pgrid["patient_change"])
            patient_config.carb_ratio_schedule.set_override(patient_pgrid["patient_change"])
            patient_config.recommendation_accept_prob = patient_pgrid["recommendation_accept_prob"]

            pump_config.basal_schedule.set_override(pump_pgrid["br_change"])
            pump_config.insulin_sensitivity_schedule.set_override(pump_pgrid["isf_change"])
            pump_config.carb_ratio_schedule.set_override(pump_pgrid["cir_change"])

            sensor_config.std_dev = 1.0

            t0, sim = get_canonical_simulation(
                patient_config=patient_config,
                patient_class=VirtualPatientModel,
                sensor_config=sensor_config,
                sensor_class=NoisySensor,
                pump_config=pump_config,
                pump_class=ContinuousInsulinPump,
                controller_class=LoopController,
                multiprocess=True,
                duration_hrs=48,
            )

            param_dict = pump_pgrid.copy()
            param_dict.update({"patient_change": patient_pgrid["patient_change"]})
            sim_id = str(param_dict)
            sims[sim_id] = sim

            patient_config.basal_schedule.unset_override()
            patient_config.insulin_sensitivity_schedule.unset_override()
            patient_config.carb_ratio_schedule.unset_override()

            pump_config.basal_schedule.unset_override()
            pump_config.insulin_sensitivity_schedule.unset_override()
            pump_config.carb_ratio_schedule.unset_override()

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

    sims = build_metabolic_sensitivity_sims()

    save_dir = "../../data/results/simulations/sensitivity_analysis_testing"
    all_results = run_simulations(sims,
                    save_dir=save_dir,
                    save_results=True,
                    num_procs=100)

    # all_results = load_results(save_dir)

    plot_sim_results(all_results, n_sims_max_legend=10)
    # plot_insulin_changes(all_results)

