import sys
sys.path.append('..')
sys.path.append('../../')
sys.path.append('../../..')

import copy
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
    t0, patient_config = get_canonical_virtual_patient_model_config() # patient has many attributes e.g. starting glucose (default: 110), recommendatio accept probability, etc.    
    
    t0, sensor_config = get_canonical_sensor_config(t0) # sensor config has a blood glucose history. right now looks like the starting value repeated 'n' times every 5 minutes before t0
    

    t0, pump_config = get_canonical_risk_pump_config(t0) # sets a carb timeline, bolus timeline (both initialized with 0 at t=0, assuming new values are then added on), basal schedule (e.g. 0.3 units delivered for 24 hours), carb ratio schedule (e.g. constant carb ratio of 20 for 24 hours). similarly for ISR and target range. ques: is there a schedule different from 24 hours?

    patient_config.recommendation_accept_prob = 1.0  # Accept the bolus


    sims = {}
    # for exercise_preset_p in np.arange(0.1, 1.09, 0.1):
    for exercise_preset_p in [0.7, 1.0]:

        basal_p_factor = -1 + exercise_preset_p  # note: funky math because of how override function works
        pump_config.basal_schedule.set_override(basal_p_factor)

        isf_p_factor = -1 + 1 / (exercise_preset_p)  # note: funky math because of how override function works
        pump_config.insulin_sensitivity_schedule.set_override(isf_p_factor)

        cir_p_factor = -1 + 1 / (exercise_preset_p)  # note: funky math because of how override function works
        pump_config.carb_ratio_schedule.set_override(cir_p_factor)

        sensor_config.std_dev = 1.0 # ques: what does this do

        t0, sim = get_canonical_simulation(
            patient_config=patient_config,
            patient_class=VirtualPatientModel,
            sensor_config=sensor_config,
            sensor_class=NoisySensor,
            pump_config=pump_config,
            pump_class=ContinuousInsulinPump,
            controller_class=LoopController,
            multiprocess=True,
            duration_hrs=1,
        )

        sim_id = f"Preset_{exercise_preset_p}"
        sims[sim_id] = sim

        pump_config.basal_schedule.unset_override()
        pump_config.insulin_sensitivity_schedule.unset_override()
        pump_config.carb_ratio_schedule.unset_override()

    return sims


def plot_insulin_changes(all_results, save_dir):

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
    plt.savefig(save_dir)
    plt.show()
    


if __name__ == "__main__":

    sims = build_metabolic_sensitivity_sims()

    save_dir = "./simulation_results/"
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        
    all_results, summary_results_df = run_simulations(sims,
                    save_dir=save_dir,
                    save_results=True,
                    num_procs=1)

    for sim_id, results_df in all_results.items():
        bolus_insulin_total = results_df["true_bolus"].sum()
        basal_insulin_total = results_df["delivered_basal_insulin"].sum()
        print(sim_id, bolus_insulin_total, basal_insulin_total, bolus_insulin_total + basal_insulin_total)

    all_results = load_results(save_dir)

    plot_sim_results(all_results, n_sims_max_legend=10, save_path=save_dir)
    plot_insulin_changes(all_results, save_dir)
