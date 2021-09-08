__author__ = "Cameron Summers"

import types

import logging
logger = logging.getLogger(__name__)

import time
import os
import datetime
import json
import copy
import re
from collections import defaultdict

import numpy as np
from numpy.random import RandomState
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.controller import LoopControllerDisconnectorOverrides
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.sensor_icgm import (
    NoisySensorInitialOffset, SensoriCGMInitialOffset, CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, SensoriCGMModelOverlayV1,
    NoisySensorOverrides
)

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace
from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


def generate_icgm_point_error_simulations(json_sim_base_config, base_sim_seed):
    """
    Generator simulations from a base configuration that have different true bg
    starting points and different t0 sensor error values.
    """
    num_history_values = len(json_sim_base_config["patient"]["sensor"]["glucose_history"]["value"])

    true_glucose_start_values = range(40, 405, 5)
    error_glucose_values = [v for v in true_glucose_start_values[::-1]]

    # true_glucose_start_values = [90]  # testing
    # error_glucose_values = [300]

    random_state = RandomState(base_sim_seed)

    for true_start_glucose in true_glucose_start_values:
        for initial_error_value in error_glucose_values:

            new_sim_base_config = copy.deepcopy(json_sim_base_config)

            new_sim_base_config["controller"]["settings"]["max_physiologic_slope"] = 4  # add in velocity cap

            # ======== Max Basal Rate Guardrail Logic ========
            # new_sim_base_config["controller"]["settings"]["max_basal_rate"] = 1e6  # testing diff between basal/bolus rec

            cir = new_sim_base_config["patient"]["patient_model"]["metabolism_settings"]["carb_insulin_ratio"]["values"][0]
            basal = new_sim_base_config["patient"]["patient_model"]["metabolism_settings"]["basal_rate"]["values"][0]
            max_basal_guardrail = max(basal, 70/cir)
            new_sim_base_config["controller"]["settings"]["max_basal_rate"] = max_basal_guardrail
            logger.info("Max basal guardrail: {}. Basal {}. CIR {}. Mult: {}".format(max_basal_guardrail, basal, cir, max_basal_guardrail/basal))
            # ===========================

            glucose_history_values = {i: true_start_glucose for i in range(num_history_values)}

            new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
            new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values

            date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
            glucose_datetimes = [datetime.datetime.strptime(dt_str, date_str_format)
                                 for dt_str in
                                 new_sim_base_config["patient"]["sensor"]["glucose_history"]["datetime"].values()]
            t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

            sim_parser = ScenarioParserV2()

            sensor = get_sensor_with_overrides(t0_init=t0 - datetime.timedelta(minutes=len(glucose_history_values) * 5.0),
                                               random_state=random_state,
                                               overrides_datetime_value_dict={
                                                   t0 - datetime.timedelta(minutes=0): initial_error_value
                                                   # t0 - datetime.timedelta(minutes=15) + datetime.timedelta(minutes=5 * i): 200# + 25*i
                                                   #  for i in range(1, 5)
                                               })
            # Update state through time until t0 according to behavior model
            for dt, true_bg in zip(glucose_datetimes, glucose_history_values.values()):
                sensor.update(dt, patient_true_bg=true_bg, patient_true_bg_prediction=[])

            sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(new_sim_base_config, sensor=sensor)

            virtual_patient.sensor = sensor

            virtual_patient.patient_config.recommendation_accept_prob = 0.0  # Makes temp basal only
            virtual_patient.patient_config.min_bolus_rec_threshold = 2.0

            sim = Simulation(sim_start_time,
                             duration_hrs=duration_hrs,
                             virtual_patient=virtual_patient,
                             controller=controller,
                             multiprocess=True,
                             sim_id="icgm_jun2021_vp_{}_tbg={}_sbg={}".format(new_sim_base_config["patient_id"], true_start_glucose, initial_error_value)
                             )

            sim.random_state = random_state

            yield sim


def get_sensor_with_overrides(t0_init, random_state, overrides_datetime_value_dict):

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.std_dev = 3.0

    sensor = NoisySensorOverrides(
        time=t0_init,
        sensor_config=sensor_config,
        random_state=random_state,
        overrides_datetime_value_dict=overrides_datetime_value_dict)
    sensor.name = "NoisySensorOverrides"

    return sensor


def build_icgm_sim_generator(json_base_configs, sim_batch_size=30):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    """
    for i, json_config in enumerate(json_base_configs, 1):
        logger.info("VP: {}. {} of {}".format(json_config["patient_id"], i, len(json_base_configs)))

        sim_ctr = 0
        sims = {}

        for sim in generate_icgm_point_error_simulations(json_config, base_sim_seed=i):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

        yield sims


def plot_patient_population(json_base_configs):
    """
    Plot metabolic settings and ages for icgm patients
    """

    isfs = []
    cirs = []
    basals = []
    ages = []
    for config in json_base_configs:
        isf = config["patient"]["patient_model"]["metabolism_settings"]["insulin_sensitivity_factor"]["values"][0]
        cir = config["patient"]["patient_model"]["metabolism_settings"]["carb_insulin_ratio"]["values"][0]
        basal = config["patient"]["patient_model"]["metabolism_settings"]["basal_rate"]["values"][0]
        age = config["patient"]["age"]
        ages.append(age)

        isfs.append(isf)
        cirs.append(cir)
        basals.append(basal)

    fig, ax = plt.subplots(1, 4, figsize=(20, 8))
    fig.suptitle("Patient Population")
    ax[0].scatter(isfs, cirs)
    ax[0].set_xlabel("Insulin Sensitivity Factor (mg/dL/U)")
    ax[0].set_ylabel("Carb Ratio (g/U)")

    ax[1].scatter(cirs, basals)
    ax[1].set_xlabel("Carb Ratio (g/U)")
    ax[1].set_ylabel("Basal Rate (U/hr)")

    ax[2].scatter(isfs, basals)
    ax[2].set_xlabel("Insulin Sensitivity Factor (mg/dL/U)")
    ax[2].set_ylabel("Basal Rate (U/hr)")

    ax[3].hist(ages, bins=20)
    ax[3].set_xlabel("Age (yrs)")
    ax[3].set_ylabel("Count")

    plt.show()


if __name__ == "__main__":

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    result_dir = os.path.join(DATA_DIR, "processed/icgm-sensitivity-analysis-results-" + today_timestamp)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info("Made director for results: {}".format(result_dir))

    if 1:
        sim_batch_size = 1

        json_base_configs = transform_icgm_json_to_v2_parser()

        # plot_patient_population(json_base_configs)
        sim_batch_generator = build_icgm_sim_generator(json_base_configs, sim_batch_size=sim_batch_size)

        start_time = time.time()
        for i, sim_batch in enumerate(sim_batch_generator):

            batch_start_time = time.time()

            full_results, summary_results_df = run_simulations(
                sim_batch,
                save_dir=result_dir,
                save_results=True,
                num_procs=sim_batch_size
            )
            batch_total_time = (time.time() - batch_start_time) / 60
            run_total_time = (time.time() - start_time) / 60
            logger.info("Batch {}".format(i))
            logger.info("Minutes to build sim batch {} of {} sensors. Total minutes {}".format(batch_total_time, len(sim_batch), run_total_time))

            # plot_sim_results(full_results)

            # for sim_id, result_df in summary_results_df.items():
            #     sim_id_icgm = sim_id

    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-02-positive_bias_with_requirements/"
    # result_dir = "./data/processed/icgm/big_run_subset_2020-12-11"

    # result_dir = "./data/processed/icgm/icgm-sensitivity-analysis-results-2020-12-11"  # 200k run on compute-2

    # On compute-1
    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-03/"  # worst case negative bias, 887 sims
    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-04/"  # temp basal case

    # plot_icgm_results(result_dir, sim_inspect_id=None)

    # compute_sim_summary_stats(result_dir)

    # sim_run_887_filename_pos_bias_corr_bolus = "result_summary_positive_bias.csv"
    # sim_run_200k_filename = "result_summary_2020-12-13T05:55:01.004257.csv"
    # summary_df_positive_bias_sims = pd.read_csv(sim_run_200k_filename, sep="\t")

    # Compute the risk table
    # compute_risk_stats(summary_df_positive_bias_sims)
