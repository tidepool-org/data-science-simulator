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
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.sensor_icgm import (
    SensoriCGM, SensoriCGMInitialOffset, CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, SensoriCGMModelOverlayV1,
)

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace
from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


def generate_icgm_initial_positive_bias_simulations(json_sim_base_config):
    """
    Generator simulations from a base configuration that have different true bg
    starting points and different t0 sensor error values.
    """
    num_history_values = len(json_sim_base_config["patient"]["sensor"]["glucose_history"]["value"])

    # true_glucose_start_values = range(40, 400, 5)
    true_glucose_start_values = [250]  # testing

    for true_start_glucose in true_glucose_start_values:

        new_sim_base_config = copy.deepcopy(json_sim_base_config)

        new_sim_base_config["controller"]["settings"]["max_physiologic_slope"] = 4  # add in velocity cap
        # new_sim_base_config["controller"]["settings"]["max_basal_rate"] = 1e6  # testing diff between basal/bolus rec

        glucose_history_values = {i: true_start_glucose for i in range(num_history_values)}

        new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
        new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values

        date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
        glucose_datetimes = [datetime.datetime.strptime(dt_str, date_str_format)
                             for dt_str in new_sim_base_config["patient"]["sensor"]["glucose_history"]["datetime"].values()]
        t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

        sampled_error_values = sample_uniformly_positive_error_cgm_ranges(true_start_glucose)
        sampled_error_values = [350]
        for initial_error_value in sampled_error_values:

            sim_parser = ScenarioParserV2()
            random_state = RandomState(1234)

            sensor = get_initial_offset_sensor(t0, random_state=random_state, initial_error_value=initial_error_value)
            # Update state through time until t0 according to behavior model
            for dt, true_bg in zip(glucose_datetimes, glucose_history_values.values()):
                sensor.update(dt, patient_true_bg=true_bg)

            sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(new_sim_base_config, sensor=sensor)

            virtual_patient.sensor = sensor

            def does_accept_bolus_recommendation(self, bolus):
                return self.time == t0
            virtual_patient.does_accept_bolus_recommendation = types.MethodType(does_accept_bolus_recommendation, virtual_patient)

            sim = Simulation(sim_start_time,
                             duration_hrs=duration_hrs,
                             virtual_patient=virtual_patient,
                             controller=controller,
                             multiprocess=True,
                             sim_id="icgm_positive_bias_tbg={}_sbg={}".format(true_start_glucose, initial_error_value)
                             )

            sim.random_state = random_state
            yield sim


def get_ideal_sensor(t0, sim_parser):

    ideal_sensor_config = SensorConfig(sensor_bg_history=sim_parser.patient_glucose_history)
    sensor = IdealSensor(time=t0, sensor_config=ideal_sensor_config)
    return sensor


def get_initial_offset_sensor(t0, random_state, initial_error_value):
    """
    Get iCGM sensor that has a manually specified error at t0 of simulation.
    """

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.history_window_hrs = 24 * 10

    sensor_config.behavior_models = [
        SensoriCGMModelOverlayV1(bias=0, sigma=4, delay=0, spurious_value_prob=0.0, num_consecutive_spurious=1),
    ]

    sensor_config.sensor_range = range(40, 401)
    sensor_config.special_controls = iCGM_THRESHOLDS
    sensor_config.initial_controls = CLEAN_INITIAL_CONTROLS
    sensor_config.do_look_ahead = True
    sensor_config.look_ahead_min_prob = 0.7

    num_history_values = 24 * 12  # 24 hours
    sensor = SensoriCGMInitialOffset(
                        time=t0 - datetime.timedelta(minutes=num_history_values * 5.0),
                        t0_error_bg=initial_error_value,
                        sensor_config=sensor_config,
                        random_state=random_state,
                        sim_start_time=t0)
    sensor.name = "iCGM_{}".format(initial_error_value)

    return sensor


def sample_uniformly_positive_error_cgm_ranges(value, num_samples=10):

    if value < 40:
        icgm_low, icgm_high = (40, 80)
    elif 40 <= value <= 60:
        icgm_low, icgm_high = (40, 120)
    elif 60 < value <= 80:
        icgm_low, icgm_high = (61, 120)
    elif 80 < value <= 120:
        icgm_low, icgm_high = (81, 200)
    elif 120 < value <= 160:
        icgm_low, icgm_high = (121, 250)
    elif 160 < value <= 200:
        icgm_low, icgm_high = (161, 300)
    elif 200 < value <= 250:
        icgm_low, icgm_high = (201, 350)
    elif 250 < value <= 300:
        icgm_low, icgm_high = (251, 400)
    elif 300 < value <= 350:
        icgm_low, icgm_high = (301, 400)
    elif 350 < value <= 400:
        icgm_low, icgm_high = (351, 400)
    elif value > 400:
        icgm_low, icgm_high = (351, 401)

    # step = (icgm_high - icgm_low) / (num_samples)

    step = 5.0

    sampled_errors = [round(v) for v in np.arange(icgm_low, icgm_high + step, step)]

    return sampled_errors


def build_icgm_sim_generator(json_base_configs, sim_batch_size=30):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    """

    sim_ctr = 0
    sims = {}

    for json_config in json_base_configs:
        # logger.info("VP: {}".format(vp_idx))

        for sim in generate_icgm_initial_positive_bias_simulations(json_config):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

    yield sims


if __name__ == "__main__":

    # TODO: Big run with many sims

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    result_dir = os.path.join(DATA_DIR, "processed/icgm-sensitivity-analysis-results-" + today_timestamp)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info("Made director for results: {}".format(result_dir))

    if 1:
        sim_batch_size = 1

        json_base_configs = transform_icgm_json_to_v2_parser()
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

            plot_sim_results(full_results)

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

    # Fit the requirements
    # requirements_model = PositiveBiasiCGMRequirements()
    # requirements_model.fit_positive_bias_prob(summary_df_positive_bias_sims)
    # requirements_model.fit_positive_bias_range_and_prob(summary_df_positive_bias_sims)

    # To check before running
    # 1. Sensor behavior model & properties
    # 2. Future tbg prob