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
from tidepool_data_science_simulator.models.patient import VirtualPatientModel, MealModel
from tidepool_data_science_simulator.models.controller import LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.measures import Carb, Bolus
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.sensor_icgm import (
    NoisySensorInitialOffset, SensoriCGMInitialOffset, CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, SensoriCGMModelOverlayV1,
)

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace
from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel, CesconCarbModel


def get_new_random_state(seed=1234567):
    return RandomState(seed)


def generate_double_low_simulations(json_sim_base_config, base_sim_seed):
    """
    Generator simulations from a base configuration that have different true bg
    starting points and different t0 sensor error values.
    """
    patient_random_state = get_new_random_state()  # Single instance generates different patients
    sims = {}

    true_start_glucose = 120
    for vp_idx, icgm_patient_obj in enumerate(json_sim_base_config):

        num_history_values = len(json_sim_base_config["patient"]["sensor"]["glucose_history"]["value"])

        new_sim_base_config = copy.deepcopy(json_sim_base_config)


        glucose_history_values = {i: true_start_glucose for i in range(num_history_values)}
        new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
        new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values

        date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
        glucose_datetimes = [datetime.datetime.strptime(dt_str, date_str_format)
                             for dt_str in
                             new_sim_base_config["patient"]["sensor"]["glucose_history"]["datetime"].values()]
        t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

        # Setup patient config
        sim_parser = ScenarioParserV2()
        sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(
            new_sim_base_config)

        sensor = get_sensor_noisy(
            t0_init=t0 - datetime.timedelta(minutes=len(glucose_history_values) * 5.0),
            random_state=patient_random_state)

        # Update state through time until t0 according to behavior model
        for dt, true_bg in zip(glucose_datetimes, glucose_history_values.values()):
            sensor.update(dt, patient_true_bg=true_bg, patient_true_bg_prediction=[])


        # def does_accept_bolus_recommendation(self, bolus):
        #     return self.time == t0
        #
        # virtual_patient.does_accept_bolus_recommendation = types.MethodType(does_accept_bolus_recommendation,
        #                                                                     virtual_patient)

        patient_config = sim_parser.get_patient_config()

        patient_config.recommendation_accept_prob = 0.8  # patient_random_state.uniform(0.8, 0.99)
        patient_config.min_bolus_rec_threshold = patient_random_state.uniform(0.4, 0.6)
        patient_config.correct_bolus_bg_threshold = patient_random_state.uniform(140, 190)  # no impact
        patient_config.correct_bolus_delay_minutes = patient_random_state.uniform(20, 40)  # no impact
        patient_config.correct_carb_bg_threshold = patient_random_state.uniform(70, 90)
        patient_config.correct_carb_delay_minutes = patient_random_state.uniform(5, 15)
        patient_config.carb_count_noise_percentage = patient_random_state.uniform(0.2, 0.25)
        patient_config.report_bolus_probability = patient_random_state.uniform(1.0, 1.0)  # no impact
        patient_config.report_carb_probability = patient_random_state.uniform(0.95, 1.0)

        patient_config.recommendation_meal_attention_time_minutes = 30

        patient_config.prebolus_minutes_choices = [0]
        patient_config.carb_reported_minutes_choices = [0]

        pump_config = sim_parser.get_pump_config()

        pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

        pump.pump_config.basal_schedule.set_override(0.2)
        pump.pump_config.insulin_sensitivity_schedule.set_override(-0.2)
        pump.pump_config.carb_ratio_schedule.set_override(-0.2)

        vp = VirtualPatientModel(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
            random_state=patient_random_state,
            id=vp_idx
        )
        # vp.meal_model = [
        #     MealModel("Lunch", datetime.time(hour=11), datetime.time(hour=13), 1.0),
        # ]

        new_sim_base_config["controller"]["settings"]["max_physiologic_slope"] = 4  # add in velocity cap
        new_sim_base_config["controller"]["settings"]["model"] = "rapid_acting_adult"
        # new_sim_base_config["controller"]["settings"]["retrospective_correction_enabled"] = False
        # new_sim_base_config["controller"]["settings"]["max_basal_rate"] = 4

        controller = sim_parser.get_controller(t0, new_sim_base_config)

        # Setup Sims
        sim_id = "VP".format(vp_idx)
        sim = Simulation(
            time=t0,
            duration_hrs=72,
            virtual_patient=vp,
            sim_id=sim_id,
            controller=controller,
            multiprocess=True,
            random_state=patient_random_state
        )
        sims[sim_id] = sim

    yield sim


def get_sensor_noisy(t0_init, random_state):

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.std_dev = 3.0

    sensor = NoisySensor(
        time=t0_init,
        sensor_config=sensor_config,
        random_state=random_state)
    sensor.name = "NoisySensor"

    return sensor


def build_icgm_sim_generator(json_base_configs, sim_batch_size=30):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    """
    for i, json_config in enumerate(json_base_configs, 1):
        if i != 2:
            continue

        logger.info("VP: {}. {} of {}".format(json_config["patient_id"], i, len(json_base_configs)))

        sim_ctr = 0
        sims = {}

        for sim in generate_double_low_simulations(json_config, base_sim_seed=i):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

        yield sims


if __name__ == "__main__":

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    result_dir = os.path.join(DATA_DIR, "processed/double-low-analysis-results-" + today_timestamp)

    # for tau in range(1, 150, 1):
    #     carb_model = CesconCarbModel(isf=100, cir=10, tau=tau, theta=20)
    #     t_min, bg_delta, bg = carb_model.run(10, 10, five_min=True)
    #     # plt.plot(bg)
    #     total_delta = 0
    #     expected_delta = 100 / 10 * 10
    #     for i in range(len(bg_delta)):
    #         total_delta += bg_delta[i]
    #         if total_delta >= 0.99 * expected_delta:
    #             print("{}: {},".format(i*5, tau))
    #             break
    # # plt.show()


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

