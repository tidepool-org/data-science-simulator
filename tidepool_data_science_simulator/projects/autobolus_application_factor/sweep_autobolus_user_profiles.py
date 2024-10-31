__author__ = "Mark Connolly"

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
    NoisySensorInitialOffset, SensoriCGMInitialOffset, CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, SensoriCGMModelOverlayV1,
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import Bolus, Carb, TargetRange

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace
from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


def generate_autobolus_simulations(json_sim_base_config, base_sim_seed):
    """
    Generator simulations from a base configuration that have different true bg
    starting points and different t0 sensor error values.
    """
    num_history_values = len(json_sim_base_config["patient"]["sensor"]["glucose_history"]["value"])

    partial_application_factors = np.linspace(0, 10, 11)/10
    # partial_application_factors = [1]

    true_start_glucose = 190 

    random_state = RandomState(base_sim_seed)

    for paf in partial_application_factors:
        
        new_sim_base_config = copy.deepcopy(json_sim_base_config)    

        date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
        t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

        glucose_history_values = {i: true_start_glucose for i in range(num_history_values)}

        new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
        new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values        

        true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])
        reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])            
        
        sim_parser = ScenarioParserV2()
        sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(new_sim_base_config)
        
        virtual_patient.carb_event_timeline = true_carb_timeline
        virtual_patient.pump.carb_event_timeline = reported_carb_timeline
        
        controller.controller_config.controller_settings['minimum_autobolus'] = 0.1
        controller.controller_config.controller_settings['maximum_autobolus'] = 10
        controller.controller_config.controller_settings['partial_application_factor'] = paf

        patient_id = new_sim_base_config['patient_id']
        sim_id="patient_id={}_paf={}".format(patient_id, paf)
        
        sim = Simulation(sim_start_time,
                            duration_hrs=duration_hrs,
                            virtual_patient=virtual_patient,
                            controller=controller,
                            multiprocess=True,
                            sim_id=sim_id
                            )

        sim.random_state = random_state

        yield sim


def get_ideal_sensor(t0, sim_parser):

    ideal_sensor_config = SensorConfig(sensor_bg_history=sim_parser.patient_model_glucose_history)
    sensor = IdealSensor(time=t0, sensor_config=ideal_sensor_config)
    return sensor


def get_initial_offset_sensor_noisy(t0_init, t0, random_state, initial_error_value):

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.std_dev = 3.0

    sensor = NoisySensorInitialOffset(
        time=t0_init,
        t0_error_bg=initial_error_value,
        sensor_config=sensor_config,
        random_state=random_state,
        sim_start_time=t0)
    sensor.name = "NoisySensor_{}".format(initial_error_value)

    return sensor


def get_initial_offset_sensor(t0_init, t0, random_state, initial_error_value):
    """
    Get iCGM sensor that has a manually specified error at t0 of simulation.
    """

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.history_window_hrs = 24 * 10

    sensor_config.behavior_models = [
        SensoriCGMModelOverlayV1(bias=0, sigma=2, delay=0, spurious_value_prob=0.0, num_consecutive_spurious=1),
    ]

    sensor_config.sensor_range = range(40, 401)
    sensor_config.special_controls = iCGM_THRESHOLDS
    sensor_config.initial_controls = CLEAN_INITIAL_CONTROLS
    sensor_config.do_look_ahead = True
    sensor_config.look_ahead_min_prob = 0.7

    sensor = SensoriCGMInitialOffset(
                        time=t0_init,
                        t0_error_bg=initial_error_value,
                        sensor_config=sensor_config,
                        random_state=random_state,
                        sim_start_time=t0)
    sensor.name = "iCGM_{}".format(initial_error_value)

    return sensor


def build_autobolus_sim_generator(json_base_configs, sim_batch_size=30):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    """
    for i, json_config in enumerate(json_base_configs, 1):
        logger.info("VP: {}. {} of {}".format(json_config["patient_id"], i, len(json_base_configs)))

        sim_ctr = 0
        sims = {}

        for sim in generate_autobolus_simulations(json_config, base_sim_seed=i):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

        if sims:
            yield sims


if __name__ == "__main__":

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    result_dir = os.path.join(DATA_DIR, "processed/autobolus-analysis-results-" + today_timestamp)

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info("Made director for results: {}".format(result_dir))

    if 1:
        sim_batch_size = 1

        json_base_configs = transform_icgm_json_to_v2_parser()
        sim_batch_generator = build_autobolus_sim_generator(json_base_configs, sim_batch_size=sim_batch_size)

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

            # plot_sim_results(full_results)

            logger.info("Batch {}".format(i))
            logger.info("Minutes to build sim batch {} of {} sensors. Total minutes {}".format(batch_total_time, len(sim_batch), run_total_time))
