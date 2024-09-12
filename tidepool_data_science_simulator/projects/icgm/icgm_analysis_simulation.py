__author__ = "Mark Connolly"

import logging
logger = logging.getLogger(__name__)

import time
import os
import datetime
import argparse

from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

import types
from numpy.random import RandomState

import time
import os
import copy
import numexpr 

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.models.sensor_icgm import (
    NoisySensorInitialOffset, SensoriCGMInitialOffset, CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, SensoriCGMModelOverlayV1,
)

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
    # error_glucose_values = [90]

    random_state = RandomState(base_sim_seed)

    for true_start_glucose in true_glucose_start_values:
        for initial_error_value in error_glucose_values:

            new_sim_base_config = copy.deepcopy(json_sim_base_config)
            
            new_sim_base_config["controller"]["settings"]["max_physiologic_slope"] = 4  # add in velocity cap
            glucose_history_values = {i: true_start_glucose for i in range(num_history_values)}

            new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
            new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values

            date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
            glucose_datetimes = [datetime.datetime.strptime(dt_str, date_str_format)
                                    for dt_str in
                                    new_sim_base_config["patient"]["sensor"]["glucose_history"]["datetime"].values()]
            t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

            sim_parser = ScenarioParserV2()
            
            # sim_id = "icgm_analysis_coastal_vp_{}_{}_tbg={}_sbg=IDEAL".format(base_sim_seed, new_sim_base_config["patient_id"], true_start_glucose)
            # sensor = get_ideal_sensor(t0=t0, sim_parser=sim_parser)

            sim_id = "icgm_analysis_coastal_vp_{}_{}_tbg={}_sbg={}".format(base_sim_seed, new_sim_base_config["patient_id"], true_start_glucose, initial_error_value)
            sensor = get_initial_offset_sensor_noisy(t0_init=t0 - datetime.timedelta(minutes=len(glucose_history_values) * 5.0),
                                               t0=t0,
                                               random_state=random_state,
                                               initial_error_value=initial_error_value)
            
            # Update state through time until t0 according to behavior model
            for dt, true_bg in zip(glucose_datetimes, glucose_history_values.values()):
                sensor.update(dt, patient_true_bg=true_bg, patient_true_bg_prediction=[])

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
                                sim_id=sim_id
                                )

            sim.random_state = random_state

            yield sim


# def get_ideal_sensor(t0, sim_parser):

#     ideal_sensor_config = SensorConfig(sensor_bg_history=sim_parser.patient_glucose_history)
#     sensor = IdealSensor(time=t0, sensor_config=ideal_sensor_config)
#     return sensor

def get_ideal_sensor(t0, sim_parser):

    ideal_sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
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


def build_icgm_sim_generator(json_base_configs, sim_batch_size=30):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    """
    for i, json_config in enumerate(json_base_configs, 1):
        logger.info("VP: {}. {} of {}".format(json_config["patient_id"], i, len(json_base_configs)))

        sim_ctr = 0
        sims = {}

        for sim in generate_icgm_point_error_simulations(json_config,base_sim_seed=i):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

        yield sims


if __name__ == "__main__":
    parser = argparse.ArgumentParser("icgm_analysis_simulation")
    parser.add_argument("sim_batch_size", help="Number of simulations to run per batch. Should be less than the number of cores", type=int)
    args = parser.parse_args()

    sim_batch_size = args.sim_batch_size
    os.environ['NUMEXPR_MAX_THREADS'] = str(sim_batch_size)
    numexpr.set_num_threads(sim_batch_size)
    
    
    result_dir = os.path.join(DATA_DIR, "processed/icgm-sensitivity-analysis-results-COASTAL")
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info("Made director for results: {}".format(result_dir))

    
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

