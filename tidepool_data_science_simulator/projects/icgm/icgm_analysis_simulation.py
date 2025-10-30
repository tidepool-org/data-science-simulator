__author__ = "Mark Connolly"

import logging
import subprocess

import numpy as np

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_metrics.glucose import glucose

logger = logging.getLogger(__name__)

import time
import os
import datetime
import argparse
import math

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
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index

def generate_icgm_point_error_simulations(json_sim_base_config, base_sim_seed, paf, positive_rc, 
                                         true_bg_values=None, sensor_bg_values=None):
    """
    Generator simulations from a base configuration that have different true bg
    starting points and different t0 sensor error values.
    
    Args:
        json_sim_base_config: Base simulation configuration
        base_sim_seed: Random seed
        paf: Partial application factor
        positive_rc: Include positive RC and momentum
        true_bg_values: Optional list/range of true glucose values (default: range(40, 405, 5))
        sensor_bg_values: Optional list/range of sensor glucose values (default: true_bg_values)
    """
    IDEAL = True
    num_history_values = len(json_sim_base_config["patient"]["sensor"]["glucose_history"]["value"])

    # Use provided values or defaults
    if true_bg_values is None:
        true_glucose_start_values = range(40, 405, 5)
    else:
        true_glucose_start_values = true_bg_values
    
    if sensor_bg_values is None:
        error_glucose_values = true_glucose_start_values
    else:
        error_glucose_values = sensor_bg_values

    random_state = RandomState(base_sim_seed)

    for true_start_glucose in true_glucose_start_values:
        for initial_error_value in error_glucose_values:

            new_sim_base_config = copy.deepcopy(json_sim_base_config)
            
            new_sim_base_config["controller"]["settings"]["max_physiologic_slope"] = 4  # add in velocity cap
            glucose_history_values = {i: true_start_glucose for i in range(num_history_values)}

            new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
            new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values
            
            new_sim_base_config["controller"]["id"] = 'swift'
            new_sim_base_config["controller"]["settings"]["partial_application_factor"] = paf
            new_sim_base_config["controller"]["settings"]["use_mid_absorption_isf"] = True
            new_sim_base_config["controller"]["settings"]["include_positive_velocity_and_RC"] = positive_rc
            new_sim_base_config["controller"]["settings"]["suspend_threshold"] = 70
            
            date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
            glucose_datetimes = [datetime.datetime.strptime(dt_str, date_str_format)
                                    for dt_str in
                                    new_sim_base_config["patient"]["sensor"]["glucose_history"]["datetime"].values()]
            
            t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

            sim_parser = ScenarioParserV2()

            sim_id = "icgm_analysis_vp_{}_{}_tbg={}_sbg={}".format(base_sim_seed, new_sim_base_config["patient_id"], true_start_glucose, initial_error_value)
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
                # return False 
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
        
    return


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


def build_icgm_sim_generator(json_base_configs, paf, positive_rc, sim_batch_size=30, 
                             true_bg_values=None, sensor_bg_values=None):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    
    Args:
        json_base_configs: List of base simulation configurations
        paf: Partial application factor
        positive_rc: Include positive RC and momentum
        sim_batch_size: Number of simulations per batch
        true_bg_values: Optional list/range of true glucose values
        sensor_bg_values: Optional list/range of sensor glucose values
    """
    for i, json_config in enumerate(json_base_configs, 1):

        logger.info("VP: {}. {} of {}".format(json_config["patient_id"], i, len(json_base_configs)))

        sim_ctr = 0
        sims = {}

        for sim in generate_icgm_point_error_simulations(json_config, base_sim_seed=i, paf=paf, positive_rc=positive_rc,
                                                         true_bg_values=true_bg_values, sensor_bg_values=sensor_bg_values):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

        yield sims


def run_icgm_simulations(paf_values=None, positive_rc_values=None, base_result_dir=None, 
                         num_vps=None, true_bg_values=None, sensor_bg_values=None):
    """
    Pipeline wrapper to run iCGM simulations with configurable parameters.
    
    Args:
        paf_values: List of partial application factor values to test (default: [0.4])
        positive_rc_values: List of positive RC boolean values to test (default: [True])
        base_result_dir: Base directory for results (default: DATA_DIR/processed/)
        num_vps: Number of virtual patients (None = all available)
        true_bg_values: Optional list/range of true glucose values (default: range(40, 80, 5))
        sensor_bg_values: Optional list/range of sensor glucose values (default: range(80, 120, 5))
    
    Returns:
        result_dirs: List of directories where results were saved
    """
    # Set defaults
    if paf_values is None:
        paf_values = [0.4]
    if positive_rc_values is None:
        positive_rc_values = [True]
    if base_result_dir is None:
        base_result_dir = os.path.join(DATA_DIR, "processed")
    if true_bg_values is None:
        true_bg_values = range(40, 80, 5)
    if sensor_bg_values is None:
        sensor_bg_values = range(80, 120, 5)
    
    # Setup for multiprocessing
    sim_batch_size = os.cpu_count() or 1
    os.environ['NUMEXPR_MAX_THREADS'] = str(sim_batch_size)
    numexpr.set_num_threads(sim_batch_size)
    
    # Disable logging for run and utils modules
    logging.getLogger("tidepool_data_science_simulator.run").disabled = True 
    logging.getLogger("tidepool_data_science_simulator.utils").disabled = True 
    
    # Get timestamp and git hash for directory naming
    date_string = datetime.datetime.now().strftime(r"%Y_%m_%d_T_%H_%M_%S_")
    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'], text=True).strip()
    
    # Get virtual patient configurations
    json_base_configs = transform_icgm_json_to_v2_parser()
    
    # Limit to specified number of VPs if requested
    if num_vps is not None:
        json_base_configs = json_base_configs[:num_vps]
    
    # Calculate total expected batches for progress tracking
    sims_per_config = len(true_bg_values) * len(sensor_bg_values)  
    batches_per_config = math.ceil(sims_per_config / sim_batch_size)
    total_expected_batches = len(json_base_configs) * batches_per_config * len(paf_values) * len(positive_rc_values)
    
    logger.info(f"Expected to process {total_expected_batches} total batches across {len(json_base_configs)} virtual patients")
    logger.info(f"Simulations per config: {sims_per_config}, Batch size: {sim_batch_size}, Batches per config: {batches_per_config}")
    
    # Initialize batch tracking variables
    completed_batches = 0
    batch_durations = []  # Track recent batch durations for rolling average
    overall_start_time = time.time()
    
    result_dirs = []
    
    # Iterate through all combinations of PAF and POSITIVE_RC values
    for paf in paf_values:
        for positive_rc in positive_rc_values:
            logger.info(f"Running simulations with PAF={paf}, POSITIVE_RC={positive_rc}")
            
            result_dir = os.path.join(base_result_dir, f"icgm_sensitivity_analysis_paf={paf}_posrc={positive_rc}_" + date_string + short_hash)
            
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
                logger.info("Made directory for results: {}".format(result_dir))
            
            result_dirs.append(result_dir)

            sim_batch_generator = build_icgm_sim_generator(
                json_base_configs, 
                paf=paf, 
                positive_rc=positive_rc, 
                sim_batch_size=sim_batch_size, 
                true_bg_values=true_bg_values, 
                sensor_bg_values=sensor_bg_values
            )

            parameter_start_time = time.time()
            for i, sim_batch in enumerate(sim_batch_generator):
                if sim_batch:
                    batch_start_time = time.time()

                    full_results, summary_results_df = run_simulations(
                        sim_batch,
                        save_dir=result_dir,
                        save_results=True,
                        compute_summary_metrics=False,
                        num_procs=sim_batch_size
                    )
                    
                    # Track batch timing and progress
                    batch_duration_minutes = (time.time() - batch_start_time) / 60
                    batch_durations.append(batch_duration_minutes)
                    completed_batches += 1
                    
                    # Keep only last 10 batch durations for rolling average
                    if len(batch_durations) > 10:
                        batch_durations.pop(0)
                    
                    # Calculate progress statistics
                    progress_percentage = (completed_batches / total_expected_batches) * 100
                    average_batch_time = sum(batch_durations) / len(batch_durations)
                    remaining_batches = total_expected_batches - completed_batches
                    estimated_remaining_minutes = remaining_batches * average_batch_time
                    
                    # Calculate total elapsed time
                    total_elapsed_minutes = (time.time() - overall_start_time) / 60
                    parameter_elapsed_minutes = (time.time() - parameter_start_time) / 60
                    
                    # Enhanced logging with progress and ETA
                    logger.info(f"=== BATCH PROGRESS ===")
                    logger.info(f"Batch {completed_batches}/{total_expected_batches} ({progress_percentage:.1f}% complete)")
                    logger.info(f"Current batch: {batch_duration_minutes:.2f} min | {len(sim_batch)} simulations")
                    logger.info(f"Average batch time: {average_batch_time:.2f} min (last {len(batch_durations)} batches)")
                    logger.info(f"Estimated time remaining: {estimated_remaining_minutes:.1f} min ({estimated_remaining_minutes/60:.1f} hrs)")
                    logger.info(f"Total elapsed: {total_elapsed_minutes:.1f} min | Parameter set elapsed: {parameter_elapsed_minutes:.1f} min")
                    logger.info(f"PAF={paf}, POSITIVE_RC={positive_rc}")
            
            parameter_total_time = (time.time() - parameter_start_time) / 60
            logger.info(f"=== PARAMETER SET COMPLETE ===")
            logger.info(f"Completed simulations for PAF={paf}, POSITIVE_RC={positive_rc}")
            logger.info(f"Parameter set time: {parameter_total_time:.2f} minutes ({parameter_total_time/60:.2f} hours)")
            logger.info(f"Overall progress: {completed_batches}/{total_expected_batches} batches completed")
    
    logger.info(f"All simulations complete. Results saved to: {result_dirs}")
    return result_dirs
