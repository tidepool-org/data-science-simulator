"""
Functional simulation builder for insulin algorithm testing.

This module provides pure functions for converting scenario dictionaries into
Simulation objects. It follows functional programming principles:
- Pure functions (no side effects)
- Immutability (no mutation of inputs)
- Function composition
- Partial application for configuration injection

Example:
    >>> from simulation_builder import build_simulation
    >>> simulation = build_simulation(config, scenario)
    
    >>> # Or with partial application for batch processing
    >>> from functools import partial
    >>> build_fn = partial(build_simulation, config)
    >>> simulations = map(build_fn, scenarios)
"""

import logging
import copy
import datetime
import types
from typing import Dict, Any, Optional, Callable, Iterator, List, Iterable, Tuple
from functools import partial, reduce

from numpy.random import RandomState
import numpy as np

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import CarbTimeline
from tidepool_data_science_simulator.models.measures import Carb
from tidepool_data_science_simulator.models.sensor_icgm import NoisySensorInitialOffset
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from itertools import product
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import (
    ExperimentConfig, AlgorithmConfig, SimulationConfig
)

logger = logging.getLogger(__name__)


# ========================================
# Pure Configuration Functions
# ========================================

def configure_initial_glucose(
    patient_config: Dict[str, Any], 
    true_start_bg: float
) -> Dict[str, Any]:
    """
    Returns new patient config with initial glucose values set.
    
    Pure function - does not mutate input.
    Uses selective copying for performance (only deep copies paths that will be mutated).
    
    Args:
        patient_config: Patient configuration dictionary
        true_start_bg: Initial blood glucose value (mg/dL)
        
    Returns:
        New patient config with glucose history initialized
    """
    # Selective copy: shallow copy top level, deep copy only mutated paths
    new_config = patient_config.copy()
    new_config["patient"] = patient_config["patient"].copy()
    
    # Deep copy only the sensor and patient_model sections (where glucose_history is mutated)
    new_config["patient"]["sensor"] = copy.deepcopy(patient_config["patient"]["sensor"])
    new_config["patient"]["patient_model"] = copy.deepcopy(patient_config["patient"]["patient_model"])
    
    # Set glucose history
    num_history_values = len(new_config["patient"]["sensor"]["glucose_history"]["value"])
    glucose_history_values = {i: true_start_bg for i in range(num_history_values)}
    
    new_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
    new_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values
    
    return new_config


def configure_algorithm_settings(
    sim_config: Dict[str, Any],
    algorithm_type: str,
    algorithm_config: AlgorithmConfig,
    partial_application_factor: Optional[float] = None,
    gradual_transition_threshold: Optional[float] = None
) -> Dict[str, Any]:
    """
    Returns new config with algorithm-specific settings.
    
    Pure function - does not mutate input.
    Uses selective copying for performance (only deep copies paths that will be mutated).
    
    Args:
        sim_config: Simulation configuration dictionary
        algorithm_type: 'tempbasal' or 'autobolus'
        algorithm_config: Algorithm configuration object
        partial_application_factor: PAF value (for autobolus)
        gradual_transition_threshold: Gradual transition threshold value
        
    Returns:
        New config with algorithm settings applied
    """
    # Selective copy: shallow copy top level, deep copy only mutated paths
    new_config = sim_config.copy()
    
    # Deep copy only controller section (where settings are mutated)
    new_config["controller"] = copy.deepcopy(sim_config["controller"])
    
    # Set controller ID
    new_config["controller"]["id"] = algorithm_config.controller_id
    
    # Set basal rate cap
    basal_rate = new_config['patient']['patient_model']['metabolism_settings']['basal_rate']['values'][0]
    new_config['controller']['settings']['max_basal_rate'] = basal_rate * algorithm_config.max_basal_multiplier
    
    # Set gradual transition threshold
    if gradual_transition_threshold is not None:
        new_config["controller"]["settings"]["gradual_transitions_threshold"] = gradual_transition_threshold
    else:
        new_config["controller"]["settings"]["gradual_transitions_threshold"] = algorithm_config.gradual_transition_thresholds[0]
    
    # Common settings
    new_config["controller"]["settings"]["include_positive_velocity_and_RC"] = algorithm_config.include_positive_velocity_and_RC
    new_config["controller"]["settings"]["use_mid_absorption_isf"] = algorithm_config.use_mid_absorption_isf
    
    # Add velocity cap to match original implementation
    new_config["controller"]["settings"]["max_physiologic_slope"] = 4
    
    # Add suspend threshold to match original implementation
    new_config["controller"]["settings"]["suspend_threshold"] = 70
    
    # Algorithm-specific settings
    if algorithm_type == 'autobolus':
        new_config["controller"]["settings"]["minimum_autobolus"] = algorithm_config.minimum_autobolus
        new_config["controller"]["settings"]["maximum_autobolus"] = algorithm_config.maximum_autobolus
        
        if partial_application_factor is not None:
            new_config["controller"]["settings"]["partial_application_factor"] = partial_application_factor
        else:
            new_config["controller"]["settings"]["partial_application_factor"] = algorithm_config.partial_application_factors[0]
    
    return new_config


def configure_settings_mismatches(
    sim_config: Dict[str, Any],
    settings_multipliers: Dict[str, float]
) -> Dict[str, Any]:
    """
    Returns new config with settings mismatches applied.
    
    Pure function - does not mutate input.
    Uses selective copying for performance (only deep copies paths that will be mutated).
    
    Args:
        sim_config: Simulation configuration dictionary
        settings_multipliers: Dictionary of parameter multipliers
        
    Returns:
        New config with adjusted settings
    """
    # Selective copy: shallow copy top level, deep copy only mutated paths
    new_config = sim_config.copy()
    new_config["patient"] = sim_config["patient"].copy()
    new_config["patient"]["patient_model"] = sim_config["patient"]["patient_model"].copy()
    
    # Deep copy only metabolism_settings (where values are mutated)
    new_config["patient"]["patient_model"]["metabolism_settings"] = copy.deepcopy(
        sim_config["patient"]["patient_model"]["metabolism_settings"]
    )
    
    metabolism_settings = new_config['patient']['patient_model']['metabolism_settings']
    
    if 'isf' in settings_multipliers:
        isf_values = metabolism_settings['insulin_sensitivity_factor']['values']
        metabolism_settings['insulin_sensitivity_factor']['values'] = [
            v * settings_multipliers['isf'] for v in isf_values
        ]
    
    if 'cir' in settings_multipliers:
        cir_values = metabolism_settings['carb_insulin_ratio']['values']
        metabolism_settings['carb_insulin_ratio']['values'] = [
            v * settings_multipliers['cir'] for v in cir_values
        ]
    
    if 'basal' in settings_multipliers:
        basal_values = metabolism_settings['basal_rate']['values']
        metabolism_settings['basal_rate']['values'] = [
            v * settings_multipliers['basal'] for v in basal_values
        ]
    
    return new_config


# ========================================
# Timeline Creation Functions
# ========================================

def create_meal_timeline(
    base_time: datetime.datetime,
    meal_scenario: Dict[str, Any]
) -> CarbTimeline:
    """
    Creates a meal timeline from scenario parameters.
    
    Pure function.
    
    Args:
        base_time: Base simulation time
        meal_scenario: Meal configuration dictionary
        
    Returns:
        CarbTimeline object
    """
    # Add meal timing offset
    meal_time = base_time + datetime.timedelta(minutes=meal_scenario.get('timing', 0))
    
    # Create carb event
    carb_amount = meal_scenario['size']
    absorption_time = meal_scenario.get('absorption_time', 240)
    
    carb_event = Carb(carb_amount, "g", absorption_time)
    return CarbTimeline(datetimes=[meal_time], events=[carb_event])


def parse_base_time(config: Dict[str, Any]) -> datetime.datetime:
    """
    Extracts base simulation time from config.
    
    Pure function.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Parsed datetime object
    """
    date_str_format = "%m/%d/%Y %H:%M:%S"
    return datetime.datetime.strptime(config["time_to_calculate_at"], date_str_format)


def parse_glucose_datetimes(config: Dict[str, Any]) -> List[datetime.datetime]:
    """
    Extracts glucose history datetimes from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of datetime objects for glucose history
    """
    date_str_format = "%m/%d/%Y %H:%M:%S"
    datetime_dict = config["patient"]["sensor"]["glucose_history"]["datetime"]
    return [
        datetime.datetime.strptime(dt_str, date_str_format)
        for dt_str in datetime_dict.values()
    ]


# ========================================
# Sensor Creation Functions (for iCGM scenarios)
# ========================================

def create_noisy_sensor_initial_offset(
    t0_init: datetime.datetime,
    t0: datetime.datetime,
    random_state: RandomState,
    initial_error_value: float,
    std_dev: float = 3.0
) -> NoisySensorInitialOffset:
    """
    Creates a NoisySensorInitialOffset sensor with specified error at t0.
    
    This sensor type allows specifying the exact sensor BG value at t0,
    which is critical for iCGM sensitivity analysis scenarios.
    
    Args:
        t0_init: Time to initialize sensor (typically t0 - history_length)
        t0: Simulation start time (when sensor error is applied)
        random_state: Random state for reproducibility
        initial_error_value: The sensor BG value to report at t0 (mg/dL)
        std_dev: Standard deviation for sensor noise (default: 3.0)
        
    Returns:
        Configured NoisySensorInitialOffset sensor
    """
    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.std_dev = std_dev
    
    sensor = NoisySensorInitialOffset(
        time=t0_init,
        t0_error_bg=initial_error_value,
        sensor_config=sensor_config,
        random_state=random_state,
        sim_start_time=t0
    )
    sensor.name = f"NoisySensor_{initial_error_value}"
    
    return sensor


def configure_sensor_history(
    sensor: NoisySensorInitialOffset,
    glucose_datetimes: List[datetime.datetime],
    glucose_history_values: Dict[int, float]
) -> NoisySensorInitialOffset:
    """
    Updates sensor state through glucose history time points.
    
    This is required to properly initialize the sensor's internal state
    before the simulation begins.
    
    Args:
        sensor: The sensor to update
        glucose_datetimes: List of datetime objects for history points
        glucose_history_values: Dict mapping index -> glucose value
        
    Returns:
        The updated sensor (same object, mutated)
    """
    for dt, true_bg in zip(glucose_datetimes, glucose_history_values.values()):
        sensor.update(dt, patient_true_bg=true_bg, patient_true_bg_prediction=[])
    
    return sensor


def create_bolus_acceptance_method(t0: datetime.datetime):
    """
    Creates a bolus acceptance method that always returns False.
    
    This is used for iCGM scenarios where we do NOT want the patient to 
    accept any bolus recommendations (matching the original implementation).
    
    Args:
        t0: The time at which to accept bolus recommendations (unused, kept for API compatibility)
        
    Returns:
        Method function to be bound to virtual patient
    """
    def does_accept_bolus_recommendation(self, bolus):
        return False
    
    return does_accept_bolus_recommendation


# ========================================
# Simulation ID Generation
# ========================================

def generate_simulation_id(
    algorithm_type: str,
    patient_id: str,
    scenario: Dict[str, Any],
    algorithm_config: AlgorithmConfig
) -> str:
    """
    Generates unique simulation ID from scenario parameters.
    
    Pure function.
    
    Args:
        algorithm_type: Algorithm type ('tempbasal' or 'autobolus')
        patient_id: Patient identifier
        scenario: Scenario dictionary
        algorithm_config: Algorithm configuration
        
    Returns:
        Simulation ID string
    """
    true_start_bg = scenario['true_start_bg']
    sensor_start_bg = scenario.get('sensor_start_bg')
    true_meal_scenario = scenario.get('true_meal_scenario')
    reported_meal_scenario = scenario.get('reported_meal_scenario')
    partial_application_factor = scenario.get('partial_application_factor')
    settings_multipliers = scenario.get('settings_multipliers')
    gradual_transition_threshold = scenario.get('gradual_transition_threshold')
    
    # Base ID components
    id_parts = [
        f"alg={algorithm_type}",
        f"patient={patient_id}",
        f"tbg={true_start_bg}",
    ]
    
    # Add sensor BG for iCGM scenarios (critical for uniqueness)
    if sensor_start_bg is not None:
        id_parts.append(f"sbg={sensor_start_bg}")
    
    # Add meal scenarios
    id_parts.append(f"true_meal_size={true_meal_scenario['size']}g" if true_meal_scenario else "true_meal_size=None")
    id_parts.append(f"reported_meal_size={reported_meal_scenario['size']}g" if reported_meal_scenario else "reported_meal_size=None")
    
    # Add algorithm settings
    id_parts.append(f"posvel={algorithm_config.include_positive_velocity_and_RC}")
    id_parts.append(f"midisf={algorithm_config.use_mid_absorption_isf}")
    
    # Add partial application factor for autobolus
    if partial_application_factor is not None:
        id_parts.append(f"paf={partial_application_factor}")
    
    # Add gradual transition threshold
    if gradual_transition_threshold is not None:
        id_parts.append(f"gradthresh={gradual_transition_threshold}")
    
    # Add settings mismatches
    if settings_multipliers:
        for param, multiplier in settings_multipliers.items():
            id_parts.append(f"{param}={multiplier}")
    
    return "_".join(id_parts)


def generate_simulations(
    config: ExperimentConfig,
    patient_configs: List[Dict[str, Any]],
    true_bg_range: tuple = (40, 405, 5),
    sensor_bg_range: Optional[tuple] = None,
    algorithm: str = 'autobolus',
    sensor_std_dev: float = 3.0
) -> Tuple[Iterator[Tuple[str, Simulation]], int]:
    """
    Generate iCGM simulations directly from config, bypassing scenario dictionaries.
    
    This is a streamlined generator that yields ready-to-run Simulation objects
    directly, without creating intermediate scenario dictionaries.
    
    Args:
        config: Experiment configuration
        patient_configs: List of patient configuration dictionaries
        true_bg_range: (start, end, step) for true BG values in mg/dL
        sensor_bg_range: (start, end, step) for sensor BG values.
                        If None, uses same as true_bg_range
        algorithm: Algorithm to test (typically 'autobolus')
        sensor_std_dev: Standard deviation for sensor noise model
        
    Returns:
        Tuple of (generator, num_sims) where:
        - generator yields (sim_id, simulation) tuples
        - num_sims is the total count of simulations
        
    Example:
        >>> sim_generator, num_sims = generate_simulations(config, patient_configs)
        >>> for sim_id, simulation in sim_generator:
        ...     simulation.run()
    """
    if sensor_bg_range is None:
        sensor_bg_range = true_bg_range
    
    algorithm_config = config.get_algorithm_config(algorithm)
    sim_config_obj = config.get_simulation_config()
    
    # Calculate total number of simulations
    num_sims = count_simulations(config, patient_configs, true_bg_range, sensor_bg_range, algorithm)
    
    def _simulation_generator():
        """Inner generator function that yields simulations."""
        # Generate BG grids
        true_start, true_end, true_step = true_bg_range
        sensor_start, sensor_end, sensor_step = sensor_bg_range
        
        true_bg_values = list(range(true_start, true_end, true_step))
        sensor_bg_values = list(range(sensor_start, sensor_end, sensor_step))
        
        logger.info(
            f"Generating {num_sims} iCGM simulations directly "
            f"({len(patient_configs)} patients × {len(true_bg_values)} true BG × "
            f"{len(sensor_bg_values)} sensor BG × "
            f"{len(algorithm_config.partial_application_factors)} PAF × "
            f"{len(algorithm_config.gradual_transition_thresholds)} thresholds)"
        )
        
        sim_count = 0
        
        # Generate all combinations of parameters
        combinations = product(
            enumerate(patient_configs, start=1),
            true_bg_values,
            sensor_bg_values,
            algorithm_config.partial_application_factors,
            algorithm_config.gradual_transition_thresholds
        )
        
        for (vp_index, patient_config), true_bg, sensor_bg, paf, gradual_threshold in combinations:
            # Initialize random state for this simulation
            np.random.seed(vp_index)
            random_state = RandomState(vp_index)
            
            # Configure initial glucose
            sim_config = configure_initial_glucose(patient_config, true_bg)
            
            # Configure algorithm settings
            sim_config = configure_algorithm_settings(
                sim_config, algorithm, algorithm_config,
                paf, gradual_threshold
            )
            
            # Get base time
            base_time = parse_base_time(sim_config)
            
            # Create sensor with initial offset
            num_history_values = len(sim_config["patient"]["sensor"]["glucose_history"]["value"])
            t0_init = base_time - datetime.timedelta(minutes=num_history_values * 5.0)
            
            sensor = create_noisy_sensor_initial_offset(
                t0_init=t0_init,
                t0=base_time,
                random_state=random_state,
                initial_error_value=sensor_bg,
                std_dev=sensor_std_dev
            )
            
            # Update sensor through glucose history
            glucose_datetimes = parse_glucose_datetimes(sim_config)
            glucose_history_values = sim_config["patient"]["sensor"]["glucose_history"]["value"]
            configure_sensor_history(sensor, glucose_datetimes, glucose_history_values)
            
            # Build simulation components
            sim_parser = ScenarioParserV2()
            sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(
                sim_config, sensor=sensor, random_state=random_state
            )
            
            # Set sensor on virtual patient
            virtual_patient.sensor = sensor
            
            # Configure bolus acceptance (always False for iCGM)
            bolus_method = create_bolus_acceptance_method(base_time)
            virtual_patient.does_accept_bolus_recommendation = types.MethodType(
                bolus_method, virtual_patient
            )
            
            # Generate simulation ID
            patient_id = patient_config.get('patient_id', 'unknown')
            sim_id = (
                f"alg={algorithm}_patient={patient_id}_tbg={true_bg}_sbg={sensor_bg}_"
                f"true_meal_size=None_reported_meal_size=None_"
                f"posvel={algorithm_config.include_positive_velocity_and_RC}_"
                f"midisf={algorithm_config.use_mid_absorption_isf}_"
                f"paf={paf}_gradthresh={gradual_threshold}"
            )
            
            # Create simulation
            simulation = Simulation(
                sim_start_time,
                duration_hrs=sim_config_obj.duration_hours,
                virtual_patient=virtual_patient,
                controller=controller,
                multiprocess=True,
                sim_id=sim_id,
                random_state=random_state
            )
            
            sim_count += 1
            if sim_count % 100 == 0:
                logger.debug(f"Generated {sim_count} simulations")
            
            yield (sim_id, simulation)
        
        logger.info(f"Generated {sim_count} total iCGM simulations")
    
    return _simulation_generator(), num_sims


# ========================================
# Utility Functions
# ========================================

def count_simulations(
    config: ExperimentConfig,
    patient_configs: List[Dict[str, Any]],
    true_bg_range: tuple,
    sensor_bg_range: Optional[tuple] = None,
    algorithm: str = 'autobolus'
) -> int:
    """
    Calculate the total number of simulations that will be generated.
    
    This is useful for progress tracking when using generators.
    
    Args:
        config: Experiment configuration
        patient_configs: List of patient configuration dictionaries
        true_bg_range: (start, end, step) for true BG values in mg/dL
        sensor_bg_range: (start, end, step) for sensor BG values.
                        If None, uses same as true_bg_range
        algorithm: Algorithm to test (typically 'autobolus')
        
    Returns:
        Total number of simulations
        
    Example:
        >>> num_sims = count_simulations(config, patient_configs, true_bg_range)
        >>> for sim in generate_simulations(config, patient_configs, true_bg_range):
        ...     # process sim
    """
    if sensor_bg_range is None:
        sensor_bg_range = true_bg_range
    
    algorithm_config = config.get_algorithm_config(algorithm)
    
    true_bg_count = len(range(true_bg_range[0], true_bg_range[1], true_bg_range[2]))
    sensor_bg_count = len(range(sensor_bg_range[0], sensor_bg_range[1], sensor_bg_range[2]))
    
    return (
        len(patient_configs) *
        true_bg_count *
        sensor_bg_count *
        len(algorithm_config.partial_application_factors) *
        len(algorithm_config.gradual_transition_thresholds)
    )
