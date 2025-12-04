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
from typing import Dict, Any, Optional, Callable, Iterator, List, Iterable
from functools import partial, reduce

from numpy.random import RandomState
import numpy as np

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import CarbTimeline
from tidepool_data_science_simulator.models.measures import Carb
from tidepool_data_science_simulator.models.sensor_icgm import NoisySensorInitialOffset
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
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
    
    Args:
        patient_config: Patient configuration dictionary
        true_start_bg: Initial blood glucose value (mg/dL)
        
    Returns:
        New patient config with glucose history initialized
    """
    # Deep copy to avoid mutation
    new_config = copy.deepcopy(patient_config)
    
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
    
    Args:
        sim_config: Simulation configuration dictionary
        algorithm_type: 'tempbasal' or 'autobolus'
        algorithm_config: Algorithm configuration object
        partial_application_factor: PAF value (for autobolus)
        gradual_transition_threshold: Gradual transition threshold value
        
    Returns:
        New config with algorithm settings applied
    """
    # Deep copy to avoid mutation
    new_config = copy.deepcopy(sim_config)
    
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
    
    Args:
        sim_config: Simulation configuration dictionary
        settings_multipliers: Dictionary of parameter multipliers
        
    Returns:
        New config with adjusted settings
    """
    # Deep copy to avoid mutation
    new_config = copy.deepcopy(sim_config)
    
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
    Creates a bolus acceptance method that only accepts at t0.
    
    This is used for iCGM scenarios where we want to simulate the impact
    of a single spurious CGM reading at t0.
    
    Args:
        t0: The time at which to accept bolus recommendations
        
    Returns:
        Method function to be bound to virtual patient
    """
    def does_accept_bolus_recommendation(self, bolus):
        return self.time == t0
    
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


# ========================================
# Main Builder Function
# ========================================

def build_simulation(
    config: ExperimentConfig,
    scenario: Dict[str, Any],
    random_state: Optional[RandomState] = None
) -> Simulation:
    """
    Builds a Simulation object from a scenario dictionary.
    
    This is a pure function that transforms scenario data into a Simulation.
    It applies a series of pure configuration transformations and creates
    the final Simulation object.
    
    Args:
        config: Experiment configuration
        scenario: Scenario dictionary containing all parameters
        random_state: Optional random state for reproducibility
        
    Returns:
        Configured Simulation object
        
    Example:
        >>> simulation = build_simulation(config, scenario)
        >>> 
        >>> # Or with partial application
        >>> from functools import partial
        >>> build_fn = partial(build_simulation, config)
        >>> simulations = map(build_fn, scenarios)
    """
    # Extract scenario parameters
    algorithm_type = scenario['algorithm_type']
    patient_config = scenario['patient_config']
    true_start_bg = scenario['true_start_bg']
    true_meal_scenario = scenario.get('true_meal_scenario')
    reported_meal_scenario = scenario.get('reported_meal_scenario')
    partial_application_factor = scenario.get('partial_application_factor')
    settings_multipliers = scenario.get('settings_multipliers')
    gradual_transition_threshold = scenario.get('gradual_transition_threshold')
    
    # Get configuration objects
    algorithm_config = config.get_algorithm_config(algorithm_type)
    sim_config_obj = config.get_simulation_config()
    
    # Apply configuration transformations (all pure functions)
    sim_config = configure_initial_glucose(patient_config, true_start_bg)
    sim_config = configure_algorithm_settings(
        sim_config, algorithm_type, algorithm_config,
        partial_application_factor, gradual_transition_threshold
    )
    
    if settings_multipliers:
        sim_config = configure_settings_mismatches(sim_config, settings_multipliers)
    
    # Get base time and glucose history for sensor configuration
    base_time = parse_base_time(sim_config)
    
    # Determine random state: use scenario's random_seed, then provided random_state, then default
    scenario_seed = scenario.get('random_seed')
    if scenario_seed is not None:
        # Synchronize both the global numpy RNG and the local RandomState so
        # code paths that use either `np.random` or a passed RandomState
        # produce the same deterministic noise as legacy code.
        np.random.seed(scenario_seed)
        random_state = RandomState(scenario_seed)
    elif random_state is None:
        # Default reproducible seed for both global and local RNGs
        np.random.seed(42)
        random_state = RandomState(42)  # Default seed for reproducibility
    
    # Check for iCGM scenario (sensor_start_bg present)
    sensor_start_bg = scenario.get('sensor_start_bg')
    accept_bolus_at_t0_only = scenario.get('accept_bolus_at_t0_only', False)
    sensor = None
    
    if sensor_start_bg is not None:
        # Create sensor with initial offset for iCGM scenarios
        num_history_values = len(sim_config["patient"]["sensor"]["glucose_history"]["value"])
        t0_init = base_time - datetime.timedelta(minutes=num_history_values * 5.0)
        
        # Use the random_state (already set above from scenario or provided)
        sensor_random_state = random_state
        
        sensor = create_noisy_sensor_initial_offset(
            t0_init=t0_init,
            t0=base_time,
            random_state=sensor_random_state,
            initial_error_value=sensor_start_bg
        )
        
        # Get glucose history and update sensor through history
        glucose_datetimes = parse_glucose_datetimes(sim_config)
        glucose_history_values = sim_config["patient"]["sensor"]["glucose_history"]["value"]
        configure_sensor_history(sensor, glucose_datetimes, glucose_history_values)
    
    # Parse configuration and create simulation components
    sim_parser = ScenarioParserV2()
    sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(
        sim_config, sensor=sensor
    )
    
    # If sensor was created, set it on virtual patient
    if sensor is not None:
        virtual_patient.sensor = sensor
    
    # Configure bolus acceptance for iCGM scenarios
    if accept_bolus_at_t0_only or sensor_start_bg is not None:
        bolus_method = create_bolus_acceptance_method(base_time)
        virtual_patient.does_accept_bolus_recommendation = types.MethodType(
            bolus_method, virtual_patient
        )
    
    # Set meal timelines
    if true_meal_scenario:
        true_meal_timeline = create_meal_timeline(base_time, true_meal_scenario)
        virtual_patient.carb_event_timeline = true_meal_timeline
    
    if reported_meal_scenario:
        reported_meal_timeline = create_meal_timeline(base_time, reported_meal_scenario)
        controller.reported_carb_event_timeline = reported_meal_timeline
    
    # Generate simulation ID
    patient_id = patient_config.get('patient_id', 'unknown')
    sim_id = generate_simulation_id(algorithm_type, patient_id, scenario, algorithm_config)
    
    # Create simulation object
    simulation = Simulation(
        sim_start_time,
        duration_hrs=sim_config_obj.duration_hours,
        virtual_patient=virtual_patient,
        controller=controller,
        multiprocess=True,
        sim_id=sim_id
    )
    
    if random_state is not None:
        simulation.random_state = random_state
    
    return simulation


def build_simulations(
    config: ExperimentConfig,
    scenarios: Iterable[Dict[str, Any]],
    random_state: Optional[RandomState] = None
) -> Dict[str, Simulation]:
    """
    Build multiple Simulation objects from scenario dictionaries.
    
    This is the primary batch-building function that converts scenario
    dictionaries into ready-to-run Simulation objects.
    
    Args:
        config: Experiment configuration
        scenarios: Iterable of scenario dictionaries
        random_state: Optional random state for reproducibility
        
    Returns:
        Dictionary mapping simulation_id -> Simulation object
        
    Example:
        >>> simulations = build_simulations(config, scenarios)
        >>> print(f"Built {len(simulations)} simulations")
    """
    simulations = {}
    
    for scenario in scenarios:
        simulation = build_simulation(config, scenario, random_state=random_state)
        simulations[simulation.sim_id] = simulation
    
    return simulations


# ========================================
# Utility Functions
# ========================================

def count_scenarios(scenarios: Iterable[Dict[str, Any]]) -> int:
    """
    Counts scenarios in an iterable (consumes iterator).
    
    Args:
        scenarios: Iterable of scenario dictionaries
        
    Returns:
        Count of scenarios
    """
    return sum(1 for _ in scenarios)


