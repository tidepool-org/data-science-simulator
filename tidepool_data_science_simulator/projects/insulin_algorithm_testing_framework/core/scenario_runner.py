"""
Core simulation runner for insulin algorithm testing.

This module provides the main interface for running simulations with different
insulin delivery algorithms (temp basal vs autobolus) using the Tidepool simulator.
"""

import logging
import copy
import datetime
import time
from typing import Dict, Any, List, Optional, Tuple, Iterator

import pandas as pd
import numpy as np
from numpy.random import RandomState

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import CarbTimeline
from tidepool_data_science_simulator.models.measures import Carb
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import (
    ExperimentConfig, AlgorithmConfig, SimulationConfig
)

logger = logging.getLogger(__name__)


class ScenarioRunner:
    """
    Main class for running insulin algorithm simulations.
    
    Integrates with the Tidepool simulator to run comparisons between
    temp basal and autobolus algorithms across different scenarios.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the simulation runner.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.sim_config = config.get_simulation_config()
        self.processing_config = config.get_processing_config()
        self.random_state = RandomState(config.random_seed)
        
        logger.info(f"Initialized SimulationRunner with config: {config}")
    
    def run_single_simulation(
        self,
        algorithm_type: str,
        patient_config: Dict[str, Any],
        initial_bg: float,
        meal_scenario: Dict[str, Any],
        partial_application_factor: Optional[float] = None,
        settings_multipliers: Optional[Dict[str, float]] = None
    ) -> Tuple[str, pd.DataFrame]:
        """
        Run a single simulation with specified parameters.
        
        Args:
            algorithm_type: 'tempbasal' or 'autobolus'
            patient_config: Patient configuration dictionary
            initial_bg: Initial blood glucose (mg/dL)
            meal_scenario: Meal configuration (size, timing, etc.)
            partial_application_factor: For autobolus algorithm (0.2-0.6)
            settings_multipliers: Multipliers for ISF/CIR/basal mismatches
            
        Returns:
            Tuple of (simulation_id, results_dataframe)
        """
        # Create a deep copy of the base configuration
        sim_config = copy.deepcopy(patient_config)
        
        # Set initial glucose values
        num_history_values = len(sim_config["patient"]["sensor"]["glucose_history"]["value"])
        glucose_history_values = {i: initial_bg for i in range(num_history_values)}
        
        sim_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
        sim_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values
        
        # Configure algorithm-specific settings
        algorithm_config = self.config.get_algorithm_config(algorithm_type)
        self._configure_algorithm(sim_config, algorithm_type, algorithm_config, partial_application_factor)
        
        # Apply settings mismatches if specified
        if settings_multipliers:
            self._apply_settings_mismatches(sim_config, settings_multipliers)
        
        # Setup meal scenario
        meal_timeline = self._create_meal_timeline(sim_config, meal_scenario)
        
        # Parse configuration and create simulation components
        sim_parser = ScenarioParserV2()
        sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(sim_config)
        
        # Set meal timeline
        virtual_patient.carb_event_timeline = meal_timeline
        
        # Generate simulation ID
        sim_id = self._generate_simulation_id(
            algorithm_type, patient_config, initial_bg, meal_scenario, 
            partial_application_factor, settings_multipliers
        )
        
        # Create and run simulation
        simulation = Simulation(
            sim_start_time,
            duration_hrs=duration_hrs,
            virtual_patient=virtual_patient,
            controller=controller,
            multiprocess=False,  # Single simulation
            sim_id=sim_id
        )
        
        simulation.random_state = self.random_state
        results = simulation.run()
        
        # Convert to DataFrame
        results_df = simulation.get_results_df()
        
        logger.debug(f"Completed simulation: {sim_id}")
        return sim_id, results_df
    
    def run_batch_scenarios(
        self,
        scenarios: Iterator[Dict[str, Any]],
        save_dir: Optional[str] = None
    ) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame]:
        """
        Run a batch of scenarios, processing them in chunks for efficiency.

        Args:
            scenarios: Iterator of scenario dictionaries.
            save_dir: Optional directory to save results.

        Returns:
            Dictionary of simulation_id -> results DataFrame for all scenarios.
        """
        simulations = {}
        batch_counter = 0
        total_batch_counter = 0
        
        full_results = {}
        total_start_time = time.time()

        for scenario in scenarios:
            # Create Simulation object from scenario
            simulation = self.create_simulation_from_scenario(scenario)
            simulations[simulation.sim_id] = simulation
            batch_counter += 1
            total_batch_counter += 1

            if batch_counter % self.processing_config.parallel_processes == 0:
                # Run batch simulations every N scenarios for efficiency
                results = self.run_parallel_batch_simulations(
                    simulations, 
                    save_dir=save_dir,
                    total_scenarios=total_batch_counter,
                    total_start_time=total_start_time
                )

                # Merge new results into the full results dictionary
                full_results = full_results | results
                simulations = {}  # Reset for next batch
                batch_counter = 0

        if simulations:
            # Run any remaining simulations that didn't fill a complete batch
            results = self.run_parallel_batch_simulations(
                simulations, 
                save_dir=save_dir,
                total_scenarios=total_batch_counter,
                total_start_time=total_start_time,
                is_final_batch=True
            )
            
            full_results = full_results | results  # Merge results

        total_duration = time.time() - total_start_time
        logger.info(f"Completed all {len(full_results)} simulations in {total_duration:.2f}s")

        return full_results

    
    def run_parallel_batch_simulations(self,
        simulations: Dict[str, Simulation],
        save_dir: Optional[str] = None,
        total_scenarios: Optional[int] = None,
        total_start_time: Optional[float] = None,
        is_final_batch: bool = False
    ) -> Dict[str, pd.DataFrame]:
        """
        Run a batch of simulations in parallel.
        
        Args:
            simulations: Dictionary of simulation_id -> Simulation objects
            save_dir: Optional directory to save results
            total_scenarios: Total number of scenarios processed so far
            total_start_time: Start time of the entire batch operation
            is_final_batch: Whether this is the final batch
            
        Returns:
            Dictionary of simulation_id -> results DataFrame
        """
        batch_start_time = time.time()
        
        results, _ = run_simulations(
            simulations,
            save_dir=save_dir or self.config.output_dir,
            save_results=self.processing_config.save_individual_results,
            compute_summary_metrics=True,
            num_procs=self.processing_config.parallel_processes
        )
        
        batch_end_time = time.time()
        batch_duration = batch_end_time - batch_start_time
        
        # Log timing information if tracking parameters are provided
        if total_scenarios is not None and total_start_time is not None:
            total_elapsed = time.time() - total_start_time
            batch_type = "final batch" if is_final_batch else "batch"
            logger.info(f"Completed {batch_type} of {len(simulations)} simulations in {batch_duration:.2f}s "
                       f"(total: {total_scenarios} scenarios, elapsed: {total_elapsed:.2f}s)")

        return results
    
    def _configure_algorithm(
        self,
        sim_config: Dict[str, Any],
        algorithm_type: str,
        algorithm_config: AlgorithmConfig,
        partial_application_factor: Optional[float] = None
    ) -> None:
        """Configure algorithm-specific settings."""
        
        # Set controller ID
        sim_config["controller"]["id"] = algorithm_config.controller_id
        
        # Set basal rate cap
        basal_rate = sim_config['patient']['patient_model']['metabolism_settings']['basal_rate']['values'][0]
        sim_config['controller']['settings']['max_basal_rate'] = basal_rate * algorithm_config.max_basal_multiplier
        
        # Algorithm-specific settings
        if algorithm_type == 'tempbasal':
            sim_config["controller"]["settings"]["include_positive_velocity_and_RC"] = algorithm_config.include_positive_velocity_and_RC
            sim_config["controller"]["settings"]["use_mid_absorption_isf"] = algorithm_config.use_mid_absorption_isf
            
        elif algorithm_type == 'autobolus':
            sim_config["controller"]["settings"]["include_positive_velocity_and_RC"] = algorithm_config.include_positive_velocity_and_RC
            sim_config["controller"]["settings"]["use_mid_absorption_isf"] = algorithm_config.use_mid_absorption_isf
            sim_config["controller"]["settings"]["minimum_autobolus"] = algorithm_config.minimum_autobolus
            sim_config["controller"]["settings"]["maximum_autobolus"] = algorithm_config.maximum_autobolus
            
            if partial_application_factor is not None:
                sim_config["controller"]["settings"]["partial_application_factor"] = partial_application_factor
            else:
                # Use first value from config if not specified
                sim_config["controller"]["settings"]["partial_application_factor"] = algorithm_config.partial_application_factors[0]
        
        else:
            raise ValueError(f"Unknown algorithm type: {algorithm_type}")
    
    def _apply_settings_mismatches(
        self,
        sim_config: Dict[str, Any],
        settings_multipliers: Dict[str, float]
    ) -> None:
        """Apply settings mismatches by multiplying true parameters."""
        
        metabolism_settings = sim_config['patient']['patient_model']['metabolism_settings']
        
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
    
    def _create_meal_timeline(
        self,
        sim_config: Dict[str, Any],
        meal_scenario: Dict[str, Any]
    ) -> CarbTimeline:
        """Create meal timeline from scenario configuration."""
        
        date_str_format = "%m/%d/%Y %H:%M:%S"
        t0 = datetime.datetime.strptime(sim_config["time_to_calculate_at"], date_str_format)
        
        # Add meal timing offset
        meal_time = t0 + datetime.timedelta(minutes=meal_scenario.get('timing', 0))
        
        # Create carb event
        carb_amount = meal_scenario['size']
        absorption_time = meal_scenario.get('absorption_time', 240)
        
        carb_event = Carb(carb_amount, "g", absorption_time)
        meal_timeline = CarbTimeline(datetimes=[meal_time], events=[carb_event])
        
        return meal_timeline
    
    def _generate_simulation_id(
        self,
        algorithm_type: str,
        patient_config: Dict[str, Any],
        initial_bg: float,
        meal_scenario: Dict[str, Any],
        partial_application_factor: Optional[float] = None,
        settings_multipliers: Optional[Dict[str, float]] = None
    ) -> str:
        """Generate unique simulation ID."""
        
        patient_id = patient_config.get('patient_id', 'unknown')
        
        # Base ID components
        id_parts = [
            f"alg={algorithm_type}",
            f"patient={patient_id}",
            f"ibg={initial_bg}",
            f"meal={meal_scenario['size']}g"
        ]
        
        # Add partial application factor for autobolus
        if partial_application_factor is not None:
            id_parts.append(f"paf={partial_application_factor}")
        
        # Add settings mismatches
        if settings_multipliers:
            for param, multiplier in settings_multipliers.items():
                id_parts.append(f"{param}={multiplier}")
        
        return "_".join(id_parts)
    
    def create_simulation_from_scenario(
        self,
        scenario: Dict[str, Any]
    ) -> Simulation:
        """
        Create a Simulation object from a scenario dictionary.
        
        Args:
            scenario: Dictionary containing all scenario parameters
            
        Returns:
            Configured Simulation object
        """
        # Extract scenario parameters
        algorithm_type = scenario['algorithm_type']
        patient_config = scenario['patient_config']
        initial_bg = scenario['initial_bg']
        meal_scenario = scenario['meal_scenario']
        partial_application_factor = scenario.get('partial_application_factor')
        settings_multipliers = scenario.get('settings_multipliers')
        
        # Create a deep copy of the base configuration
        sim_config = copy.deepcopy(patient_config)
        
        # Set initial glucose values
        num_history_values = len(sim_config["patient"]["sensor"]["glucose_history"]["value"])
        glucose_history_values = {i: initial_bg for i in range(num_history_values)}
        
        sim_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
        sim_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values
        
        # Configure algorithm-specific settings
        algorithm_config = self.config.get_algorithm_config(algorithm_type)
        self._configure_algorithm(sim_config, algorithm_type, algorithm_config, partial_application_factor)
        
        # Apply settings mismatches if specified
        if settings_multipliers:
            self._apply_settings_mismatches(sim_config, settings_multipliers)
        
        # Setup meal scenario
        meal_timeline = self._create_meal_timeline(sim_config, meal_scenario)
        
        # Parse configuration and create simulation components
        sim_parser = ScenarioParserV2()
        sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(sim_config)
        
        # Set meal timeline
        virtual_patient.carb_event_timeline = meal_timeline
        
        # Generate simulation ID
        sim_id = self._generate_simulation_id(
            algorithm_type, patient_config, initial_bg, meal_scenario, 
            partial_application_factor, settings_multipliers
        )
        
        # Create simulation
        simulation = Simulation(
            sim_start_time,
            duration_hrs=self.sim_config.duration_hours,
            virtual_patient=virtual_patient,
            controller=controller,
            multiprocess=True,  # For batch processing
            sim_id=sim_id
        )
        
        simulation.random_state = self.random_state
        
        return simulation
    
    def extract_glucose_trace(
        self,
        results_df: pd.DataFrame,
        start_hours: float = 0,
        duration_hours: Optional[float] = None
    ) -> np.ndarray:
        """
        Extract glucose trace from simulation results.
        
        Args:
            results_df: Simulation results DataFrame
            start_hours: Start time in hours from simulation start
            duration_hours: Duration to extract (None for all remaining)
            
        Returns:
            Glucose trace as numpy array
        """
        # Calculate indices
        start_idx = int(start_hours * 12)  # 5-minute intervals
        
        if duration_hours is not None:
            end_idx = start_idx + int(duration_hours * 12)
        else:
            end_idx = len(results_df)
        
        # Extract active data only
        active_data = results_df[results_df['active'] == 1]
        
        if len(active_data) == 0:
            logger.warning("No active data found in results")
            return np.array([])
        
        # Slice the data
        sliced_data = active_data.iloc[start_idx:end_idx]
        
        return sliced_data['bg'].values
    
    def extract_insulin_delivery(
        self,
        results_df: pd.DataFrame,
        start_hours: float = 0,
        duration_hours: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Extract insulin delivery data from simulation results.

        Args:
            results_df: Simulation results DataFrame
            start_hours: Start time in hours from simulation start
            duration_hours: Duration to extract (None for all remaining)

        Returns:
            Dictionary with insulin delivery metrics
        """
        # Calculate indices
        start_idx = int(start_hours * 12)  # 5-minute intervals
        
        if duration_hours is not None:
            end_idx = start_idx + int(duration_hours * 12)
        else:
            end_idx = len(results_df)
        
        # Extract active data only
        active_data = results_df[results_df['active'] == 1]
        
        if len(active_data) == 0:
            logger.warning("No active data found in results")
            return {'basal': 0.0, 'bolus': 0.0, 'total': 0.0}
        
        # Slice the data
        sliced_data = active_data.iloc[start_idx:end_idx]
        
        # Calculate insulin delivery
        basal_delivered = sliced_data['delivered_basal_insulin'].sum()
        bolus_delivered = sliced_data['true_bolus'].sum()
        total_delivered = basal_delivered + bolus_delivered
        
        return {
            'basal': basal_delivered,
            'bolus': bolus_delivered,
            'total': total_delivered
        }


def run_experiment(
    config: ExperimentConfig, 
    max_patients: Optional[int] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Run complete insulin algorithm comparison experiment.
    
    This function provides a simplified interface for running the entire
    insulin algorithm testing workflow, from loading patient data through
    statistical analysis.
    
    Args:
        config: ExperimentConfig object containing all experiment settings
        max_patients: Optional limit on number of patients (useful for testing/debugging)
    
    Returns:
        Tuple of (metrics_df, comparison_results) where:
        - metrics_df: DataFrame with detailed metrics for all simulations
        - comparison_results: Statistical comparison results between algorithms
    
    Example:
        >>> config = ExperimentConfig()
        >>> config.set('scenarios.initial_bg.range', [100, 200])
        >>> metrics_df, comparison_results = run_experiment(config, max_patients=5)
        >>> print(f"Completed {len(metrics_df)} simulations")
    """
    logger.info(f"Starting experiment: {config.experiment_name}")
    
    # Import required modules (using lazy imports to avoid circular dependencies)
    from .data_loader import DataLoader
    from .scenario_generator import ScenarioGenerator
    from .metrics_calculator import MetricsCalculator
    from ..analysis.statistical_analyzer import StatisticalAnalyzer
    
    try:
        # 1. Load patient data
        logger.info("Loading patient configurations...")
        data_loader = DataLoader(config)
        patient_configs = data_loader.load_patient_configs(max_patients=max_patients)
        
        if not patient_configs:
            raise ValueError("No patient configurations loaded")
        
        logger.info(f"Loaded {len(patient_configs)} patient configurations")
        
        # 2. Generate scenarios
        logger.info("Generating scenarios...")
        scenario_generator = ScenarioGenerator(config)
        
        # Get scenario summary for logging
        summary = scenario_generator.get_scenario_summary(patient_configs)
        logger.info(f"Scenario summary: {summary}")
        
        # Generate all scenarios
        scenarios = scenario_generator.generate_all_scenarios(patient_configs)
        logger.info(f"Generated scenario iterator (estimated: {summary['estimated_total_scenarios']} scenarios)")
        
        # 3. Run simulations
        logger.info("Running batch simulations...")
        simulation_runner = ScenarioRunner(config)
        full_results = simulation_runner.run_batch_scenarios(scenarios)
        
        if not full_results:
            raise ValueError("No simulation results generated")
        
        logger.info(f"Completed {len(full_results)} simulations")
        
        # 4. Calculate metrics
        logger.info("Calculating metrics...")
        metrics_calculator = MetricsCalculator(config)
        
        # Calculate metrics for all results
        metrics_dict = metrics_calculator.calculate_metrics_batch(full_results)
        
        # Create metrics DataFrame
        metrics_df = metrics_calculator.create_metrics_dataframe(metrics_dict)
        
        if metrics_df.empty:
            raise ValueError("No metrics calculated")
        
        logger.info(f"Calculated metrics for {len(metrics_dict)} simulations")
        logger.info(f"Metrics columns: {list(metrics_df.columns)}")
        
        # 5. Statistical analysis
        logger.info("Performing statistical analysis...")
        statistical_analyzer = StatisticalAnalyzer(config)
        
        # Get enabled algorithms for comparison
        enabled_algorithms = config.get_enabled_algorithms()
        
        if len(enabled_algorithms) < 2:
            logger.warning("Statistical comparison requires at least 2 algorithms")
            comparison_results = {}
        else:
            # Use first algorithm as reference, others as comparison
            reference_algorithm = enabled_algorithms[0]
            comparison_algorithms = enabled_algorithms[1:]
            
            # Perform paired comparison
            comparison_results = statistical_analyzer.compare_algorithms(
                metrics_df, 
                reference_algorithm=reference_algorithm,
                comparison_algorithms=comparison_algorithms
            )
        
        logger.info("Statistical analysis completed")
        
        # 6. Return results
        logger.info(f"Experiment completed successfully!")
        logger.info(f"Total simulations: {len(full_results)}")
        logger.info(f"Metrics calculated: {len(metrics_df)}")
        
        return metrics_df, comparison_results
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise
