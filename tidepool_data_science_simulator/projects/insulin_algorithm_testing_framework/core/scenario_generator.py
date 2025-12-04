"""
Scenario generator for insulin algorithm testing.

This module generates comprehensive test scenarios for comparing insulin delivery
algorithms across different patient parameters, initial conditions, and meal scenarios.
"""

import logging
import itertools
from typing import Dict, Any, List, Iterator, Optional, Tuple

import pandas as pd
import numpy as np

from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import (
    ExperimentConfig, ScenarioConfig
)

logger = logging.getLogger(__name__)


class ScenarioGenerator:
    """
    Generates test scenarios for insulin algorithm comparisons.
    
    Creates combinations of:
    - Initial blood glucose values
    - Meal scenarios (unannounced meals)
    - Settings mismatches (ISF, CIR, basal multipliers)
    - Partial application factors (for autobolus)
    - Patient parameters
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the scenario generator.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.scenario_config = config.get_scenario_config()
        
        logger.info(f"Initialized ScenarioGenerator with config: {config}")
    
    def generate_all_scenarios(
        self,
        patient_configs: List[Dict[str, Any]],
        algorithms: Optional[List[str]] = None
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate all scenario combinations.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            algorithms: List of algorithms to test (None for all enabled)
            
        Yields:
            Dictionary containing scenario parameters
        """
        if algorithms is None:
            algorithms = self.config.get_enabled_algorithms()
        
        logger.info(f"Generating scenarios for algorithms: {algorithms}")
        logger.info(f"Patient configs: {len(patient_configs)}")
        
        total_scenarios = self._estimate_total_scenarios(patient_configs, algorithms)
        logger.info(f"Estimated total scenarios: {total_scenarios}")
        
        scenario_count = 0
        
        for patient_config in patient_configs:
            for algorithm in algorithms:
                for scenario in self._generate_algorithm_scenarios(patient_config, algorithm):
                    scenario_count += 1
                    if scenario_count % 1000 == 0:
                        logger.debug(f"Generated {scenario_count} scenarios")
                    
                    yield scenario
        
        logger.info(f"Generated {scenario_count} total scenarios")
    
    def generate_scenarios_for_algorithm(
        self,
        patient_configs: List[Dict[str, Any]],
        algorithm: str
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate scenarios for a specific algorithm.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            algorithm: Algorithm name ('tempbasal' or 'autobolus')
            
        Yields:
            Dictionary containing scenario parameters
        """
        logger.info(f"Generating scenarios for algorithm: {algorithm}")
        
        for patient_config in patient_configs:
            for scenario in self._generate_algorithm_scenarios(patient_config, algorithm):
                yield scenario
    
    def _generate_algorithm_scenarios(
        self,
        patient_config: Dict[str, Any],
        algorithm: str
    ) -> Iterator[Dict[str, Any]]:
        """Generate scenarios for a specific patient and algorithm."""
        
        # Get algorithm configuration
        algorithm_config = self.config.get_algorithm_config(algorithm)
        
        # Generate base parameter combinations
        for true_start_bg in self._generate_initial_bg_values():
            for meal_scenario in self._generate_meal_scenarios():
                for settings_multipliers in self._generate_settings_mismatches():
                    for gradual_threshold in algorithm_config.gradual_transition_thresholds:
                        
                        if algorithm == 'tempbasal':
                            # Temp basal doesn't use partial application factor
                            yield {
                                'algorithm_type': algorithm,
                                'patient_config': patient_config,
                                'true_start_bg': true_start_bg,
                                'meal_scenario': meal_scenario,
                                'settings_multipliers': settings_multipliers,
                                'partial_application_factor': None,
                                'gradual_transition_threshold': gradual_threshold
                            }
                        
                        elif algorithm == 'autobolus':
                            # Generate scenarios for each partial application factor
                            for paf in algorithm_config.partial_application_factors:
                                yield {
                                    'algorithm_type': algorithm,
                                    'patient_config': patient_config,
                                    'true_start_bg': true_start_bg,
                                    'meal_scenario': meal_scenario,
                                    'settings_multipliers': settings_multipliers,
                                    'partial_application_factor': paf,
                                    'gradual_transition_threshold': gradual_threshold
                                }
    
    def _generate_initial_bg_values(self) -> List[float]:
        """Generate initial blood glucose values."""
        start, end = self.scenario_config.initial_bg_range
        step = self.scenario_config.initial_bg_step
        
        return list(range(start, end + 1, step))
    
    def _generate_meal_scenarios(self) -> List[Dict[str, Any]]:
        """Generate meal scenarios."""
        meal_scenarios = []
        
        for meal_size in self.scenario_config.unannounced_meals:
            meal_scenarios.append({
                'size': meal_size,
                'timing': self.scenario_config.meal_timing,
                'absorption_time': self.scenario_config.absorption_time,
                'type': 'unannounced'
            })
        
        return meal_scenarios
    
    def _generate_settings_mismatches(self) -> List[Dict[str, float]]:
        """Generate settings mismatch combinations."""
        settings_combinations = []
        
        # Generate all combinations of multipliers for each parameter
        multipliers = self.scenario_config.settings_multipliers
        apply_to = self.scenario_config.settings_apply_to
        
        # Create all combinations
        for combo in itertools.product(multipliers, repeat=len(apply_to)):
            settings_dict = dict(zip(apply_to, combo))
            settings_combinations.append(settings_dict)
        
        return settings_combinations
    
    def _estimate_total_scenarios(
        self,
        patient_configs: List[Dict[str, Any]],
        algorithms: List[str]
    ) -> int:
        """Estimate total number of scenarios."""
        
        num_patients = len(patient_configs)
        num_initial_bg = len(self._generate_initial_bg_values())
        num_meals = len(self._generate_meal_scenarios())
        num_settings = len(self._generate_settings_mismatches())
        
        total = 0
        
        for algorithm in algorithms:
            algorithm_config = self.config.get_algorithm_config(algorithm)
            
            if algorithm == 'tempbasal':
                num_paf = 1  # No partial application factor
            else:
                num_paf = len(algorithm_config.partial_application_factors)
            
            num_gradual_thresholds = len(algorithm_config.gradual_transition_thresholds)
            
            algorithm_scenarios = num_patients * num_initial_bg * num_meals * num_settings * num_paf * num_gradual_thresholds
            total += algorithm_scenarios
        
        return total
    
    def generate_paired_scenarios(
        self,
        patient_configs: List[Dict[str, Any]],
        reference_algorithm: str = 'tempbasal',
        comparison_algorithms: Optional[List[str]] = None
    ) -> Iterator[Tuple[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Generate paired scenarios for direct comparison.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            reference_algorithm: Reference algorithm for comparison
            comparison_algorithms: List of algorithms to compare against reference
            
        Yields:
            Tuple of (reference_scenario, list_of_comparison_scenarios)
        """
        if comparison_algorithms is None:
            comparison_algorithms = [alg for alg in self.config.get_enabled_algorithms() 
                                   if alg != reference_algorithm]
        
        logger.info(f"Generating paired scenarios: {reference_algorithm} vs {comparison_algorithms}")
        
        for patient_config in patient_configs:
            for true_start_bg in self._generate_initial_bg_values():
                for meal_scenario in self._generate_meal_scenarios():
                    for settings_multipliers in self._generate_settings_mismatches():
                        
                        # Create reference scenario
                        reference_scenario = {
                            'algorithm_type': reference_algorithm,
                            'patient_config': patient_config,
                            'true_start_bg': true_start_bg,
                            'meal_scenario': meal_scenario,
                            'settings_multipliers': settings_multipliers,
                            'partial_application_factor': None
                        }
                        
                        # Create comparison scenarios
                        comparison_scenarios = []
                        
                        for algorithm in comparison_algorithms:
                            algorithm_config = self.config.get_algorithm_config(algorithm)
                            
                            if algorithm == 'tempbasal':
                                comparison_scenarios.append({
                                    'algorithm_type': algorithm,
                                    'patient_config': patient_config,
                                    'true_start_bg': true_start_bg,
                                    'meal_scenario': meal_scenario,
                                    'settings_multipliers': settings_multipliers,
                                    'partial_application_factor': None
                                })
                            
                            elif algorithm == 'autobolus':
                                for paf in algorithm_config.partial_application_factors:
                                    comparison_scenarios.append({
                                        'algorithm_type': algorithm,
                                        'patient_config': patient_config,
                                        'true_start_bg': true_start_bg,
                                        'meal_scenario': meal_scenario,
                                        'settings_multipliers': settings_multipliers,
                                        'partial_application_factor': paf
                                    })
                        
                        yield reference_scenario, comparison_scenarios
    
    def generate_scenarios_dataframe(
        self,
        patient_configs: List[Dict[str, Any]],
        algorithms: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Generate all scenarios as a pandas DataFrame.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            algorithms: List of algorithms to test (None for all enabled)
            
        Returns:
            DataFrame with scenario parameters
        """
        scenarios = list(self.generate_all_scenarios(patient_configs, algorithms))
        
        # Flatten scenarios for DataFrame
        flattened_scenarios = []
        
        for scenario in scenarios:
            flat_scenario = {
                'algorithm_type': scenario['algorithm_type'],
                'patient_id': scenario['patient_config'].get('patient_id', 'unknown'),
                'true_start_bg': scenario['true_start_bg'],
                'meal_size': scenario['meal_scenario']['size'],
                'meal_timing': scenario['meal_scenario']['timing'],
                'meal_absorption_time': scenario['meal_scenario']['absorption_time'],
                'partial_application_factor': scenario.get('partial_application_factor'),
                'gradual_transition_threshold': scenario.get('gradual_transition_threshold'),
            }
            
            # Add settings multipliers
            if scenario.get('settings_multipliers'):
                for param, multiplier in scenario['settings_multipliers'].items():
                    flat_scenario[f'{param}_multiplier'] = multiplier
            
            flattened_scenarios.append(flat_scenario)
        
        return pd.DataFrame(flattened_scenarios)
    
    def filter_scenarios_by_criteria(
        self,
        scenarios: Iterator[Dict[str, Any]],
        criteria: Dict[str, Any]
    ) -> Iterator[Dict[str, Any]]:
        """
        Filter scenarios based on criteria.
        
        Args:
            scenarios: Iterator of scenario dictionaries
            criteria: Dictionary of filtering criteria
            
        Yields:
            Filtered scenario dictionaries
        """
        for scenario in scenarios:
            include = True
            
            for key, value in criteria.items():
                if key in scenario:
                    if isinstance(value, list):
                        if scenario[key] not in value:
                            include = False
                            break
                    else:
                        if scenario[key] != value:
                            include = False
                            break
                elif key in scenario.get('settings_multipliers', {}):
                    if isinstance(value, list):
                        if scenario['settings_multipliers'][key] not in value:
                            include = False
                            break
                    else:
                        if scenario['settings_multipliers'][key] != value:
                            include = False
                            break
            
            if include:
                yield scenario
    
    def sample_scenarios(
        self,
        scenarios: Iterator[Dict[str, Any]],
        n_samples: int,
        random_seed: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Randomly sample scenarios.
        
        Args:
            scenarios: Iterator of scenario dictionaries
            n_samples: Number of scenarios to sample
            random_seed: Random seed for reproducibility
            
        Returns:
            List of sampled scenario dictionaries
        """
        if random_seed is not None:
            np.random.seed(random_seed)
        
        # Convert to list first
        scenario_list = list(scenarios)
        
        if len(scenario_list) <= n_samples:
            return scenario_list
        
        # Random sampling without replacement
        indices = np.random.choice(len(scenario_list), size=n_samples, replace=False)
        
        return [scenario_list[i] for i in indices]
    
    def get_scenario_summary(
        self,
        patient_configs: List[Dict[str, Any]],
        algorithms: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for scenario generation.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            algorithms: List of algorithms to test (None for all enabled)
            
        Returns:
            Dictionary with summary statistics
        """
        if algorithms is None:
            algorithms = self.config.get_enabled_algorithms()
        
        summary = {
            'num_patients': len(patient_configs),
            'algorithms': algorithms,
            'initial_bg_range': self.scenario_config.initial_bg_range,
            'initial_bg_step': self.scenario_config.initial_bg_step,
            'num_initial_bg_values': len(self._generate_initial_bg_values()),
            'meal_scenarios': self.scenario_config.unannounced_meals,
            'num_meal_scenarios': len(self._generate_meal_scenarios()),
            'settings_multipliers': self.scenario_config.settings_multipliers,
            'settings_apply_to': self.scenario_config.settings_apply_to,
            'num_settings_combinations': len(self._generate_settings_mismatches()),
            'estimated_total_scenarios': self._estimate_total_scenarios(patient_configs, algorithms)
        }
        
        # Add algorithm-specific details
        for algorithm in algorithms:
            algorithm_config = self.config.get_algorithm_config(algorithm)
            summary[f'{algorithm}_gradual_transition_thresholds'] = algorithm_config.gradual_transition_thresholds
            if algorithm == 'autobolus':
                summary[f'{algorithm}_partial_application_factors'] = algorithm_config.partial_application_factors
        
        return summary
    
    # ========================================================================
    # iCGM Sensitivity Analysis Scenarios
    # ========================================================================
    
    def generate_icgm_scenarios(
        self,
        patient_configs: List[Dict[str, Any]],
        true_bg_range: Tuple[int, int, int] = (40, 405, 5),
        sensor_bg_range: Optional[Tuple[int, int, int]] = None,
        algorithm: str = 'autobolus',
        sensor_model_type: str = 'NoisySensorInitialOffset',
        sensor_std_dev: float = 3.0
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate iCGM sensitivity analysis scenarios.
        
        Creates a grid of (true BG, sensor BG) combinations to test how the
        algorithm handles spurious sensor errors. Each scenario represents a
        single point in the grid with a specific sensor error.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            true_bg_range: (start, end, step) for true BG values in mg/dL
            sensor_bg_range: (start, end, step) for sensor BG values. 
                           If None, uses same as true_bg_range
            algorithm: Algorithm to test (typically 'autobolus')
            sensor_model_type: 'NoisySensorInitialOffset' or 'SensoriCGMInitialOffset'
            sensor_std_dev: Standard deviation for sensor noise model
            
        Yields:
            Dictionary containing iCGM scenario parameters
            
        Example:
            >>> generator = ScenarioGenerator(config)
            >>> scenarios = generator.generate_icgm_scenarios(
            ...     patient_configs,
            ...     true_bg_range=(40, 405, 5),
            ...     sensor_bg_range=(40, 405, 5)
            ... )
        """
        if sensor_bg_range is None:
            sensor_bg_range = true_bg_range
        
        algorithm_config = self.config.get_algorithm_config(algorithm)
        
        # Generate BG grid
        true_bg_values, sensor_bg_values = self._generate_icgm_bg_grid(
            true_bg_range, sensor_bg_range
        )
        
        total_scenarios = (
            len(patient_configs) * 
            len(true_bg_values) * 
            len(sensor_bg_values) *
            len(algorithm_config.partial_application_factors) *
            len(algorithm_config.gradual_transition_thresholds)
        )
        
        logger.info(
            f"Generating iCGM scenarios: {total_scenarios} total "
            f"({len(patient_configs)} patients × {len(true_bg_values)} true BG × "
            f"{len(sensor_bg_values)} sensor BG × "
            f"{len(algorithm_config.partial_application_factors)} PAF × "
            f"{len(algorithm_config.gradual_transition_thresholds)} gradual thresholds)"
        )
        
        scenario_count = 0
        
        for patient_config in patient_configs:
            for true_bg in true_bg_values:
                for sensor_bg in sensor_bg_values:
                    for paf in algorithm_config.partial_application_factors:
                        for gradual_threshold in algorithm_config.gradual_transition_thresholds:
                            
                            scenario = {
                                'scenario_type': 'icgm_sensitivity',
                                'algorithm_type': algorithm,
                                'patient_config': patient_config,
                                
                                # iCGM-specific parameters
                                'true_start_bg': true_bg,
                                'sensor_start_bg': sensor_bg,
                                'sensor_error': sensor_bg - true_bg,
                                'sensor_model_type': sensor_model_type,
                                'sensor_std_dev': sensor_std_dev,
                                
                                # Algorithm parameters
                                'partial_application_factor': paf,
                                'gradual_transition_threshold': gradual_threshold,
                                
                                # No meals or settings mismatches for iCGM
                                'meal_scenario': None,
                                'settings_multipliers': None,
                                
                                # Bolus acceptance only at t0
                                'bolus_acceptance_mode': 't0_only'
                            }
                            
                            scenario_count += 1
                            if scenario_count % 10000 == 0:
                                logger.debug(f"Generated {scenario_count} iCGM scenarios")
                            
                            yield scenario
        
        logger.info(f"Generated {scenario_count} total iCGM scenarios")
    
    def _generate_icgm_bg_grid(
        self,
        true_bg_range: Tuple[int, int, int],
        sensor_bg_range: Tuple[int, int, int]
    ) -> Tuple[List[int], List[int]]:
        """
        Generate BG grid for iCGM sensitivity analysis.
        
        Args:
            true_bg_range: (start, end, step) for true BG values
            sensor_bg_range: (start, end, step) for sensor BG values
            
        Returns:
            Tuple of (true_bg_values, sensor_bg_values)
        """
        true_start, true_end, true_step = true_bg_range
        sensor_start, sensor_end, sensor_step = sensor_bg_range
        
        true_bg_values = list(range(true_start, true_end, true_step))
        sensor_bg_values = list(range(sensor_start, sensor_end, sensor_step))
        
        logger.info(
            f"Generated iCGM BG grid: {len(true_bg_values)} true BG values "
            f"({true_start}-{true_end-true_step} by {true_step}), "
            f"{len(sensor_bg_values)} sensor BG values "
            f"({sensor_start}-{sensor_end-sensor_step} by {sensor_step})"
        )
        
        return true_bg_values, sensor_bg_values
    
    def generate_icgm_scenarios_for_mitigation_testing(
        self,
        patient_configs: List[Dict[str, Any]],
        true_bg_range: Tuple[int, int, int] = (40, 405, 5),
        sensor_bg_range: Optional[Tuple[int, int, int]] = None,
        mitigation_thresholds: List[float] = [20.0, 30.0, 40.0],
        include_unmitigated: bool = True,
        paf: float = 0.4
    ) -> Iterator[Dict[str, Any]]:
        """
        Generate iCGM scenarios specifically for testing mitigation strategies.
        
        This creates scenarios to compare different gradual transition threshold
        values (mitigation) against unmitigated baseline.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            true_bg_range: (start, end, step) for true BG values
            sensor_bg_range: (start, end, step) for sensor BG values
            mitigation_thresholds: List of gradual transition thresholds to test
            include_unmitigated: If True, also generate unmitigated scenarios
            paf: Partial application factor to use
            
        Yields:
            Dictionary containing iCGM scenario parameters
        """
        algorithm_config = self.config.get_algorithm_config('autobolus')
        
        # Override thresholds with provided values
        test_thresholds = mitigation_thresholds.copy()
        if include_unmitigated:
            test_thresholds.append(10000.0)  # Very high threshold = unmitigated
        
        true_bg_values, sensor_bg_values = self._generate_icgm_bg_grid(
            true_bg_range, sensor_bg_range or true_bg_range
        )
        
        logger.info(
            f"Generating iCGM mitigation scenarios: "
            f"{len(test_thresholds)} threshold values "
            f"(mitigated: {mitigation_thresholds}, "
            f"unmitigated: {include_unmitigated})"
        )
        
        for patient_config in patient_configs:
            for true_bg in true_bg_values:
                for sensor_bg in sensor_bg_values:
                    for threshold in test_thresholds:
                        
                        yield {
                            'scenario_type': 'icgm_mitigation',
                            'algorithm_type': 'autobolus',
                            'patient_config': patient_config,
                            'true_start_bg': true_bg,
                            'sensor_start_bg': sensor_bg,
                            'sensor_error': sensor_bg - true_bg,
                            'sensor_model_type': 'NoisySensorInitialOffset',
                            'sensor_std_dev': 3.0,
                            'partial_application_factor': paf,
                            'gradual_transition_threshold': threshold,
                            'is_mitigated': threshold < 1000.0,
                            'meal_scenario': None,
                            'settings_multipliers': None,
                            'bolus_acceptance_mode': 't0_only'
                        }
    
    def generate_icgm_scenarios_dataframe(
        self,
        patient_configs: List[Dict[str, Any]],
        **kwargs
    ) -> pd.DataFrame:
        """
        Generate iCGM scenarios as a pandas DataFrame.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            **kwargs: Arguments to pass to generate_icgm_scenarios()
            
        Returns:
            DataFrame with iCGM scenario parameters
        """
        scenarios = list(self.generate_icgm_scenarios(patient_configs, **kwargs))
        
        flattened_scenarios = []
        for scenario in scenarios:
            flat_scenario = {
                'scenario_type': scenario['scenario_type'],
                'algorithm_type': scenario['algorithm_type'],
                'patient_id': scenario['patient_config'].get('patient_id', 'unknown'),
                'true_start_bg': scenario['true_start_bg'],
                'sensor_start_bg': scenario['sensor_start_bg'],
                'sensor_error': scenario['sensor_error'],
                'sensor_model_type': scenario['sensor_model_type'],
                'partial_application_factor': scenario['partial_application_factor'],
                'gradual_transition_threshold': scenario['gradual_transition_threshold'],
                'bolus_acceptance_mode': scenario['bolus_acceptance_mode']
            }
            flattened_scenarios.append(flat_scenario)
        
        return pd.DataFrame(flattened_scenarios)
    
    def get_icgm_scenario_summary(
        self,
        patient_configs: List[Dict[str, Any]],
        true_bg_range: Tuple[int, int, int] = (40, 405, 5),
        sensor_bg_range: Optional[Tuple[int, int, int]] = None
    ) -> Dict[str, Any]:
        """
        Get summary statistics for iCGM scenario generation.
        
        Args:
            patient_configs: List of patient configuration dictionaries
            true_bg_range: (start, end, step) for true BG values
            sensor_bg_range: (start, end, step) for sensor BG values
            
        Returns:
            Dictionary with summary statistics
        """
        if sensor_bg_range is None:
            sensor_bg_range = true_bg_range
        
        true_bg_values, sensor_bg_values = self._generate_icgm_bg_grid(
            true_bg_range, sensor_bg_range
        )
        
        algorithm_config = self.config.get_algorithm_config('autobolus')
        
        total_scenarios = (
            len(patient_configs) * 
            len(true_bg_values) * 
            len(sensor_bg_values) *
            len(algorithm_config.partial_application_factors) *
            len(algorithm_config.gradual_transition_thresholds)
        )
        
        return {
            'scenario_type': 'icgm_sensitivity',
            'num_patients': len(patient_configs),
            'true_bg_range': true_bg_range,
            'num_true_bg_values': len(true_bg_values),
            'sensor_bg_range': sensor_bg_range,
            'num_sensor_bg_values': len(sensor_bg_values),
            'grid_size': len(true_bg_values) * len(sensor_bg_values),
            'partial_application_factors': algorithm_config.partial_application_factors,
            'gradual_transition_thresholds': algorithm_config.gradual_transition_thresholds,
            'estimated_total_scenarios': total_scenarios
        }
