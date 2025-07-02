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
        for initial_bg in self._generate_initial_bg_values():
            for meal_scenario in self._generate_meal_scenarios():
                for settings_multipliers in self._generate_settings_mismatches():
                    
                    if algorithm == 'tempbasal':
                        # Temp basal doesn't use partial application factor
                        yield {
                            'algorithm_type': algorithm,
                            'patient_config': patient_config,
                            'initial_bg': initial_bg,
                            'meal_scenario': meal_scenario,
                            'settings_multipliers': settings_multipliers,
                            'partial_application_factor': None
                        }
                    
                    elif algorithm == 'autobolus':
                        # Generate scenarios for each partial application factor
                        for paf in algorithm_config.partial_application_factors:
                            yield {
                                'algorithm_type': algorithm,
                                'patient_config': patient_config,
                                'initial_bg': initial_bg,
                                'meal_scenario': meal_scenario,
                                'settings_multipliers': settings_multipliers,
                                'partial_application_factor': paf
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
            
            algorithm_scenarios = num_patients * num_initial_bg * num_meals * num_settings * num_paf
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
            for initial_bg in self._generate_initial_bg_values():
                for meal_scenario in self._generate_meal_scenarios():
                    for settings_multipliers in self._generate_settings_mismatches():
                        
                        # Create reference scenario
                        reference_scenario = {
                            'algorithm_type': reference_algorithm,
                            'patient_config': patient_config,
                            'initial_bg': initial_bg,
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
                                    'initial_bg': initial_bg,
                                    'meal_scenario': meal_scenario,
                                    'settings_multipliers': settings_multipliers,
                                    'partial_application_factor': None
                                })
                            
                            elif algorithm == 'autobolus':
                                for paf in algorithm_config.partial_application_factors:
                                    comparison_scenarios.append({
                                        'algorithm_type': algorithm,
                                        'patient_config': patient_config,
                                        'initial_bg': initial_bg,
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
                'initial_bg': scenario['initial_bg'],
                'meal_size': scenario['meal_scenario']['size'],
                'meal_timing': scenario['meal_scenario']['timing'],
                'meal_absorption_time': scenario['meal_scenario']['absorption_time'],
                'partial_application_factor': scenario.get('partial_application_factor'),
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
            if algorithm == 'autobolus':
                summary[f'{algorithm}_partial_application_factors'] = algorithm_config.partial_application_factors
        
        return summary
