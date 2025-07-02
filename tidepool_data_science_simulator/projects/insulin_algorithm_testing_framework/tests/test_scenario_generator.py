#!/usr/bin/env python3
"""
Unit tests for scenario generator module.

Tests the ScenarioGenerator class functionality including scenario generation,
filtering, and parameter combinations.
"""

import unittest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch
import sys
from pathlib import Path

# Add framework to path
framework_path = Path(__file__).parent.parent
sys.path.insert(0, str(framework_path))

from config.experiment_config import ExperimentConfig
from core.scenario_generator import ScenarioGenerator


class TestScenarioGenerator(unittest.TestCase):
    """Test cases for ScenarioGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create mock configuration
        self.config = ExperimentConfig()
        
        # Set test configuration values
        self.config.set('scenarios.initial_bg.range', [100, 150])
        self.config.set('scenarios.initial_bg.step', 25)
        self.config.set('scenarios.meal_scenarios.unannounced_meals', [30, 60])
        self.config.set('scenarios.meal_scenarios.meal_timing', 120)
        self.config.set('scenarios.meal_scenarios.absorption_time', 180)
        self.config.set('scenarios.settings_mismatches.multipliers', [0.8, 1.0, 1.2])
        self.config.set('scenarios.settings_mismatches.apply_to', ['isf', 'cir'])
        self.config.set('algorithms.temp_basal.enabled', True)
        self.config.set('algorithms.autobolus.enabled', True)
        self.config.set('algorithms.autobolus.partial_application_factors', [0.4, 0.6])
        
        self.generator = ScenarioGenerator(self.config)
        
        # Create test patient configurations
        self.patient_configs = [
            {'patient_id': 'patient_1', 'weight': 70, 'isf': 50, 'cir': 15},
            {'patient_id': 'patient_2', 'weight': 80, 'isf': 45, 'cir': 12}
        ]
    
    def test_initialization(self):
        """Test ScenarioGenerator initialization."""
        self.assertIsInstance(self.generator, ScenarioGenerator)
        self.assertEqual(self.generator.config, self.config)
        self.assertIsNotNone(self.generator.scenario_config)
    
    def test_generate_initial_bg_values(self):
        """Test initial BG value generation."""
        bg_values = self.generator._generate_initial_bg_values()
        
        expected_values = [100, 125, 150]
        self.assertEqual(bg_values, expected_values)
    
    def test_generate_meal_scenarios(self):
        """Test meal scenario generation."""
        meal_scenarios = self.generator._generate_meal_scenarios()
        
        self.assertEqual(len(meal_scenarios), 2)
        
        for scenario in meal_scenarios:
            self.assertIn('size', scenario)
            self.assertIn('timing', scenario)
            self.assertIn('absorption_time', scenario)
            self.assertIn('type', scenario)
            self.assertEqual(scenario['type'], 'unannounced')
            self.assertIn(scenario['size'], [30, 60])
    
    def test_generate_settings_mismatches(self):
        """Test settings mismatch generation."""
        settings_combinations = self.generator._generate_settings_mismatches()
        
        # Should have 3^2 = 9 combinations (3 multipliers for 2 parameters)
        self.assertEqual(len(settings_combinations), 9)
        
        for combo in settings_combinations:
            self.assertIn('isf', combo)
            self.assertIn('cir', combo)
            self.assertIn(combo['isf'], [0.8, 1.0, 1.2])
            self.assertIn(combo['cir'], [0.8, 1.0, 1.2])
    
    def test_generate_scenarios_for_temp_basal(self):
        """Test scenario generation for temp basal algorithm."""
        scenarios = list(self.generator.generate_scenarios_for_algorithm(
            self.patient_configs, 'temp_basal'
        ))
        
        # Calculate expected number of scenarios
        # 2 patients * 3 initial_bg * 2 meals * 9 settings = 108 scenarios
        expected_count = 2 * 3 * 2 * 9
        self.assertEqual(len(scenarios), expected_count)
        
        # Check scenario structure
        for scenario in scenarios[:5]:  # Check first 5
            self.assertEqual(scenario['algorithm_type'], 'temp_basal')
            self.assertIn(scenario['patient_config'], self.patient_configs)
            self.assertIn(scenario['initial_bg'], [100, 125, 150])
            self.assertIsNone(scenario['partial_application_factor'])
    
    def test_generate_scenarios_for_autobolus(self):
        """Test scenario generation for autobolus algorithm."""
        scenarios = list(self.generator.generate_scenarios_for_algorithm(
            self.patient_configs, 'autobolus'
        ))
        
        # Calculate expected number of scenarios
        # 2 patients * 3 initial_bg * 2 meals * 9 settings * 2 PAF = 216 scenarios
        expected_count = 2 * 3 * 2 * 9 * 2
        self.assertEqual(len(scenarios), expected_count)
        
        # Check scenario structure
        for scenario in scenarios[:5]:  # Check first 5
            self.assertEqual(scenario['algorithm_type'], 'autobolus')
            self.assertIn(scenario['patient_config'], self.patient_configs)
            self.assertIn(scenario['initial_bg'], [100, 125, 150])
            self.assertIn(scenario['partial_application_factor'], [0.4, 0.6])
    
    def test_generate_all_scenarios(self):
        """Test generation of all scenarios."""
        scenarios = list(self.generator.generate_all_scenarios(self.patient_configs))
        
        # Should include both temp_basal and autobolus scenarios
        # 108 (temp_basal) + 216 (autobolus) = 324 total
        expected_count = 108 + 216
        self.assertEqual(len(scenarios), expected_count)
        
        # Check algorithm distribution
        temp_basal_count = sum(1 for s in scenarios if s['algorithm_type'] == 'temp_basal')
        autobolus_count = sum(1 for s in scenarios if s['algorithm_type'] == 'autobolus')
        
        self.assertEqual(temp_basal_count, 108)
        self.assertEqual(autobolus_count, 216)
    
    def test_generate_paired_scenarios(self):
        """Test paired scenario generation."""
        paired_scenarios = list(self.generator.generate_paired_scenarios(
            self.patient_configs,
            reference_algorithm='temp_basal',
            comparison_algorithms=['autobolus']
        ))
        
        # Should have one pair for each combination of patient, initial_bg, meal, settings
        # 2 patients * 3 initial_bg * 2 meals * 9 settings = 108 pairs
        expected_pairs = 2 * 3 * 2 * 9
        self.assertEqual(len(paired_scenarios), expected_pairs)
        
        # Check pair structure
        for ref_scenario, comp_scenarios in paired_scenarios[:3]:  # Check first 3
            self.assertEqual(ref_scenario['algorithm_type'], 'temp_basal')
            self.assertIsNone(ref_scenario['partial_application_factor'])
            
            # Should have 2 comparison scenarios (one for each PAF)
            self.assertEqual(len(comp_scenarios), 2)
            for comp_scenario in comp_scenarios:
                self.assertEqual(comp_scenario['algorithm_type'], 'autobolus')
                self.assertIn(comp_scenario['partial_application_factor'], [0.4, 0.6])
    
    def test_filter_scenarios_by_criteria(self):
        """Test scenario filtering."""
        scenarios = list(self.generator.generate_all_scenarios(self.patient_configs))
        
        # Filter for specific algorithm
        criteria = {'algorithm_type': 'temp_basal'}
        filtered = list(self.generator.filter_scenarios_by_criteria(scenarios, criteria))
        
        self.assertTrue(all(s['algorithm_type'] == 'temp_basal' for s in filtered))
        self.assertEqual(len(filtered), 108)
        
        # Filter for specific initial BG
        criteria = {'initial_bg': 125}
        filtered = list(self.generator.filter_scenarios_by_criteria(scenarios, criteria))
        
        self.assertTrue(all(s['initial_bg'] == 125 for s in filtered))
        
        # Filter for multiple criteria
        criteria = {'algorithm_type': 'autobolus', 'initial_bg': 100}
        filtered = list(self.generator.filter_scenarios_by_criteria(scenarios, criteria))
        
        self.assertTrue(all(
            s['algorithm_type'] == 'autobolus' and s['initial_bg'] == 100 
            for s in filtered
        ))
    
    def test_sample_scenarios(self):
        """Test scenario sampling."""
        scenarios = list(self.generator.generate_all_scenarios(self.patient_configs))
        
        # Sample 50 scenarios
        sampled = self.generator.sample_scenarios(scenarios, n_samples=50, random_seed=42)
        
        self.assertEqual(len(sampled), 50)
        self.assertTrue(all(s in scenarios for s in sampled))
        
        # Test with more samples than available
        all_sampled = self.generator.sample_scenarios(scenarios, n_samples=1000, random_seed=42)
        self.assertEqual(len(all_sampled), len(scenarios))
    
    def test_generate_scenarios_dataframe(self):
        """Test scenario DataFrame generation."""
        df = self.generator.generate_scenarios_dataframe(self.patient_configs)
        
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 324)  # Total expected scenarios
        
        # Check required columns
        required_columns = [
            'algorithm_type', 'patient_id', 'initial_bg', 'meal_size',
            'meal_timing', 'meal_absorption_time', 'partial_application_factor'
        ]
        
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Check settings multiplier columns
        self.assertIn('isf_multiplier', df.columns)
        self.assertIn('cir_multiplier', df.columns)
    
    def test_get_scenario_summary(self):
        """Test scenario summary generation."""
        summary = self.generator.get_scenario_summary(self.patient_configs)
        
        self.assertIsInstance(summary, dict)
        
        # Check required summary fields
        required_fields = [
            'num_patients', 'algorithms', 'initial_bg_range', 'initial_bg_step',
            'num_initial_bg_values', 'meal_scenarios', 'num_meal_scenarios',
            'settings_multipliers', 'settings_apply_to', 'num_settings_combinations',
            'estimated_total_scenarios'
        ]
        
        for field in required_fields:
            self.assertIn(field, summary)
        
        # Check values
        self.assertEqual(summary['num_patients'], 2)
        self.assertEqual(summary['algorithms'], ['temp_basal', 'autobolus'])
        self.assertEqual(summary['estimated_total_scenarios'], 324)
    
    def test_estimate_total_scenarios(self):
        """Test total scenario estimation."""
        algorithms = ['temp_basal', 'autobolus']
        total = self.generator._estimate_total_scenarios(self.patient_configs, algorithms)
        
        # 2 patients * 3 initial_bg * 2 meals * 9 settings * (1 + 2) algorithms
        # temp_basal: 2 * 3 * 2 * 9 * 1 = 108
        # autobolus: 2 * 3 * 2 * 9 * 2 = 216
        # total: 324
        expected_total = 324
        self.assertEqual(total, expected_total)
    
    def test_invalid_algorithm(self):
        """Test handling of invalid algorithm."""
        with self.assertRaises(Exception):
            list(self.generator.generate_scenarios_for_algorithm(
                self.patient_configs, 'invalid_algorithm'
            ))
    
    def test_empty_patient_configs(self):
        """Test handling of empty patient configurations."""
        scenarios = list(self.generator.generate_all_scenarios([]))
        self.assertEqual(len(scenarios), 0)
        
        summary = self.generator.get_scenario_summary([])
        self.assertEqual(summary['num_patients'], 0)
        self.assertEqual(summary['estimated_total_scenarios'], 0)


class TestScenarioGeneratorIntegration(unittest.TestCase):
    """Integration tests for ScenarioGenerator."""
    
    def setUp(self):
        """Set up integration test fixtures."""
        self.config = ExperimentConfig()
        self.generator = ScenarioGenerator(self.config)
        
        self.patient_configs = [
            {'patient_id': 'test_patient', 'weight': 70, 'isf': 50, 'cir': 15}
        ]
    
    def test_end_to_end_scenario_generation(self):
        """Test complete scenario generation workflow."""
        # Generate scenarios
        scenarios = list(self.generator.generate_all_scenarios(self.patient_configs))
        
        self.assertGreater(len(scenarios), 0)
        
        # Convert to DataFrame
        df = self.generator.generate_scenarios_dataframe(self.patient_configs)
        self.assertEqual(len(df), len(scenarios))
        
        # Filter scenarios
        temp_basal_scenarios = list(self.generator.filter_scenarios_by_criteria(
            scenarios, {'algorithm_type': 'temp_basal'}
        ))
        
        self.assertGreater(len(temp_basal_scenarios), 0)
        self.assertTrue(all(s['algorithm_type'] == 'temp_basal' for s in temp_basal_scenarios))
        
        # Sample scenarios
        sampled = self.generator.sample_scenarios(scenarios, n_samples=10, random_seed=42)
        self.assertLessEqual(len(sampled), min(10, len(scenarios)))
        
        # Get summary
        summary = self.generator.get_scenario_summary(self.patient_configs)
        self.assertEqual(summary['estimated_total_scenarios'], len(scenarios))


if __name__ == '__main__':
    unittest.main()
