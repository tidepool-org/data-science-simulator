#!/usr/bin/env python3
"""
Unit tests for Algorithm Risk Profile Comparison Analysis

Tests cover data preprocessing, statistical calculations, and acceptance criteria evaluation.
"""

__author__ = "Shawn Foster"

import os
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from post_processing.algorithm_risk_comparison import (
    parse_sim_id,
    parse_matching_key,
    format_unmatched_scenarios,
    create_matching_key,
    validate_pairwise_matching,
    export_unmatched_scenarios_to_csv,
    merge_paired_data,
    calc_range_difference,
    calc_mean_difference,
    calc_wilcoxon_signed_rank,
    calc_cohens_d_paired,
    calc_paired_ttest,
    calc_std_difference,
    calc_median_difference,
    calculate_hyperglycemia_score,
    determine_harm_and_severity,
    aggregate_risk_by_scenario_type,
    detect_severity_threshold_crossing_aggregated,
    detect_harm_type_change_aggregated,
    evaluate_acceptance_criteria,
    run_question1_analysis,
    run_question2_analysis,
    run_question3_analysis,
    round_half_up,
    # Module 9
    parse_probability,
    compute_risk_score,
    get_acceptability,
    acceptability_rank,
    enrich_with_jira_data,
    classify_hazard_rows,
    assign_hazard_rows_to_sheets,
    compute_hazard_summary_stats,
    write_hazardous_situation_comparison_excel,
    PATIENT_PROFILES,
    METRICS
)


class TestRoundHalfUp(unittest.TestCase):
    """Tests for round_half_up rounding consistency with create_severity_summary.py."""

    def test_rounds_half_up_not_to_even(self):
        """0.5 boundaries must always round UP, not to nearest even."""
        # Banker's rounding: 0.5->0, 1.5->2, 2.5->2  (WRONG for severity)
        # round_half_up:     0.5->1, 1.5->2, 2.5->3  (conservative)
        self.assertEqual(round_half_up(0.5), 1)
        self.assertEqual(round_half_up(1.5), 2)
        self.assertEqual(round_half_up(2.5), 3)
        self.assertEqual(round_half_up(3.5), 4)

    def test_normal_rounding(self):
        """Values away from 0.5 boundaries round as expected."""
        self.assertEqual(round_half_up(0.4), 0)
        self.assertEqual(round_half_up(0.6), 1)
        self.assertEqual(round_half_up(2.4), 2)
        self.assertEqual(round_half_up(2.6), 3)

    def test_aggregate_uses_round_half_up(self):
        """aggregate_risk_by_scenario_type must use round_half_up for score columns."""
        # Construct a group whose mean LBGI score is exactly 2.5.
        # Banker's rounding -> 2; round_half_up -> 3 (conservative).
        df = pd.DataFrame({
            'matching_key': ['k1', 'k2'],
            'risk_name':    ['TLR-001', 'TLR-001'],
            'scenario_type': ['pre-mitigation', 'pre-mitigation'],
            'patient_profile': ['adolescent', 'median'],
            'lbgi_risk_score_v1': [2.0, 3.0],   # mean = 2.5
            'dka_risk_score_v1':  [1.0, 2.0],   # mean = 1.5
            'percent_cgm_gt_180_v1': [5.0, 5.0],
            # v2 columns required by merge but not used here
            'lbgi_risk_score_v2': [0.0, 0.0],
            'dka_risk_score_v2':  [0.0, 0.0],
            'percent_cgm_gt_180_v2': [0.0, 0.0],
        })

        agg = aggregate_risk_by_scenario_type(df, '_v1')
        row = agg[agg['risk_name'] == 'TLR-001'].iloc[0]

        # round_half_up(2.5) == 3, not 2
        self.assertEqual(row['lbgi_avg'], 3,
            'LBGI avg of 2.5 should round UP to 3 (conservative), not down to 2')
        # round_half_up(1.5) == 2, not 2  (same result here, but verifies path)
        self.assertEqual(row['dka_avg'], 2)



class TestSimIdParsing(unittest.TestCase):
    """Tests for sim_id parsing functionality."""

    def test_parse_pre_mitigation_underscore(self):
        """Test parsing pre-mitigation with underscore format."""
        result = parse_sim_id('pre-Loop_NoMitigations_t1_adolescent')
        self.assertEqual(result['scenario_type'], 'pre-mitigation')
        self.assertEqual(result['patient_profile'], 'adolescent')
    
    def test_parse_pre_mitigation_dash(self):
        """Test parsing pre-mitigation with dash format."""
        result = parse_sim_id('pre-Loop-NoMitigations_t1_sensitive')
        self.assertEqual(result['scenario_type'], 'pre-mitigation')
        self.assertEqual(result['patient_profile'], 'sensitive')
    
    def test_parse_pre_mitigation_lowercase(self):
        """Test parsing pre-mitigation with lowercase format."""
        result = parse_sim_id('pre-Loop-noMitigations_t1_resistant')
        self.assertEqual(result['scenario_type'], 'pre-mitigation')
        self.assertEqual(result['patient_profile'], 'resistant')
    
    def test_parse_noloop_lowercase(self):
        """Test parsing noLoop scenario with lowercase."""
        result = parse_sim_id('pre-noLoop_t1_median')
        self.assertEqual(result['scenario_type'], 'noLoop')
        self.assertEqual(result['patient_profile'], 'median')
    
    def test_parse_noloop_uppercase(self):
        """Test parsing noLoop scenario with uppercase."""
        result = parse_sim_id('pre-NoLoop_t1_adolescent')
        self.assertEqual(result['scenario_type'], 'noLoop')
        self.assertEqual(result['patient_profile'], 'adolescent')
    
    def test_parse_post_mitigation(self):
        """Test parsing post-mitigation scenario."""
        result = parse_sim_id('post-Loop-WithMitigations_t1_sensitive')
        self.assertEqual(result['scenario_type'], 'post-mitigation')
        self.assertEqual(result['patient_profile'], 'sensitive')
    
    def test_parse_all_profiles(self):
        """Test that all patient profiles are correctly identified."""
        for profile in PATIENT_PROFILES:
            result = parse_sim_id(f'pre-Loop_NoMitigations_t1_{profile}')
            self.assertEqual(result['patient_profile'], profile)
    
    def test_parse_resistant_typo(self):
        """Test that 'resistnat' typo is correctly mapped to 'resistant'."""
        result = parse_sim_id('post-Loop-WithMitigations_t1_resistnat')
        self.assertEqual(result['patient_profile'], 'resistant')
        self.assertEqual(result['scenario_type'], 'post-mitigation')
    
    def test_parse_no_profile(self):
        """Test that sim_id without profile returns 'unknown'."""
        result = parse_sim_id('pre-Loop_NoMitigations')
        self.assertEqual(result['patient_profile'], 'unknown')
        self.assertEqual(result['scenario_type'], 'pre-mitigation')


class TestParseMatchingKey(unittest.TestCase):
    """Tests for matching key parsing."""
    
    def test_parse_matching_key_full(self):
        """Test parsing a complete matching key."""
        key = 'TLR-899_1_125|pre-mitigation|adolescent|Simulation-Config.json'
        result = parse_matching_key(key)
        
        self.assertEqual(result['risk_name'], 'TLR-899_1_125')
        self.assertEqual(result['scenario_type'], 'pre-mitigation')
        self.assertEqual(result['patient_profile'], 'adolescent')
        self.assertEqual(result['scenario_name'], 'Simulation-Config.json')
    
    def test_parse_matching_key_malformed(self):
        """Test parsing a malformed key gracefully."""
        key = 'incomplete_key'
        result = parse_matching_key(key)
        
        self.assertEqual(result['risk_name'], 'incomplete_key')
        self.assertEqual(result['scenario_type'], 'unknown')
    
    def test_format_unmatched_scenarios(self):
        """Test formatting unmatched scenarios into readable list."""
        keys = [
            'TLR-899|pre-mitigation|adolescent|Config1.json',
            'TLR-1049|noLoop|sensitive|Config2.json'
        ]
        result = format_unmatched_scenarios(keys)
        
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['risk_name'], 'TLR-1049')  # Sorted alphabetically
        self.assertEqual(result[1]['risk_name'], 'TLR-899')
    
    def test_format_unmatched_scenarios_empty(self):
        """Test formatting empty list."""
        result = format_unmatched_scenarios([])
        self.assertEqual(result, [])


class TestMatchingKey(unittest.TestCase):
    """Tests for matching key generation."""
    
    def test_create_matching_key(self):
        """Test that matching keys are created correctly."""
        row = pd.Series({
            'sim_id': 'pre-Loop_NoMitigations_t1_adolescent',
            'risk_name': 'TLR-899_1_125',
            'scenario_name': 'Simulation-Configuration-TLR-899_1_125_adolescent_profile_v1.json'
        })
        key = create_matching_key(row)
        
        self.assertIn('TLR-899_1_125', key)
        self.assertIn('pre-mitigation', key)
        self.assertIn('adolescent', key)
    
    def test_matching_keys_consistent(self):
        """Test that equivalent scenarios produce the same key."""
        row1 = pd.Series({
            'sim_id': 'pre-Loop_NoMitigations_t1_sensitive',
            'risk_name': 'TLR-1049_18',
            'scenario_name': 'Simulation-Configuration-TLR-1049_18_Sensitive_profile_v1.json'
        })
        row2 = pd.Series({
            'sim_id': 'pre-Loop-NoMitigations_t1_sensitive',  # Different format
            'risk_name': 'TLR-1049_18',
            'scenario_name': 'Simulation-Configuration-TLR-1049_18_Sensitive_profile_v1.json'
        })
        
        key1 = create_matching_key(row1)
        key2 = create_matching_key(row2)
        
        self.assertEqual(key1, key2)


class TestPairwiseMatching(unittest.TestCase):
    """Tests for pairwise matching validation."""
    
    def setUp(self):
        """Set up test DataFrames."""
        self.df_v1 = pd.DataFrame({
            'matching_key': ['A', 'B', 'C'],
            'value': [1, 2, 3]
        })
        self.df_v2 = pd.DataFrame({
            'matching_key': ['A', 'B', 'C'],
            'value': [4, 5, 6]
        })
    
    def test_valid_matching(self):
        """Test validation when all pairs match."""
        result = validate_pairwise_matching(self.df_v1, self.df_v2)
        
        self.assertTrue(result['is_valid'])
        self.assertEqual(result['matched'], 3)
        self.assertEqual(len(result['only_in_v1']), 0)
        self.assertEqual(len(result['only_in_v2']), 0)
    
    def test_unmatched_in_v1(self):
        """Test validation when V1 has unmatched scenarios."""
        df_v1_extra = pd.DataFrame({
            'matching_key': ['A', 'B', 'C', 'D'],
            'value': [1, 2, 3, 4]
        })
        
        result = validate_pairwise_matching(df_v1_extra, self.df_v2)
        
        self.assertFalse(result['is_valid'])
        self.assertEqual(result['matched'], 3)
        self.assertIn('D', result['only_in_v1'])
    
    def test_unmatched_in_v2(self):
        """Test validation when V2 has unmatched scenarios."""
        df_v2_extra = pd.DataFrame({
            'matching_key': ['A', 'B', 'C', 'E'],
            'value': [4, 5, 6, 7]
        })
        
        result = validate_pairwise_matching(self.df_v1, df_v2_extra)
        
        self.assertFalse(result['is_valid'])
        self.assertIn('E', result['only_in_v2'])
    
    def test_unmatched_details_populated(self):
        """Test that unmatched details are populated with parsed information."""
        df_v1_with_key = pd.DataFrame({
            'matching_key': ['TLR-899|pre-mitigation|adolescent|Config.json', 'B'],
            'value': [1, 2]
        })
        df_v2_with_key = pd.DataFrame({
            'matching_key': ['B', 'TLR-1049|noLoop|sensitive|Other.json'],
            'value': [3, 4]
        })
        
        result = validate_pairwise_matching(df_v1_with_key, df_v2_with_key)
        
        # Check V1 unmatched details
        self.assertEqual(len(result['unmatched_v1_details']), 1)
        self.assertEqual(result['unmatched_v1_details'][0]['risk_name'], 'TLR-899')
        self.assertEqual(result['unmatched_v1_details'][0]['scenario_type'], 'pre-mitigation')
        
        # Check V2 unmatched details
        self.assertEqual(len(result['unmatched_v2_details']), 1)
        self.assertEqual(result['unmatched_v2_details'][0]['risk_name'], 'TLR-1049')
        self.assertEqual(result['unmatched_v2_details'][0]['patient_profile'], 'sensitive')


class TestStatisticalCalculations(unittest.TestCase):
    """Tests for statistical calculation functions."""
    
    def setUp(self):
        """Set up test DataFrame with known values."""
        np.random.seed(42)
        n = 100
        
        # Create data with known properties
        self.df = pd.DataFrame({
            'lbgi_v1': np.random.uniform(0, 5, n),
            'lbgi_v2': np.random.uniform(0, 5, n),
            'dka_index_v1': np.random.uniform(0, 3, n),
            'dka_index_v2': np.random.uniform(0, 3, n),
            'percent_cgm_gt_180_v1': np.random.uniform(0, 50, n),
            'percent_cgm_gt_180_v2': np.random.uniform(0, 50, n),
            'patient_profile': np.random.choice(PATIENT_PROFILES, n)
        })
        
        # Add difference columns
        self.df['lbgi_diff'] = self.df['lbgi_v2'] - self.df['lbgi_v1']
        self.df['dkai_diff'] = self.df['dka_index_v2'] - self.df['dka_index_v1']
        self.df['tar_diff'] = self.df['percent_cgm_gt_180_v2'] - self.df['percent_cgm_gt_180_v1']
    
    def test_calc_range_difference(self):
        """Test range calculation."""
        result = calc_range_difference(self.df, 'lbgi')
        
        self.assertIn('v1_min', result)
        self.assertIn('v1_max', result)
        self.assertIn('v1_range', result)
        self.assertIn('v2_range', result)
        self.assertIn('range_diff', result)
        
        self.assertAlmostEqual(result['v1_range'], result['v1_max'] - result['v1_min'])
    
    def test_calc_mean_difference(self):
        """Test mean calculation."""
        result = calc_mean_difference(self.df, 'lbgi')
        
        self.assertAlmostEqual(result['v1_mean'], self.df['lbgi_v1'].mean())
        self.assertAlmostEqual(result['v2_mean'], self.df['lbgi_v2'].mean())
        self.assertAlmostEqual(result['mean_diff'], result['v2_mean'] - result['v1_mean'])
    
    def test_calc_median_difference(self):
        """Test median calculation."""
        result = calc_median_difference(self.df, 'lbgi')
        
        self.assertAlmostEqual(result['v1_median'], self.df['lbgi_v1'].median())
        self.assertAlmostEqual(result['v2_median'], self.df['lbgi_v2'].median())
    
    def test_calc_std_difference(self):
        """Test standard deviation calculation."""
        result = calc_std_difference(self.df, 'lbgi')
        
        self.assertAlmostEqual(result['v1_std'], self.df['lbgi_v1'].std())
        self.assertAlmostEqual(result['v2_std'], self.df['lbgi_v2'].std())
    
    def test_calc_wilcoxon_signed_rank(self):
        """Test Wilcoxon signed-rank test returns expected keys."""
        result = calc_wilcoxon_signed_rank(self.df, 'lbgi')
        
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
        self.assertIn('n_total', result)
        self.assertIn('n_non_zero', result)
        self.assertIn('n_zero', result)
        self.assertIn('pct_changed', result)
        self.assertIn('rank_biserial_r', result)
        self.assertIn('r_interpretation', result)
        
        # p-value should be between 0 and 1
        if not np.isnan(result['p_value']):
            self.assertGreaterEqual(result['p_value'], 0)
            self.assertLessEqual(result['p_value'], 1)
        
        # n_total should equal n_non_zero + n_zero
        self.assertEqual(result['n_total'], result['n_non_zero'] + result['n_zero'])
        
        # rank_biserial_r should be between -1 and 1
        if not np.isnan(result['rank_biserial_r']):
            self.assertGreaterEqual(result['rank_biserial_r'], -1)
            self.assertLessEqual(result['rank_biserial_r'], 1)
        
        # r_interpretation should be valid
        self.assertIn(result['r_interpretation'], 
                      ['negligible', 'small', 'medium', 'large', 'N/A'])
    
    def test_calc_cohens_d_paired(self):
        """Test Cohen's d calculation including non-zero variant."""
        result = calc_cohens_d_paired(self.df, 'lbgi')
        
        # Check all expected keys
        self.assertIn('cohens_d', result)
        self.assertIn('d', result)  # Backward compatibility
        self.assertIn('interpretation', result)
        self.assertIn('d_nonzero', result)
        self.assertIn('interpretation_nonzero', result)
        self.assertIn('n_nonzero', result)
        
        # Interpretations should be valid
        self.assertIn(result['interpretation'], 
                      ['negligible', 'small', 'medium', 'large', 'N/A'])
        self.assertIn(result['interpretation_nonzero'], 
                      ['negligible', 'small', 'medium', 'large', 'N/A'])
        
        # d and cohens_d should be the same (backward compatibility)
        self.assertEqual(result['d'], result['cohens_d'])
    
    def test_calc_paired_ttest(self):
        """Test paired t-test."""
        result = calc_paired_ttest(self.df, 'lbgi')
        
        self.assertIn('statistic', result)
        self.assertIn('p_value', result)
        self.assertIn('significant', result)
    
    def test_cohens_d_interpretation(self):
        """Test Cohen's d effect size interpretation thresholds."""
        # Create data with known large effect
        # Need variance in differences, not just a constant shift
        df_large = self.df.copy()
        # Add a large shift plus small noise to ensure non-zero std
        np.random.seed(42)
        df_large['lbgi_v2'] = df_large['lbgi_v1'] + 10 + np.random.normal(0, 0.5, len(df_large))
        
        result = calc_cohens_d_paired(df_large, 'lbgi')
        self.assertEqual(result['interpretation'], 'large')


class TestSeverityAndHarmDetection(unittest.TestCase):
    """Tests for severity score and harm detection functions."""
    
    def test_calculate_hyperglycemia_score_zero(self):
        """Test hyperglycemia score when TAR is 0."""
        self.assertEqual(calculate_hyperglycemia_score(0.0), 0)
    
    def test_calculate_hyperglycemia_score_low(self):
        """Test hyperglycemia score when TAR < 12."""
        self.assertEqual(calculate_hyperglycemia_score(5.0), 1)
        self.assertEqual(calculate_hyperglycemia_score(11.9), 1)
    
    def test_calculate_hyperglycemia_score_high(self):
        """Test hyperglycemia score when TAR >= 12."""
        self.assertEqual(calculate_hyperglycemia_score(12.0), 2)
        self.assertEqual(calculate_hyperglycemia_score(25.0), 2)
    
    def test_determine_harm_hypoglycemia(self):
        """Test harm determination when hypoglycemia is dominant."""
        harm, severity = determine_harm_and_severity(3, 1, 1)
        self.assertEqual(harm, 'Hypoglycemia')
        self.assertEqual(severity, 3)
    
    def test_determine_harm_dka(self):
        """Test harm determination when DKA is dominant."""
        harm, severity = determine_harm_and_severity(1, 3, 1)
        self.assertEqual(harm, 'DKA')
        self.assertEqual(severity, 3)
    
    def test_determine_harm_hyperglycemia(self):
        """Test harm determination when hyperglycemia applies."""
        harm, severity = determine_harm_and_severity(1, 0, 2)
        self.assertEqual(harm, 'Hyperglycemia')
        self.assertEqual(severity, 2)
    
    def test_determine_harm_baseline(self):
        """Test harm determination when all scores are 0."""
        harm, severity = determine_harm_and_severity(0, 0, 0)
        self.assertEqual(harm, 'Baseline')
        self.assertEqual(severity, 0)
    
    def test_determine_harm_lbgi_priority_on_tie(self):
        """Test that LBGI takes priority when tied with DKA."""
        harm, severity = determine_harm_and_severity(3, 3, 1)
        self.assertEqual(harm, 'Hypoglycemia')
        self.assertEqual(severity, 3)
    
    def test_detect_severity_threshold_crossing_increase(self):
        """Test detection of severity increase crossing threshold."""
        result = detect_severity_threshold_crossing_aggregated(2, 4)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['direction'], 'increased')
        self.assertEqual(result['v1_severity'], 2)
        self.assertEqual(result['v2_severity'], 4)
    
    def test_detect_severity_threshold_crossing_decrease(self):
        """Test detection of severity decrease crossing threshold."""
        result = detect_severity_threshold_crossing_aggregated(4, 1)
        
        self.assertIsNotNone(result)
        self.assertEqual(result['direction'], 'decreased')
    
    def test_detect_severity_threshold_no_crossing(self):
        """Test that no crossing is detected when severity stays in same band."""
        result = detect_severity_threshold_crossing_aggregated(1, 2)
        self.assertIsNone(result)
        
        result = detect_severity_threshold_crossing_aggregated(3, 4)
        self.assertIsNone(result)
    
    def test_detect_harm_type_change_hyper_to_dka(self):
        """Test detection of harm type change from hyperglycemia to DKA."""
        result = detect_harm_type_change_aggregated('Hyperglycemia', 'DKA')
        
        self.assertIsNotNone(result)
        self.assertEqual(result['v1_harm'], 'Hyperglycemia')
        self.assertEqual(result['v2_harm'], 'DKA')
    
    def test_detect_harm_type_no_change(self):
        """Test that no change is detected when harm type stays the same."""
        result = detect_harm_type_change_aggregated('Hypoglycemia', 'Hypoglycemia')
        self.assertIsNone(result)
    
    def test_detect_harm_type_hypoglycemia_not_flagged(self):
        """Test that changes involving Hypoglycemia are not flagged."""
        # Only Hyperglycemia <-> DKA changes are flagged
        result = detect_harm_type_change_aggregated('Hypoglycemia', 'DKA')
        self.assertIsNone(result)


class TestQuestion3ChangeTypes(unittest.TestCase):
    """Tests for Question 3 change type indicators."""
    
    def setUp(self):
        """Set up test data with known severity patterns."""
        np.random.seed(42)
        
        # Create data with controlled severity outcomes
        self.df = pd.DataFrame({
            'matching_key': ['k1', 'k2', 'k3', 'k4'],
            'risk_name': ['TLR-100', 'TLR-100', 'TLR-200', 'TLR-200'],
            'scenario_type': ['pre-mitigation', 'post-mitigation', 'pre-mitigation', 'post-mitigation'],
            'patient_profile': ['median', 'median', 'median', 'median'],
            # V1: high severity (score 3)
            'lbgi_risk_score_v1': [3, 3, 2, 2],
            'dka_risk_score_v1': [1, 1, 1, 1],
            'percent_cgm_gt_180_v1': [10.0, 10.0, 10.0, 10.0],
            # V2: lower severity for TLR-100, same for TLR-200
            'lbgi_risk_score_v2': [1, 1, 2, 2],
            'dka_risk_score_v2': [1, 1, 1, 1],
            'percent_cgm_gt_180_v2': [10.0, 10.0, 10.0, 10.0],
        })
    
    def test_severity_decreased_indicator(self):
        """Test that severity_decreased appears when V2 < V1."""
        results = run_question3_analysis(self.df)
        
        # TLR-100 should show severity_decreased (3 -> 1, crosses threshold)
        tlr100 = results[results['risk_name'] == 'TLR-100']
        for _, row in tlr100.iterrows():
            self.assertLess(row['severity_change'], 0)
            # Should have severity_threshold_crossing since it crosses {1,2} <-> {3,4,5}
            self.assertIn('severity_threshold_crossing', row['change_types'])
    
    def test_no_change_indicator(self):
        """Test that 'None' appears when there's no severity change."""
        results = run_question3_analysis(self.df)
        
        # TLR-200 should show 'None' (2 -> 2, no change)
        tlr200 = results[results['risk_name'] == 'TLR-200']
        for _, row in tlr200.iterrows():
            self.assertEqual(row['severity_change'], 0)
            self.assertEqual(row['change_types'], 'None')
    
    def test_severity_decreased_without_threshold_crossing(self):
        """Test severity_decreased indicator when decrease stays within same band."""
        # Create data where severity decreases but stays in low band (2 -> 1)
        df = pd.DataFrame({
            'matching_key': ['k1'],
            'risk_name': ['TLR-300'],
            'scenario_type': ['pre-mitigation'],
            'patient_profile': ['median'],
            'lbgi_risk_score_v1': [2],
            'dka_risk_score_v1': [0],
            'percent_cgm_gt_180_v1': [5.0],
            'lbgi_risk_score_v2': [1],
            'dka_risk_score_v2': [0],
            'percent_cgm_gt_180_v2': [5.0],
        })
        
        results = run_question3_analysis(df)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]['severity_change'], -1)
        self.assertIn('severity_decreased', results.iloc[0]['change_types'])
        # Should NOT have threshold crossing since both are in low band
        self.assertNotIn('severity_threshold_crossing', results.iloc[0]['change_types'])
    
    def test_severity_increased_without_threshold_crossing(self):
        """Test severity_increased indicator when increase stays within same band."""
        # Create data where severity increases but stays in high band (3 -> 4)
        df = pd.DataFrame({
            'matching_key': ['k1'],
            'risk_name': ['TLR-400'],
            'scenario_type': ['pre-mitigation'],
            'patient_profile': ['median'],
            'lbgi_risk_score_v1': [3],
            'dka_risk_score_v1': [0],
            'percent_cgm_gt_180_v1': [5.0],
            'lbgi_risk_score_v2': [4],
            'dka_risk_score_v2': [0],
            'percent_cgm_gt_180_v2': [5.0],
        })
        
        results = run_question3_analysis(df)
        
        self.assertEqual(len(results), 1)
        self.assertEqual(results.iloc[0]['severity_change'], 1)
        self.assertIn('severity_increased', results.iloc[0]['change_types'])
        # Should NOT have threshold crossing since both are in high band
        self.assertNotIn('severity_threshold_crossing', results.iloc[0]['change_types'])


class TestAcceptanceCriteria(unittest.TestCase):
    """Tests for acceptance criteria evaluation."""
    
    def test_acceptance_all_pass(self):
        """Test acceptance when all criteria pass."""
        # Mock results with no significant increases
        q1_results = {
            'lbgi': {
                'metric_name': 'LBGI',
                'wilcoxon': {'significant': False, 'p_value': 0.5},
                'mean': {'mean_diff': -0.1, 'v1_mean': 1.0, 'v2_mean': 0.9}
            },
            'dka_index': {
                'metric_name': 'DKAI',
                'wilcoxon': {'significant': False, 'p_value': 0.6},
                'mean': {'mean_diff': 0.0, 'v1_mean': 0.5, 'v2_mean': 0.5}
            },
            'percent_cgm_gt_180': {
                'metric_name': 'TAR%',
                'wilcoxon': {'significant': False, 'p_value': 0.7},
                'mean': {'mean_diff': -1.0, 'v1_mean': 15.0, 'v2_mean': 14.0}
            }
        }
        
        q2_results = {}
        for profile in PATIENT_PROFILES:
            q2_results[profile] = {
                'n_pairs': 50,
                'lbgi': {
                    'wilcoxon': {'significant': False, 'p_value': 0.5},
                    'mean': {'mean_diff': -0.1, 'v1_mean': 1.0, 'v2_mean': 0.9}
                },
                'dka_index': {
                    'wilcoxon': {'significant': False, 'p_value': 0.6},
                    'mean': {'mean_diff': 0.0, 'v1_mean': 0.5, 'v2_mean': 0.5}
                },
                'percent_cgm_gt_180': {
                    'wilcoxon': {'significant': False, 'p_value': 0.7},
                    'mean': {'mean_diff': -1.0, 'v1_mean': 15.0, 'v2_mean': 14.0}
                }
            }
        
        result = evaluate_acceptance_criteria(q1_results, q2_results)
        
        self.assertTrue(result['criterion1']['passed'])
        self.assertTrue(result['criterion2']['passed'])
        self.assertTrue(result['overall']['passed'])
        self.assertEqual(result['overall']['summary'], 'ACCEPTABLE')
    
    def test_acceptance_fail_criterion1(self):
        """Test acceptance when criterion 1 fails (significant increase)."""
        q1_results = {
            'lbgi': {
                'metric_name': 'LBGI',
                'wilcoxon': {'significant': True, 'p_value': 0.01},  # Significant
                'mean': {'mean_diff': 0.5, 'v1_mean': 1.0, 'v2_mean': 1.5}  # Increased
            },
            'dka_index': {
                'metric_name': 'DKAI',
                'wilcoxon': {'significant': False, 'p_value': 0.6},
                'mean': {'mean_diff': 0.0, 'v1_mean': 0.5, 'v2_mean': 0.5}
            },
            'percent_cgm_gt_180': {
                'metric_name': 'TAR%',
                'wilcoxon': {'significant': False, 'p_value': 0.7},
                'mean': {'mean_diff': -1.0, 'v1_mean': 15.0, 'v2_mean': 14.0}
            }
        }
        
        q2_results = {profile: {'n_pairs': 50} for profile in PATIENT_PROFILES}
        
        result = evaluate_acceptance_criteria(q1_results, q2_results)
        
        self.assertFalse(result['criterion1']['passed'])
        self.assertFalse(result['overall']['passed'])
        self.assertEqual(result['overall']['summary'], 'REQUIRES REVIEW')


class TestIntegration(unittest.TestCase):
    """Integration tests for full analysis pipeline."""
    
    def setUp(self):
        """Set up test data."""
        np.random.seed(42)
        n = 200
        
        # Create realistic test data
        profiles = np.random.choice(PATIENT_PROFILES, n)
        scenario_types = np.random.choice(['pre-mitigation', 'noLoop', 'post-mitigation'], n)
        risk_names = [f'TLR-{np.random.randint(700, 1200)}' for _ in range(n)]
        
        self.df_merged = pd.DataFrame({
            'matching_key': [f'key_{i}' for i in range(n)],
            'sim_id_v1': [f'pre-Loop_NoMitigations_t1_{p}' for p in profiles],
            'sim_id_v2': [f'pre-Loop_NoMitigations_t1_{p}' for p in profiles],
            'risk_name': risk_names,
            'risk_name_v1': risk_names,
            'risk_name_v2': risk_names,
            'scenario_name': [f'Config_{i}' for i in range(n)],
            'scenario_name_v1': [f'Config_{i}' for i in range(n)],
            'scenario_name_v2': [f'Config_{i}' for i in range(n)],
            'scenario_type': scenario_types,
            'scenario_type_v1': scenario_types,
            'scenario_type_v2': scenario_types,
            'patient_profile': profiles,
            'patient_profile_v1': profiles,
            'patient_profile_v2': profiles,
            'lbgi_v1': np.random.uniform(0, 5, n),
            'lbgi_v2': np.random.uniform(0, 5, n),
            'dka_index_v1': np.random.uniform(0, 3, n),
            'dka_index_v2': np.random.uniform(0, 3, n),
            'percent_cgm_gt_180_v1': np.random.uniform(0, 50, n),
            'percent_cgm_gt_180_v2': np.random.uniform(0, 50, n),
            'lbgi_risk_score_v1': np.random.randint(0, 5, n),
            'lbgi_risk_score_v2': np.random.randint(0, 5, n),
            'dka_risk_score_v1': np.random.randint(0, 4, n),
            'dka_risk_score_v2': np.random.randint(0, 4, n)
        })
        
        # Add difference columns
        self.df_merged['lbgi_diff'] = self.df_merged['lbgi_v2'] - self.df_merged['lbgi_v1']
        self.df_merged['dkai_diff'] = self.df_merged['dka_index_v2'] - self.df_merged['dka_index_v1']
        self.df_merged['tar_diff'] = self.df_merged['percent_cgm_gt_180_v2'] - self.df_merged['percent_cgm_gt_180_v1']
    
    def test_run_question1_analysis(self):
        """Test that Question 1 analysis completes and returns expected structure."""
        results = run_question1_analysis(self.df_merged)
        
        # Check all metrics are present
        for metric in METRICS.keys():
            self.assertIn(metric, results)
            self.assertIn('range', results[metric])
            self.assertIn('mean', results[metric])
            self.assertIn('median', results[metric])
            self.assertIn('std', results[metric])
            self.assertIn('wilcoxon', results[metric])
            self.assertIn('ttest', results[metric])
            self.assertIn('cohens_d', results[metric])
    
    def test_run_question2_analysis(self):
        """Test that Question 2 analysis completes and returns expected structure."""
        results = run_question2_analysis(self.df_merged)
        
        # Check all profiles are present
        for profile in PATIENT_PROFILES:
            self.assertIn(profile, results)
            if 'n_pairs' in results[profile]:
                self.assertGreater(results[profile]['n_pairs'], 0)
        
        # Check between-profile tests
        self.assertIn('between_profile_tests', results)
    
    def test_run_question3_analysis(self):
        """Test that Question 3 analysis completes and returns expected structure."""
        results = run_question3_analysis(self.df_merged)
        
        self.assertIsInstance(results, pd.DataFrame)
        
        # Check expected columns for aggregated output
        expected_cols = ['risk_name', 'scenario_type', 'n_profiles',
                         'harm_v1', 'severity_v1', 'lbgi_avg_v1', 'dka_avg_v1', 'tar_avg_v1',
                         'harm_v2', 'severity_v2', 'lbgi_avg_v2', 'dka_avg_v2', 'tar_avg_v2',
                         'severity_change', 'has_significant_change', 'change_types']
        for col in expected_cols:
            self.assertIn(col, results.columns)
        
        # Verify aggregation occurred (should have fewer rows than input)
        n_unique_combos = self.df_merged.groupby(['risk_name', 'scenario_type']).ngroups
        self.assertEqual(len(results), n_unique_combos)
        
        # Verify severity values are integers
        self.assertTrue(results['severity_v1'].dtype in [np.int64, np.int32, int])
        self.assertTrue(results['severity_v2'].dtype in [np.int64, np.int32, int])


class TestTable3Generation(unittest.TestCase):
    """Tests for Table 3 generation with mean, median, and Wilcoxon rows."""
    
    def setUp(self):
        """Set up test DataFrame."""
        np.random.seed(42)
        n = 100
        profiles = np.random.choice(PATIENT_PROFILES, n)
        
        self.df_merged = pd.DataFrame({
            'matching_key': [f'key_{i}' for i in range(n)],
            'risk_name': [f'TLR-{i % 10}' for i in range(n)],
            'scenario_name': [f'Config_{i}.json' for i in range(n)],
            'scenario_type': np.random.choice(['pre-mitigation', 'noLoop', 'post-mitigation'], n),
            'patient_profile': profiles,
            'lbgi_v1': np.random.uniform(0, 5, n),
            'lbgi_v2': np.random.uniform(0, 5, n),
            'dka_index_v1': np.random.uniform(0, 3, n),
            'dka_index_v2': np.random.uniform(0, 3, n),
            'percent_cgm_gt_180_v1': np.random.uniform(0, 50, n),
            'percent_cgm_gt_180_v2': np.random.uniform(0, 50, n),
        })
    
    def test_table3_includes_mean_median_wilcoxon(self):
        """Test that Table 3 includes Mean, Median, and Wilcoxon rows for each profile/metric."""
        from post_processing.algorithm_risk_comparison import generate_table3
        
        q2_results = run_question2_analysis(self.df_merged)
        table3 = generate_table3(q2_results)
        
        # Check that table has rows
        self.assertGreater(len(table3), 0)
        
        # For each profile/metric combination, should have Mean, Median, Wilcoxon rows
        test_types = set(table3['Test'].unique())
        self.assertIn('Mean', test_types)
        self.assertIn('Median', test_types)
        self.assertIn('Wilcoxon', test_types)
        
        # Count rows per profile/metric - should be 3 (Mean, Median, Wilcoxon)
        for profile in PATIENT_PROFILES:
            for metric_name in METRICS.values():
                profile_metric_rows = table3[
                    (table3['Patient Profile'] == profile.capitalize()) &
                    (table3['Metric'] == metric_name)
                ]
                # Should have exactly 3 rows (Mean, Median, Wilcoxon)
                if len(profile_metric_rows) > 0:  # Only if data exists for this combo
                    self.assertEqual(len(profile_metric_rows), 3, 
                                     f"Expected 3 rows for {profile}/{metric_name}")
    
    def test_table3_median_values_correct(self):
        """Test that median values in Table 3 are correct."""
        from post_processing.algorithm_risk_comparison import generate_table3
        
        q2_results = run_question2_analysis(self.df_merged)
        table3 = generate_table3(q2_results)
        
        # Get a median row and verify values
        median_rows = table3[table3['Test'] == 'Median']
        self.assertGreater(len(median_rows), 0)
        
        # Check that V1 Result and V2 Result are numeric strings
        for _, row in median_rows.iterrows():
            try:
                float(row['V1 Result'])
                float(row['V2 Result'])
                float(row['Difference'])
            except ValueError:
                self.fail(f"Median row has non-numeric values: {row}")


class TestExportUnmatchedScenarios(unittest.TestCase):
    """Tests for exporting unmatched scenarios to CSV."""
    
    def test_export_unmatched_scenarios_creates_file(self):
        """Test that CSV file is created with unmatched scenarios."""
        validation_report = {
            'unmatched_v1_details': [
                {'risk_name': 'TLR-899', 'scenario_type': 'pre-mitigation',
                 'patient_profile': 'adolescent', 'scenario_name': 'Config1.json'}
            ],
            'unmatched_v2_details': [
                {'risk_name': 'TLR-1049', 'scenario_type': 'noLoop',
                 'patient_profile': 'sensitive', 'scenario_name': 'Config2.json'}
            ]
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = export_unmatched_scenarios_to_csv(validation_report, tmpdir)
            
            self.assertIsNotNone(result_path)
            self.assertTrue(os.path.exists(result_path))
            
            # Verify CSV content
            df = pd.read_csv(result_path)
            self.assertEqual(len(df), 2)
            self.assertIn('Risk Name', df.columns)
            self.assertIn('Source', df.columns)
    
    def test_export_unmatched_scenarios_no_unmatched(self):
        """Test that no file is created when there are no unmatched scenarios."""
        validation_report = {
            'unmatched_v1_details': [],
            'unmatched_v2_details': []
        }
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result_path = export_unmatched_scenarios_to_csv(validation_report, tmpdir)
            
            self.assertIsNone(result_path)


class TestEdgeCases(unittest.TestCase):
    """Tests for edge cases and error handling."""
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        df_empty = pd.DataFrame(columns=['lbgi_v1', 'lbgi_v2', 'patient_profile'])
        
        # These should not raise exceptions
        result = calc_mean_difference(df_empty, 'lbgi')
        self.assertTrue(np.isnan(result['v1_mean']))
    
    def test_identical_values(self):
        """Test handling when V1 and V2 values are identical."""
        df_identical = pd.DataFrame({
            'lbgi_v1': [1.0, 2.0, 3.0],
            'lbgi_v2': [1.0, 2.0, 3.0],
            'patient_profile': ['median'] * 3
        })
        
        result = calc_wilcoxon_signed_rank(df_identical, 'lbgi')
        # Should handle gracefully (note in result or NaN p-value)
        self.assertIn('note', result) or self.assertTrue(np.isnan(result['p_value']))
    
    def test_single_observation(self):
        """Test handling of single observation."""
        df_single = pd.DataFrame({
            'lbgi_v1': [1.0],
            'lbgi_v2': [2.0],
            'patient_profile': ['median']
        })
        
        result = calc_mean_difference(df_single, 'lbgi')
        self.assertEqual(result['mean_diff'], 1.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)


# =============================================================================
# Module 9 Tests: Hazardous Situation Comparison Report
# =============================================================================

class TestParseProb(unittest.TestCase):
    """Tests for parse_probability()."""

    def test_integer_string(self):
        self.assertEqual(parse_probability('3'), 3.0)

    def test_plain_integer(self):
        self.assertEqual(parse_probability(3), 3.0)

    def test_float_string(self):
        self.assertAlmostEqual(parse_probability('2.5'), 2.5)

    def test_none_returns_none(self):
        self.assertIsNone(parse_probability(None))

    def test_non_numeric_string_returns_none(self):
        self.assertIsNone(parse_probability('1B'))

    def test_empty_string_returns_none(self):
        self.assertIsNone(parse_probability(''))


class TestComputeRiskScore(unittest.TestCase):
    """Tests for compute_risk_score()."""

    def test_pre_mitigation_uses_initial_prob(self):
        self.assertEqual(compute_risk_score(3, 2.0, 1.0, 'pre-mitigation'), 6.0)

    def test_noloop_uses_initial_prob(self):
        self.assertEqual(compute_risk_score(3, 2.0, 1.0, 'noLoop'), 6.0)

    def test_post_mitigation_uses_residual_prob(self):
        self.assertEqual(compute_risk_score(3, 2.0, 1.0, 'post-mitigation'), 3.0)

    def test_none_initial_pre_mitigation_returns_none(self):
        self.assertIsNone(compute_risk_score(3, None, 1.0, 'pre-mitigation'))

    def test_none_initial_noloop_returns_none(self):
        self.assertIsNone(compute_risk_score(3, None, 1.0, 'noLoop'))

    def test_none_residual_post_mitigation_returns_none(self):
        self.assertIsNone(compute_risk_score(3, 2.0, None, 'post-mitigation'))

    def test_zero_severity_returns_zero(self):
        self.assertEqual(compute_risk_score(0, 3.0, 3.0, 'pre-mitigation'), 0.0)


class TestGetAcceptability(unittest.TestCase):
    """Tests for get_acceptability() -- all 5 acceptability buckets."""

    def test_score_1_acceptable(self):
        label, color = get_acceptability(1, 1)
        self.assertEqual(label, 'Acceptable')
        self.assertEqual(color, 'C6EFCE')

    def test_score_3_acceptable(self):
        label, _ = get_acceptability(3, 5)
        self.assertEqual(label, 'Acceptable')

    def test_score_4_severity_1_acceptable(self):
        label, _ = get_acceptability(4, 1)
        self.assertEqual(label, 'Acceptable')

    def test_score_4_severity_3_acceptable(self):
        label, _ = get_acceptability(4, 3)
        self.assertEqual(label, 'Acceptable')

    def test_score_4_severity_4_conditionally_acceptable(self):
        label, color = get_acceptability(4, 4)
        self.assertEqual(label, 'Conditionally acceptable')
        self.assertEqual(color, 'FFEB9C')

    def test_score_4_severity_5_conditionally_acceptable(self):
        label, _ = get_acceptability(4, 5)
        self.assertEqual(label, 'Conditionally acceptable')

    def test_score_5_conditionally_acceptable(self):
        label, _ = get_acceptability(5, 1)
        self.assertEqual(label, 'Conditionally acceptable')

    def test_score_9_conditionally_acceptable(self):
        label, _ = get_acceptability(9, 3)
        self.assertEqual(label, 'Conditionally acceptable')

    def test_score_10_unacceptable(self):
        label, color = get_acceptability(10, 5)
        self.assertEqual(label, 'Unacceptable')
        self.assertEqual(color, 'FFC7CE')

    def test_score_25_unacceptable(self):
        label, _ = get_acceptability(25, 5)
        self.assertEqual(label, 'Unacceptable')

    def test_none_score_unknown(self):
        label, color = get_acceptability(None, 2)
        self.assertEqual(label, 'Unknown')
        self.assertEqual(color, 'FFFFFF')


class TestAcceptabilityRank(unittest.TestCase):
    """Tests for acceptability_rank()."""

    def test_ordering_acceptable_lt_conditionally(self):
        self.assertLess(
            acceptability_rank('Acceptable'),
            acceptability_rank('Conditionally acceptable'),
        )

    def test_ordering_conditionally_lt_unacceptable(self):
        self.assertLess(
            acceptability_rank('Conditionally acceptable'),
            acceptability_rank('Unacceptable'),
        )

    def test_unknown_is_negative(self):
        self.assertLess(acceptability_rank('Unknown'), 0)

    def test_missing_label_is_negative(self):
        self.assertLess(acceptability_rank('NotARealLabel'), 0)


class TestClassifyHazardRows(unittest.TestCase):
    """Tests for classify_hazard_rows() -- all five flags and in_scope."""

    def _make_enriched(self, overrides: dict) -> pd.DataFrame:
        """Return a one-row enriched DataFrame with sensible defaults."""
        defaults = {
            'risk_name': 'TLR-001',
            'scenario_type': 'pre-mitigation',
            'harm_v1': 'Hypoglycemia',
            'harm_v2': 'Hypoglycemia',
            'severity_v1': 2,
            'severity_v2': 3,
            'severity_change': 1,
            'change_types': 'None',
            'No Automation': False,
            'risk_score_v1': 4.0,
            'risk_score_v2': 6.0,
            'accept_label_v1': 'Acceptable',
            'accept_color_v1': 'C6EFCE',
            'accept_label_v2': 'Conditionally acceptable',
            'accept_color_v2': 'FFEB9C',
        }
        defaults.update(overrides)
        return pd.DataFrame([defaults])

    def test_flag_harm_change_set(self):
        df = self._make_enriched({'change_types': 'harm_type_change'})
        result = classify_hazard_rows(df)
        self.assertTrue(result.iloc[0]['flag_harm_change'])

    def test_flag_harm_change_not_set_when_absent(self):
        df = self._make_enriched({'change_types': 'severity_increased'})
        result = classify_hazard_rows(df)
        self.assertFalse(result.iloc[0]['flag_harm_change'])

    def test_flag_sev_increase_set(self):
        df = self._make_enriched({
            'change_types': 'severity_threshold_crossing',
            'severity_change': 2,
        })
        result = classify_hazard_rows(df)
        self.assertTrue(result.iloc[0]['flag_sev_increase'])
        self.assertFalse(result.iloc[0]['flag_sev_decrease'])

    def test_flag_sev_decrease_set(self):
        df = self._make_enriched({
            'change_types': 'severity_threshold_crossing',
            'severity_change': -2,
        })
        result = classify_hazard_rows(df)
        self.assertFalse(result.iloc[0]['flag_sev_increase'])
        self.assertTrue(result.iloc[0]['flag_sev_decrease'])

    def test_sev_flag_requires_threshold_crossing(self):
        """severity_increased without severity_threshold_crossing must NOT set sev flags."""
        df = self._make_enriched({
            'change_types': 'severity_increased',
            'severity_change': 1,
        })
        result = classify_hazard_rows(df)
        self.assertFalse(result.iloc[0]['flag_sev_increase'])

    def test_flag_risk_increase_set(self):
        df = self._make_enriched({
            'accept_label_v1': 'Acceptable',
            'accept_label_v2': 'Conditionally acceptable',
        })
        result = classify_hazard_rows(df)
        self.assertTrue(result.iloc[0]['flag_risk_increase'])
        self.assertFalse(result.iloc[0]['flag_risk_decrease'])

    def test_flag_risk_decrease_set(self):
        df = self._make_enriched({
            'accept_label_v1': 'Conditionally acceptable',
            'accept_label_v2': 'Acceptable',
        })
        result = classify_hazard_rows(df)
        self.assertFalse(result.iloc[0]['flag_risk_increase'])
        self.assertTrue(result.iloc[0]['flag_risk_decrease'])

    def test_noloop_risk_flag_blocked_when_no_automation_false(self):
        """noLoop row must NOT trigger risk flags when no_automation=False."""
        df = self._make_enriched({
            'scenario_type': 'noLoop',
            'No Automation': False,
            'accept_label_v1': 'Acceptable',
            'accept_label_v2': 'Unacceptable',
        })
        result = classify_hazard_rows(df)
        self.assertFalse(result.iloc[0]['flag_risk_increase'])

    def test_noloop_risk_flag_set_when_no_automation_true(self):
        """noLoop row SHOULD trigger risk flags when no_automation=True."""
        df = self._make_enriched({
            'scenario_type': 'noLoop',
            'No Automation': True,
            'accept_label_v1': 'Acceptable',
            'accept_label_v2': 'Unacceptable',
        })
        result = classify_hazard_rows(df)
        self.assertTrue(result.iloc[0]['flag_risk_increase'])

    def test_harm_change_not_filtered_by_noloop_no_automation(self):
        """flag_harm_change is criterion 5 -- no noLoop/no_automation filter."""
        df = self._make_enriched({
            'scenario_type': 'noLoop',
            'No Automation': False,
            'change_types': 'harm_type_change',
        })
        result = classify_hazard_rows(df)
        self.assertTrue(result.iloc[0]['flag_harm_change'])

    def test_in_scope_true_when_any_flag(self):
        df = self._make_enriched({'change_types': 'harm_type_change'})
        result = classify_hazard_rows(df)
        self.assertTrue(result.iloc[0]['in_scope'])

    def test_in_scope_false_when_no_flags(self):
        df = self._make_enriched({
            'change_types': 'None',
            'accept_label_v1': 'Acceptable',
            'accept_label_v2': 'Acceptable',
        })
        result = classify_hazard_rows(df)
        self.assertFalse(result.iloc[0]['in_scope'])


class TestAssignHazardRowsToSheets(unittest.TestCase):
    """Tests for assign_hazard_rows_to_sheets() deduplication priority."""

    def _make_classified(self, flags: dict) -> pd.DataFrame:
        """Return a one-row classified DataFrame with specified boolean flags."""
        base = {
            'risk_name': 'TLR-001',
            'scenario_type': 'pre-mitigation',
            'harm_v1': 'Hypoglycemia',
            'harm_v2': 'Hypoglycemia',
            'severity_v1': 2, 'severity_v2': 4,
            'severity_change': 2,
            'change_types': 'severity_threshold_crossing',
            'risk_score_v1': 4.0, 'risk_score_v2': 8.0,
            'accept_label_v1': 'Acceptable',
            'accept_color_v1': 'C6EFCE',
            'accept_label_v2': 'Conditionally acceptable',
            'accept_color_v2': 'FFEB9C',
            'Summary': 'Test summary',
            'Hazard Category': 'Cat A',
            'No Automation': False,
            'flag_harm_change': False,
            'flag_sev_increase': False,
            'flag_sev_decrease': False,
            'flag_risk_increase': False,
            'flag_risk_decrease': False,
            'in_scope': False,
        }
        base.update(flags)
        base['in_scope'] = any([
            base['flag_harm_change'], base['flag_sev_increase'],
            base['flag_sev_decrease'], base['flag_risk_increase'],
            base['flag_risk_decrease'],
        ])
        return pd.DataFrame([base])

    def test_harm_change_takes_priority_over_all_others(self):
        df = self._make_classified({
            'flag_harm_change': True,
            'flag_risk_increase': True,
            'flag_sev_increase': True,
        })
        result = assign_hazard_rows_to_sheets(df)
        self.assertEqual(len(result['Changes to Harm']), 1)
        self.assertEqual(len(result['Risk Score Increased']), 0)
        self.assertEqual(len(result['Severity Increased']), 0)

    def test_risk_increase_overrides_sev_increase(self):
        df = self._make_classified({
            'flag_risk_increase': True,
            'flag_sev_increase': True,
        })
        result = assign_hazard_rows_to_sheets(df)
        self.assertEqual(len(result['Risk Score Increased']), 1)
        self.assertEqual(len(result['Severity Increased']), 0)

    def test_risk_decrease_overrides_sev_decrease(self):
        df = self._make_classified({
            'flag_risk_decrease': True,
            'flag_sev_decrease': True,
        })
        result = assign_hazard_rows_to_sheets(df)
        self.assertEqual(len(result['Risk Score Decreased']), 1)
        self.assertEqual(len(result['Severity Decreased']), 0)

    def test_sev_increase_assigned_when_no_risk_change(self):
        df = self._make_classified({'flag_sev_increase': True})
        result = assign_hazard_rows_to_sheets(df)
        self.assertEqual(len(result['Severity Increased']), 1)
        self.assertEqual(len(result['Risk Score Increased']), 0)

    def test_sev_decrease_assigned_when_no_risk_change(self):
        df = self._make_classified({'flag_sev_decrease': True})
        result = assign_hazard_rows_to_sheets(df)
        self.assertEqual(len(result['Severity Decreased']), 1)

    def test_all_five_sheet_keys_always_present(self):
        df = self._make_classified({'flag_harm_change': True})
        result = assign_hazard_rows_to_sheets(df)
        for name in [
            'Changes to Harm', 'Severity Increased', 'Severity Decreased',
            'Risk Score Increased', 'Risk Score Decreased',
        ]:
            self.assertIn(name, result)

    def test_empty_sheets_have_zero_rows(self):
        df = self._make_classified({'flag_harm_change': True})
        result = assign_hazard_rows_to_sheets(df)
        for name in [
            'Severity Increased', 'Severity Decreased',
            'Risk Score Increased', 'Risk Score Decreased',
        ]:
            self.assertEqual(len(result[name]), 0)


class TestComputeHazardSummaryStats(unittest.TestCase):
    """Tests for compute_hazard_summary_stats()."""

    def setUp(self):
        self.q3_df = pd.DataFrame({
            'risk_name': ['TLR-001', 'TLR-002', 'TLR-003', 'TLR-004'],
            'scenario_type': ['pre-mitigation'] * 4,
            'severity_v1': [2, 3, 4, 2],
            'severity_v2': [3, 2, 4, 3],
            'severity_change': [1, -1, 0, 1],  # sum=1, decreased=1, increased=2
        })
        self.sheet_data = {
            'Changes to Harm': pd.DataFrame(),
            'Severity Increased': pd.DataFrame({'risk_name': ['TLR-001']}),
            'Severity Decreased': pd.DataFrame({'risk_name': ['TLR-002']}),
            'Risk Score Increased': pd.DataFrame(),
            'Risk Score Decreased': pd.DataFrame(),
        }

    def test_severity_change_sum(self):
        stats = compute_hazard_summary_stats(self.q3_df, self.sheet_data)
        self.assertEqual(stats['severity_change_sum'], 1)  # 1-1+0+1

    def test_n_decreased(self):
        stats = compute_hazard_summary_stats(self.q3_df, self.sheet_data)
        self.assertEqual(stats['n_sev_decreased'], 1)

    def test_n_increased(self):
        stats = compute_hazard_summary_stats(self.q3_df, self.sheet_data)
        self.assertEqual(stats['n_sev_increased'], 2)

    def test_practical_counts_all_present(self):
        stats = compute_hazard_summary_stats(self.q3_df, self.sheet_data)
        for name in [
            'Changes to Harm', 'Severity Increased', 'Severity Decreased',
            'Risk Score Increased', 'Risk Score Decreased',
        ]:
            self.assertIn(name, stats['practical_counts'])

    def test_practical_counts_values(self):
        stats = compute_hazard_summary_stats(self.q3_df, self.sheet_data)
        self.assertEqual(stats['practical_counts']['Changes to Harm'], 0)
        self.assertEqual(stats['practical_counts']['Severity Increased'], 1)
        self.assertEqual(stats['practical_counts']['Severity Decreased'], 1)


class TestWriteHazardousComparisonExcel(unittest.TestCase):
    """Integration tests for write_hazardous_situation_comparison_excel()."""

    def _make_sheet_data(self):
        return {
            'Changes to Harm': pd.DataFrame([{
                'risk_name': 'TLR-001',
                'Summary': 'Test hazard',
                'Hazard Category': 'Category A',
                'scenario_type': 'noLoop',
                'harm_v1': 'Hypoglycemia',
                'harm_v2': 'DKA',
                'risk_score_v1': 3.0,
                'risk_score_v2': 6.0,
                'accept_label_v1': 'Acceptable',
                'accept_color_v1': 'C6EFCE',
                'accept_label_v2': 'Conditionally acceptable',
                'accept_color_v2': 'FFEB9C',
            }]),
            'Severity Increased': pd.DataFrame(),
            'Severity Decreased': pd.DataFrame(),
            'Risk Score Increased': pd.DataFrame(),
            'Risk Score Decreased': pd.DataFrame(),
        }

    def _make_summary_stats(self, sheet_data):
        return {
            'severity_change_sum': 5,
            'n_sev_decreased': 2,
            'n_sev_increased': 3,
            'practical_counts': {k: len(v) for k, v in sheet_data.items()},
        }

    def test_file_created(self):
        sheet_data = self._make_sheet_data()
        stats = self._make_summary_stats(sheet_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.xlsx')
            result = write_hazardous_situation_comparison_excel(sheet_data, stats, path)
            self.assertEqual(result, path)
            self.assertTrue(os.path.exists(path))

    def test_all_six_sheets_present(self):
        import openpyxl
        sheet_data = self._make_sheet_data()
        stats = self._make_summary_stats(sheet_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.xlsx')
            write_hazardous_situation_comparison_excel(sheet_data, stats, path)
            wb = openpyxl.load_workbook(path)
            expected = [
                'Summary', 'Changes to Harm', 'Severity Increased',
                'Severity Decreased', 'Risk Score Increased', 'Risk Score Decreased',
            ]
            for name in expected:
                self.assertIn(name, wb.sheetnames)

    def test_empty_sheet_gets_no_criteria_message(self):
        import openpyxl
        sheet_data = self._make_sheet_data()
        stats = self._make_summary_stats(sheet_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.xlsx')
            write_hazardous_situation_comparison_excel(sheet_data, stats, path)
            wb = openpyxl.load_workbook(path)
            ws = wb['Severity Increased']
            self.assertEqual(ws['A2'].value, 'No hazardous situations meeting criteria')

    def test_data_row_written_to_changes_sheet(self):
        import openpyxl
        sheet_data = self._make_sheet_data()
        stats = self._make_summary_stats(sheet_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.xlsx')
            write_hazardous_situation_comparison_excel(sheet_data, stats, path)
            wb = openpyxl.load_workbook(path)
            ws = wb['Changes to Harm']
            self.assertEqual(ws['A2'].value, 'TLR-001')

    def test_risk_score_cell_contains_score_and_label(self):
        """V1 Risk Score cell (col 7 on Changes to Harm) should be '3, Acceptable'."""
        import openpyxl
        sheet_data = self._make_sheet_data()
        stats = self._make_summary_stats(sheet_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.xlsx')
            write_hazardous_situation_comparison_excel(sheet_data, stats, path)
            wb = openpyxl.load_workbook(path)
            ws = wb['Changes to Harm']
            # 'Changes to Harm' col order: TLR Key, Summary, Hazard Cat, Eval Stage,
            # V1 Harm, V2 Harm, V1 Risk Score (col 7), V2 Risk Score (col 8)
            v1_cell = ws.cell(row=2, column=7)
            self.assertIn('3', v1_cell.value)
            self.assertIn('Acceptable', v1_cell.value)

    def test_summary_sheet_contains_severity_change_sum(self):
        import openpyxl
        sheet_data = self._make_sheet_data()
        stats = self._make_summary_stats(sheet_data)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test.xlsx')
            write_hazardous_situation_comparison_excel(sheet_data, stats, path)
            wb = openpyxl.load_workbook(path)
            ws = wb['Summary']
            # Row 2 is first data row (row 1 = header)
            metric_col_values = [ws.cell(row=r, column=2).value for r in range(2, 10)]
            self.assertIn(5, metric_col_values)  # severity_change_sum = 5


if __name__ == '__main__':
    unittest.main(verbosity=2)
