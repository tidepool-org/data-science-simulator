#!/usr/bin/env python3
"""
Algorithm Risk Profile Comparison Analysis

This script implements the statistical analysis plan (TMP-0012) for comparing
risk profiles between two versions of the Tidepool Loop algorithm (previously
cleared version vs. version 2.0).

The analysis answers three key questions:
1. Does the overall severity of harm change when measured across all risks?
2. Does the severity of harm vary significantly across virtual patient profiles?
3. Are there individual hazardous situations with practically significant changes?

Metrics analyzed: LBGI (hypoglycemia), DKAI (DKA), TAR% (time above 180 mg/dL)
"""

__author__ = "Shawn Foster"

import argparse
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

from jira_risk_probabilities import DEFAULT_API_VERSION, fetch_probabilities, normalize_tlr_key


# =============================================================================
# Constants
# =============================================================================

METRICS = {
    'lbgi': 'LBGI (Hypoglycemia)',
    'dka_index': 'DKAI (DKA)',
    'percent_cgm_gt_180': 'TAR% (>180 mg/dL)'
}

SEVERITY_SCORE_COLUMNS = {
    'lbgi': 'lbgi_risk_score',
    'dka_index': 'dka_risk_score'
}

PATIENT_PROFILES = ['adolescent', 'sensitive', 'resistant', 'median']

SCENARIO_TYPES = ['pre-mitigation', 'noLoop', 'post-mitigation']

# Significance level
ALPHA = 0.05


# =============================================================================
# Rounding
# =============================================================================

def round_half_up(value: float) -> int:
    """
    Round a value to the nearest integer using round-half-up logic.

    Mirrors the implementation in create_severity_summary.py. Unlike Python's
    built-in round(), which uses banker's rounding (round half to even), this
    function always rounds 0.5 up to the next integer, ensuring conservative
    (higher) risk estimates when averaging severity scores.

    Args:
        value: Numeric value to round

    Returns:
        Integer result of rounding
    """
    import math
    return math.floor(value + 0.5)


# =============================================================================
# Module 1: Data Input and Preprocessing
# =============================================================================

def load_simulation_data(filepath: str) -> pd.DataFrame:
    """
    Load simulation results from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        DataFrame containing simulation results
    """
    df = pd.read_csv(filepath)
    required_columns = ['sim_id', 'lbgi', 'dka_index', 'percent_cgm_gt_180', 
                        'lbgi_risk_score', 'dka_risk_score', 'scenario_name', 'risk_name']
    
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    return df


def parse_sim_id(sim_id: str) -> Dict[str, str]:
    """
    Extract scenario type and patient profile from sim_id.
    
    Handles variations in naming conventions:
    - pre-Loop_NoMitigations_t1_*, pre-Loop-NoMitigations_t1_*, pre-Loop-noMitigations_t1_*
    - pre-noLoop_t1_*, pre-NoLoop_t1_*
    - post-Loop-WithMitigations_t1_*, post-LoopWithMitigations_t1_*
    
    Args:
        sim_id: Simulation ID string
        
    Returns:
        Dictionary with 'scenario_type' and 'patient_profile' keys
    """
    sim_id_lower = sim_id.lower()
    
    # Determine scenario type
    if 'post-' in sim_id_lower or sim_id_lower.startswith('post'):
        scenario_type = 'post-mitigation'
    elif 'noloop' in sim_id_lower and 'pre-' in sim_id_lower:
        # Check if it's a noLoop scenario (not pre-mitigation with Loop)
        if re.search(r'pre-noloop|pre-no-loop', sim_id_lower):
            scenario_type = 'noLoop'
        else:
            scenario_type = 'pre-mitigation'
    elif 'pre-' in sim_id_lower:
        if 'nomitigation' in sim_id_lower or 'nomitigations' in sim_id_lower:
            scenario_type = 'pre-mitigation'
        elif 'noloop' in sim_id_lower:
            scenario_type = 'noLoop'
        else:
            scenario_type = 'pre-mitigation'
    else:
        scenario_type = 'unknown'
    
    # Determine patient profile
    # Include common typos/variants
    profile_variants = {
        'adolescent': ['adolescent'],
        'sensitive': ['sensitive'],
        'resistant': ['resistant', 'resistnat'],  # Handle typo
        'median': ['median']
    }
    
    patient_profile = 'unknown'
    for profile, variants in profile_variants.items():
        for variant in variants:
            if variant in sim_id_lower:
                patient_profile = profile
                break
        if patient_profile != 'unknown':
            break
    
    return {
        'scenario_type': scenario_type,
        'patient_profile': patient_profile
    }


def create_matching_key(row: pd.Series) -> str:
    """
    Generate unique key for pairwise matching between versions.
    
    The key combines risk_name, scenario_name, and normalized sim_id components
    to ensure proper pairing across algorithm versions.
    
    Args:
        row: DataFrame row
        
    Returns:
        Matching key string
    """
    parsed = parse_sim_id(row['sim_id'])
    
    # Normalize scenario_name by removing version-specific elements if present
    scenario_name = row['scenario_name']
    
    # Create key from risk_name + scenario_type + patient_profile + scenario_name
    key = f"{row['risk_name']}|{parsed['scenario_type']}|{parsed['patient_profile']}|{scenario_name}"
    
    return key


def parse_matching_key(key: str) -> Dict[str, str]:
    """
    Parse a matching key back into its component parts.
    
    Args:
        key: Matching key string in format 'risk_name|scenario_type|patient_profile|scenario_name'
        
    Returns:
        Dictionary with parsed components
    """
    parts = key.split('|')
    
    if len(parts) >= 4:
        return {
            'risk_name': parts[0],
            'scenario_type': parts[1],
            'patient_profile': parts[2],
            'scenario_name': parts[3]
        }
    else:
        return {
            'risk_name': key,
            'scenario_type': 'unknown',
            'patient_profile': 'unknown',
            'scenario_name': 'unknown'
        }


def format_unmatched_scenarios(keys: List[str]) -> List[Dict[str, str]]:
    """
    Convert list of matching keys to readable scenario descriptions.
    
    Args:
        keys: List of matching key strings
        
    Returns:
        List of dictionaries with parsed scenario information
    """
    scenarios = []
    for key in sorted(keys):
        parsed = parse_matching_key(key)
        scenarios.append(parsed)
    return scenarios


def validate_pairwise_matching(df_v1: pd.DataFrame, df_v2: pd.DataFrame) -> Dict[str, Any]:
    """
    Identify unmatched pairs and report missing scenarios.
    
    Args:
        df_v1: DataFrame from version 1 (previously cleared)
        df_v2: DataFrame from version 2 (2.0)
        
    Returns:
        Dictionary containing validation results and lists of unmatched scenarios
        with detailed identification (risk_name, scenario_type, patient_profile, scenario_name)
    """
    keys_v1 = set(df_v1['matching_key'].unique())
    keys_v2 = set(df_v2['matching_key'].unique())
    
    only_in_v1 = keys_v1 - keys_v2
    only_in_v2 = keys_v2 - keys_v1
    matched = keys_v1 & keys_v2
    
    # Parse unmatched keys into readable format
    unmatched_v1_details = format_unmatched_scenarios(list(only_in_v1))
    unmatched_v2_details = format_unmatched_scenarios(list(only_in_v2))
    
    validation_report = {
        'total_v1': len(keys_v1),
        'total_v2': len(keys_v2),
        'matched': len(matched),
        'only_in_v1': list(only_in_v1),
        'only_in_v2': list(only_in_v2),
        'unmatched_v1_details': unmatched_v1_details,
        'unmatched_v2_details': unmatched_v2_details,
        'is_valid': len(only_in_v1) == 0 and len(only_in_v2) == 0
    }
    
    return validation_report


def merge_paired_data(df_v1: pd.DataFrame, df_v2: pd.DataFrame) -> pd.DataFrame:
    """
    Create merged dataframe with suffixes _v1 and _v2 for comparison.
    
    Args:
        df_v1: DataFrame from version 1
        df_v2: DataFrame from version 2
        
    Returns:
        Merged DataFrame with paired observations
    """
    # Add matching keys
    df_v1 = df_v1.copy()
    df_v2 = df_v2.copy()
    df_v1['matching_key'] = df_v1.apply(create_matching_key, axis=1)
    df_v2['matching_key'] = df_v2.apply(create_matching_key, axis=1)
    
    # Parse sim_id components
    df_v1['scenario_type'] = df_v1['sim_id'].apply(lambda x: parse_sim_id(x)['scenario_type'])
    df_v1['patient_profile'] = df_v1['sim_id'].apply(lambda x: parse_sim_id(x)['patient_profile'])
    df_v2['scenario_type'] = df_v2['sim_id'].apply(lambda x: parse_sim_id(x)['scenario_type'])
    df_v2['patient_profile'] = df_v2['sim_id'].apply(lambda x: parse_sim_id(x)['patient_profile'])
    
    # Select columns for merge
    cols_to_merge = ['matching_key', 'sim_id', 'risk_name', 'scenario_name', 
                     'scenario_type', 'patient_profile',
                     'lbgi', 'dka_index', 'percent_cgm_gt_180',
                     'lbgi_risk_score', 'dka_risk_score']
    
    # Merge on matching key
    df_merged = pd.merge(
        df_v1[cols_to_merge],
        df_v2[cols_to_merge],
        on='matching_key',
        suffixes=('_v1', '_v2'),
        how='inner'
    )
    
    # Calculate differences (v2 - v1)
    df_merged['lbgi_diff'] = df_merged['lbgi_v2'] - df_merged['lbgi_v1']
    df_merged['dkai_diff'] = df_merged['dka_index_v2'] - df_merged['dka_index_v1']
    df_merged['tar_diff'] = df_merged['percent_cgm_gt_180_v2'] - df_merged['percent_cgm_gt_180_v1']
    
    # Use v1 for shared identifiers (should be same as v2 after matching)
    df_merged['risk_name'] = df_merged['risk_name_v1']
    df_merged['scenario_name'] = df_merged['scenario_name_v1']
    df_merged['scenario_type'] = df_merged['scenario_type_v1']
    df_merged['patient_profile'] = df_merged['patient_profile_v1']
    
    return df_merged


def load_and_merge_data(filepath_v1: str, filepath_v2: str) -> Tuple[pd.DataFrame, Dict]:
    """
    Load both datasets and merge them for paired analysis.
    
    Args:
        filepath_v1: Path to version 1 results CSV
        filepath_v2: Path to version 2 results CSV
        
    Returns:
        Tuple of (merged DataFrame, validation report)
    """
    df_v1 = load_simulation_data(filepath_v1)
    df_v2 = load_simulation_data(filepath_v2)
    
    # Add matching keys for validation
    df_v1_temp = df_v1.copy()
    df_v2_temp = df_v2.copy()
    df_v1_temp['matching_key'] = df_v1_temp.apply(create_matching_key, axis=1)
    df_v2_temp['matching_key'] = df_v2_temp.apply(create_matching_key, axis=1)
    
    validation_report = validate_pairwise_matching(df_v1_temp, df_v2_temp)
    
    df_merged = merge_paired_data(df_v1, df_v2)
    
    return df_merged, validation_report


# =============================================================================
# Module 2: Question 1 Analysis - Overall Risk Profile
# =============================================================================

def calc_range_difference(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Calculate range comparison between versions.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name (lbgi, dka_index, percent_cgm_gt_180)
        
    Returns:
        Dictionary with range statistics for each version
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    v1_min, v1_max = df[v1_col].min(), df[v1_col].max()
    v2_min, v2_max = df[v2_col].min(), df[v2_col].max()
    
    return {
        'v1_min': v1_min,
        'v1_max': v1_max,
        'v1_range': v1_max - v1_min,
        'v2_min': v2_min,
        'v2_max': v2_max,
        'v2_range': v2_max - v2_min,
        'range_diff': (v2_max - v2_min) - (v1_max - v1_min)
    }


def calc_mean_difference(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Calculate mean comparison between versions.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with mean statistics
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    v1_mean = df[v1_col].mean()
    v2_mean = df[v2_col].mean()
    
    return {
        'v1_mean': v1_mean,
        'v2_mean': v2_mean,
        'mean_diff': v2_mean - v1_mean,
        'percent_change': ((v2_mean - v1_mean) / v1_mean * 100) if v1_mean != 0 else np.nan
    }


def calc_wilcoxon_signed_rank(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Perform Wilcoxon signed-rank test for paired samples.
    
    Non-parametric test appropriate for non-normally distributed severity data.
    Note: Wilcoxon excludes pairs with zero difference by design.
    
    Also calculates rank-biserial correlation (r) as an effect size measure
    that is consistent with the Wilcoxon test (uses same non-zero subset).
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with test results including n_non_zero count and rank-biserial r
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    # Calculate differences and count non-zero
    diff = df[v2_col] - df[v1_col]
    n_total = len(diff)
    n_non_zero = (diff != 0).sum()
    n_zero = n_total - n_non_zero
    pct_changed = (n_non_zero / n_total * 100) if n_total > 0 else 0
    
    if n_non_zero < 10:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n_total': n_total,
            'n_non_zero': n_non_zero,
            'n_zero': n_zero,
            'pct_changed': pct_changed,
            'rank_biserial_r': np.nan,
            'r_interpretation': 'N/A',
            'note': f'Insufficient non-zero differences ({n_non_zero}) for Wilcoxon test'
        }
    
    try:
        # Get non-zero differences for rank-biserial calculation
        non_zero_diff = diff[diff != 0].values
        
        # Perform Wilcoxon test
        statistic, p_value = stats.wilcoxon(df[v1_col], df[v2_col], alternative='two-sided')
        
        # Calculate rank-biserial correlation
        # For Wilcoxon signed-rank: r = 1 - (2*W) / (n*(n+1)/2)
        # where W is the test statistic and n is number of non-zero differences
        n = len(non_zero_diff)
        max_w = n * (n + 1) / 2
        rank_biserial_r = 1 - (2 * statistic) / max_w if max_w > 0 else 0
        
        # Interpret rank-biserial r (same thresholds as correlation)
        abs_r = abs(rank_biserial_r)
        if abs_r < 0.1:
            r_interpretation = 'negligible'
        elif abs_r < 0.3:
            r_interpretation = 'small'
        elif abs_r < 0.5:
            r_interpretation = 'medium'
        else:
            r_interpretation = 'large'
        
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < ALPHA,
            'n_total': n_total,
            'n_non_zero': n_non_zero,
            'n_zero': n_zero,
            'pct_changed': pct_changed,
            'rank_biserial_r': rank_biserial_r,
            'r_interpretation': r_interpretation,
            'note': None
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'n_total': n_total,
            'n_non_zero': n_non_zero,
            'n_zero': n_zero,
            'pct_changed': pct_changed,
            'rank_biserial_r': np.nan,
            'r_interpretation': 'N/A',
            'note': str(e)
        }


def calc_cohens_d_paired(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Calculate Cohen's d effect size for paired samples.
    
    Calculates both:
    - Standard Cohen's d (on all differences, including zeros)
    - Cohen's d on non-zero differences only (for zero-inflated data)
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with effect sizes and interpretations
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    diff = df[v2_col] - df[v1_col]
    
    # Standard Cohen's d (all differences)
    d_all = diff.mean() / diff.std() if diff.std() != 0 else 0
    
    # Cohen's d on non-zero differences only
    non_zero_diff = diff[diff != 0]
    if len(non_zero_diff) >= 2 and non_zero_diff.std() != 0:
        d_nonzero = non_zero_diff.mean() / non_zero_diff.std()
    else:
        d_nonzero = np.nan
    
    # Interpret effect sizes
    def interpret_d(d_val):
        if np.isnan(d_val):
            return 'N/A'
        abs_d = abs(d_val)
        if abs_d < 0.2:
            return 'negligible'
        elif abs_d < 0.5:
            return 'small'
        elif abs_d < 0.8:
            return 'medium'
        else:
            return 'large'
    
    return {
        'd': d_all,  # Keep for backward compatibility
        'cohens_d': d_all,
        'interpretation': interpret_d(d_all),
        'd_nonzero': d_nonzero,
        'interpretation_nonzero': interpret_d(d_nonzero),
        'n_nonzero': len(non_zero_diff)
    }


def calc_paired_ttest(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Perform paired samples t-test.
    
    Parametric test included for comparison/validation.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with test results
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    try:
        statistic, p_value = stats.ttest_rel(df[v1_col], df[v2_col])
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < ALPHA
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'note': str(e)
        }


def calc_std_difference(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Calculate standard deviation comparison between versions.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with standard deviation statistics
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    v1_std = df[v1_col].std()
    v2_std = df[v2_col].std()
    
    return {
        'v1_std': v1_std,
        'v2_std': v2_std,
        'std_diff': v2_std - v1_std
    }


def calc_median_difference(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Calculate median comparison between versions.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with median statistics
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    v1_median = df[v1_col].median()
    v2_median = df[v2_col].median()
    
    return {
        'v1_median': v1_median,
        'v2_median': v2_median,
        'median_diff': v2_median - v1_median
    }


def run_question1_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Run all Question 1 statistical tests for all metrics.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Nested dictionary with results for each metric and test
    """
    results = {}
    
    for metric_col, metric_name in METRICS.items():
        results[metric_col] = {
            'metric_name': metric_name,
            'n_pairs': len(df),
            'range': calc_range_difference(df, metric_col),
            'mean': calc_mean_difference(df, metric_col),
            'median': calc_median_difference(df, metric_col),
            'std': calc_std_difference(df, metric_col),
            'wilcoxon': calc_wilcoxon_signed_rank(df, metric_col),
            'ttest': calc_paired_ttest(df, metric_col),
            'cohens_d': calc_cohens_d_paired(df, metric_col)
        }
    
    return results


# =============================================================================
# Module 3: Question 2 Analysis - Patient Profile Stratification
# =============================================================================

def run_question2_analysis(df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
    """
    Run stratified analysis by patient profile.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        Nested dictionary with results for each profile and metric
    """
    results = {}
    
    for profile in PATIENT_PROFILES:
        df_profile = df[df['patient_profile'] == profile]
        
        if len(df_profile) == 0:
            results[profile] = {'error': 'No data for this profile'}
            continue
        
        results[profile] = {
            'n_pairs': len(df_profile)
        }
        
        for metric_col, metric_name in METRICS.items():
            results[profile][metric_col] = {
                'metric_name': metric_name,
                'mean': calc_mean_difference(df_profile, metric_col),
                'median': calc_median_difference(df_profile, metric_col),
                'wilcoxon': calc_wilcoxon_signed_rank(df_profile, metric_col),
                'ttest': calc_paired_ttest(df_profile, metric_col),
                'cohens_d': calc_cohens_d_paired(df_profile, metric_col)
            }
    
    # Add Kruskal-Wallis test for differences between profiles within v2
    results['between_profile_tests'] = {}
    for metric_col, metric_name in METRICS.items():
        results['between_profile_tests'][metric_col] = compare_profiles_kruskal(df, metric_col)
    
    return results


def compare_profiles_kruskal(df: pd.DataFrame, metric: str) -> Dict[str, Any]:
    """
    Test for significant differences between profiles using Kruskal-Wallis.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        
    Returns:
        Dictionary with test results
    """
    v2_col = f"{metric}_v2"
    
    groups = []
    for profile in PATIENT_PROFILES:
        profile_data = df[df['patient_profile'] == profile][v2_col].dropna()
        if len(profile_data) > 0:
            groups.append(profile_data)
    
    if len(groups) < 2:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'note': 'Insufficient groups for comparison'
        }
    
    try:
        statistic, p_value = stats.kruskal(*groups)
        return {
            'statistic': statistic,
            'p_value': p_value,
            'significant': p_value < ALPHA
        }
    except Exception as e:
        return {
            'statistic': np.nan,
            'p_value': np.nan,
            'significant': False,
            'note': str(e)
        }


# =============================================================================
# Module 4: Question 3 Analysis - Individual Hazardous Situations
# =============================================================================

def calculate_hyperglycemia_score(tar_value: float) -> int:
    """
    Calculate hyperglycemia score based on TAR percentage.
    
    Uses same logic as create_severity_summary.py.
    
    Args:
        tar_value: TAR percentage (percent_cgm_gt_180)
        
    Returns:
        Integer score: 0 if TAR=0, 1 if TAR<12%, 2 if TAR>=12%
    """
    if pd.isna(tar_value):
        return 1
    
    if tar_value == 0.0:
        return 0
    elif tar_value < 12.0:
        return 1
    else:
        return 2


def determine_harm_and_severity(lbgi_score: int, dka_score: int, hyperglycemia_score: int) -> Tuple[str, int]:
    """
    Determine harm type and severity score based on risk scores.
    
    Uses same logic as create_severity_summary.py.
    
    Args:
        lbgi_score: Integer LBGI risk score (rounded average)
        dka_score: Integer DKA risk score (rounded average)
        hyperglycemia_score: Integer hyperglycemia score (0, 1, or 2)
        
    Returns:
        Tuple of (harm_type, severity_score)
    """
    # If all three scores are 0, severity is equal to baseline
    if lbgi_score == 0 and dka_score == 0 and hyperglycemia_score == 0:
        return ("Baseline", 0)
    
    # If both lbgi and dka are baseline, use hyperglycemia
    if lbgi_score <= 1 and dka_score == 0:
        return ("Hyperglycemia", hyperglycemia_score)
    
    # If lbgi >= dka (lbgi takes priority in ties), use hypoglycemia
    if lbgi_score >= dka_score:
        return ("Hypoglycemia", lbgi_score)
    
    # Otherwise dka > lbgi, use DKA
    return ("DKA", dka_score)


def aggregate_risk_by_scenario_type(df: pd.DataFrame, version_suffix: str) -> pd.DataFrame:
    """
    Aggregate risk metrics by risk_name and scenario_type.
    
    For each risk_name + scenario_type combination, calculates:
    - Average LBGI risk score (rounded to int)
    - Average DKA risk score (rounded to int)
    - Average TAR%
    - Derived hyperglycemia score
    - Determined harm type and severity
    
    Args:
        df: Merged DataFrame
        version_suffix: '_v1' or '_v2'
        
    Returns:
        DataFrame with aggregated results per risk_name and scenario_type
    """
    lbgi_col = f'lbgi_risk_score{version_suffix}'
    dka_col = f'dka_risk_score{version_suffix}'
    tar_col = f'percent_cgm_gt_180{version_suffix}'
    
    # Group by risk_name and scenario_type
    grouped = df.groupby(['risk_name', 'scenario_type']).agg({
        lbgi_col: 'mean',
        dka_col: 'mean',
        tar_col: 'mean'
    }).reset_index()
    
    # Round LBGI and DKA scores to integers using round-half-up (conservative)
    grouped['lbgi_avg'] = grouped[lbgi_col].apply(round_half_up)
    grouped['dka_avg'] = grouped[dka_col].apply(round_half_up)
    grouped['tar_avg'] = grouped[tar_col]
    
    # Calculate hyperglycemia score from TAR
    grouped['hyper_score'] = grouped['tar_avg'].apply(calculate_hyperglycemia_score)
    
    # Determine harm and severity for each row
    harm_severity = grouped.apply(
        lambda row: determine_harm_and_severity(
            row['lbgi_avg'], row['dka_avg'], row['hyper_score']
        ), axis=1
    )
    grouped['harm'] = harm_severity.apply(lambda x: x[0])
    grouped['severity'] = harm_severity.apply(lambda x: x[1])
    
    # Count number of profiles aggregated
    profile_counts = df.groupby(['risk_name', 'scenario_type']).size().reset_index(name='n_profiles')
    grouped = grouped.merge(profile_counts, on=['risk_name', 'scenario_type'])
    
    return grouped[['risk_name', 'scenario_type', 'lbgi_avg', 'dka_avg', 'tar_avg', 
                    'hyper_score', 'harm', 'severity', 'n_profiles']]


def detect_severity_threshold_crossing_aggregated(sev_v1: int, sev_v2: int) -> Optional[Dict[str, Any]]:
    """
    Check if severity crosses the {1,2} <-> {3,4,5} boundary.
    
    Args:
        sev_v1: Severity score for version 1
        sev_v2: Severity score for version 2
        
    Returns:
        Dictionary with crossing details if detected, None otherwise
    """
    low_severity = {0, 1, 2}
    high_severity = {3, 4, 5}
    
    crossing_detected = False
    direction = None
    
    if sev_v1 in low_severity and sev_v2 in high_severity:
        crossing_detected = True
        direction = 'increased'
    elif sev_v1 in high_severity and sev_v2 in low_severity:
        crossing_detected = True
        direction = 'decreased'
    
    if crossing_detected:
        return {
            'change_type': 'severity_threshold_crossing',
            'direction': direction,
            'v1_severity': sev_v1,
            'v2_severity': sev_v2
        }
    
    return None


def detect_harm_type_change_aggregated(harm_v1: str, harm_v2: str) -> Optional[Dict[str, Any]]:
    """
    Check if primary harm changes between hyperglycemia and DKA.
    
    Args:
        harm_v1: Harm type for version 1
        harm_v2: Harm type for version 2
        
    Returns:
        Dictionary with change details if detected, None otherwise
    """
    # Only flag changes between Hyperglycemia and DKA
    relevant_harms = {'Hyperglycemia', 'DKA'}
    
    if harm_v1 in relevant_harms and harm_v2 in relevant_harms and harm_v1 != harm_v2:
        return {
            'change_type': 'harm_type_change',
            'v1_harm': harm_v1,
            'v2_harm': harm_v2
        }
    
    return None


def run_question3_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify hazardous situations with practically significant changes.
    
    Aggregates by risk_name and scenario_type (pre-mitigation, noLoop, post-mitigation),
    calculating total severity using the same logic as create_severity_summary.py.
    
    Args:
        df: Merged DataFrame
        
    Returns:
        DataFrame containing aggregated severity by risk and scenario type,
        with flags for significant changes
    """
    # Aggregate for each version
    agg_v1 = aggregate_risk_by_scenario_type(df, '_v1')
    agg_v2 = aggregate_risk_by_scenario_type(df, '_v2')
    
    # Merge the two versions
    merged = pd.merge(
        agg_v1, agg_v2,
        on=['risk_name', 'scenario_type'],
        suffixes=('_v1', '_v2')
    )
    
    # Detect significant changes
    results = []
    
    for _, row in merged.iterrows():
        changes = []
        
        # Check severity threshold crossing
        severity_change = detect_severity_threshold_crossing_aggregated(
            row['severity_v1'], row['severity_v2']
        )
        if severity_change:
            changes.append(severity_change)
        
        # Check harm type change
        harm_change = detect_harm_type_change_aggregated(
            row['harm_v1'], row['harm_v2']
        )
        if harm_change:
            changes.append(harm_change)
        
        # Check for severity decrease (even if no threshold crossing)
        severity_diff = row['severity_v2'] - row['severity_v1']
        if severity_diff < 0 and not any(c.get('change_type') == 'severity_threshold_crossing' for c in changes):
            changes.append({
                'change_type': 'severity_decreased',
                'v1_severity': row['severity_v1'],
                'v2_severity': row['severity_v2']
            })
        
        # Check for severity increase (even if no threshold crossing)
        if severity_diff > 0 and not any(c.get('change_type') == 'severity_threshold_crossing' for c in changes):
            changes.append({
                'change_type': 'severity_increased',
                'v1_severity': row['severity_v1'],
                'v2_severity': row['severity_v2']
            })
        
        # Determine change type string
        change_types = [c['change_type'] for c in changes] if changes else []
        
        result = {
            'risk_name': row['risk_name'],
            'scenario_type': row['scenario_type'],
            'n_profiles': row['n_profiles_v1'],
            'harm_v1': row['harm_v1'],
            'severity_v1': row['severity_v1'],
            'lbgi_avg_v1': row['lbgi_avg_v1'],
            'dka_avg_v1': row['dka_avg_v1'],
            'tar_avg_v1': row['tar_avg_v1'],
            'harm_v2': row['harm_v2'],
            'severity_v2': row['severity_v2'],
            'lbgi_avg_v2': row['lbgi_avg_v2'],
            'dka_avg_v2': row['dka_avg_v2'],
            'tar_avg_v2': row['tar_avg_v2'],
            'severity_change': severity_diff,
            'has_significant_change': len(changes) > 0,
            'change_types': ', '.join(change_types) if change_types else 'None'
        }
        results.append(result)
    
    return pd.DataFrame(results)


# =============================================================================
# Module 5: Acceptance Criteria Evaluation
# =============================================================================

def evaluate_acceptance_criteria(q1_results: Dict, q2_results: Dict) -> Dict[str, Any]:
    """
    Determine overall acceptability per Section 10 criteria (reduced scope).
    
    Criteria:
    1. No significant overall increase in severity indices (Question 1)
    2. No virtual patient profile has significantly different indices (Question 2)
    
    Args:
        q1_results: Results from Question 1 analysis
        q2_results: Results from Question 2 analysis
        
    Returns:
        Dictionary with pass/fail for each criterion and overall determination
    """
    criteria_results = {}
    
    # Criterion 1: No significant overall increase in severity indices
    criterion1_details = {}
    criterion1_pass = True
    
    for metric_col in METRICS.keys():
        wilcoxon_result = q1_results[metric_col]['wilcoxon']
        mean_result = q1_results[metric_col]['mean']
        
        # Check if significant AND increased
        is_significant = wilcoxon_result['significant']
        is_increased = mean_result['mean_diff'] > 0
        
        criterion1_details[metric_col] = {
            'significant': is_significant,
            'direction': 'increased' if is_increased else 'decreased/unchanged',
            'p_value': wilcoxon_result['p_value'],
            'mean_diff': mean_result['mean_diff'],
            'passed': not (is_significant and is_increased)
        }
        
        if is_significant and is_increased:
            criterion1_pass = False
    
    criteria_results['criterion1'] = {
        'description': 'No significant overall increase in severity indices',
        'passed': criterion1_pass,
        'details': criterion1_details
    }
    
    # Criterion 2: No virtual patient profile has significantly different indices
    criterion2_details = {}
    criterion2_pass = True
    
    for profile in PATIENT_PROFILES:
        if profile not in q2_results or 'error' in q2_results[profile]:
            criterion2_details[profile] = {'error': 'No data'}
            continue
        
        profile_results = {}
        for metric_col in METRICS.keys():
            if metric_col not in q2_results[profile]:
                continue
            wilcoxon_result = q2_results[profile][metric_col]['wilcoxon']
            mean_result = q2_results[profile][metric_col]['mean']
            
            is_significant = wilcoxon_result['significant']
            is_increased = mean_result['mean_diff'] > 0
            
            profile_results[metric_col] = {
                'significant': is_significant,
                'direction': 'increased' if is_increased else 'decreased/unchanged',
                'p_value': wilcoxon_result['p_value'],
                'mean_diff': mean_result['mean_diff'],
                'passed': not (is_significant and is_increased)
            }
            
            if is_significant and is_increased:
                criterion2_pass = False
        
        criterion2_details[profile] = profile_results
    
    criteria_results['criterion2'] = {
        'description': 'No virtual patient profile has significantly different indices',
        'passed': criterion2_pass,
        'details': criterion2_details
    }
    
    # Overall determination
    criteria_results['overall'] = {
        'passed': criterion1_pass and criterion2_pass,
        'summary': 'ACCEPTABLE' if (criterion1_pass and criterion2_pass) else 'REQUIRES REVIEW'
    }
    
    return criteria_results


def generate_acceptance_summary(acceptance_results: Dict) -> str:
    """
    Generate formatted summary of acceptance criteria evaluation.
    
    Args:
        acceptance_results: Results from evaluate_acceptance_criteria()
        
    Returns:
        Formatted string summary
    """
    lines = []
    lines.append("=" * 70)
    lines.append("ACCEPTANCE CRITERIA EVALUATION SUMMARY")
    lines.append("=" * 70)
    
    # Criterion 1
    c1 = acceptance_results['criterion1']
    status = "PASS" if c1['passed'] else "FAIL"
    lines.append(f"\nCriterion 1: {c1['description']}")
    lines.append(f"Status: {status}")
    
    for metric, details in c1['details'].items():
        metric_name = METRICS.get(metric, metric)
        p_val = details['p_value']
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        lines.append(f"  {metric_name}: p={p_str}, diff={details['mean_diff']:.4f} ({details['direction']})")
    
    # Criterion 2
    c2 = acceptance_results['criterion2']
    status = "PASS" if c2['passed'] else "FAIL"
    lines.append(f"\nCriterion 2: {c2['description']}")
    lines.append(f"Status: {status}")
    
    for profile, profile_results in c2['details'].items():
        if 'error' in profile_results:
            lines.append(f"  {profile.capitalize()}: {profile_results['error']}")
            continue
        
        lines.append(f"  {profile.capitalize()}:")
        for metric, details in profile_results.items():
            metric_name = METRICS.get(metric, metric)
            p_val = details['p_value']
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            lines.append(f"    {metric_name}: p={p_str}, diff={details['mean_diff']:.4f}")
    
    # Overall
    lines.append("\n" + "=" * 70)
    overall = acceptance_results['overall']
    lines.append(f"OVERALL DETERMINATION: {overall['summary']}")
    lines.append("=" * 70)
    
    return "\n".join(lines)


# =============================================================================
# Module 6: Visualization
# =============================================================================

def plot_combined_scatterplot(df: pd.DataFrame, metric: str, output_dir: str) -> str:
    """
    Create V1 vs V2 scatter plot with identity line, colored by patient profile.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        output_dir: Directory to save plot
        
    Returns:
        Path to saved figure
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create scatter plot colored by patient profile
    colors = {'adolescent': '#1f77b4', 'sensitive': '#ff7f0e', 
              'resistant': '#2ca02c', 'median': '#d62728', 'unknown': '#7f7f7f'}
    
    for profile in PATIENT_PROFILES + ['unknown']:
        mask = df['patient_profile'] == profile
        if mask.sum() > 0:
            label = profile.capitalize() if profile != 'unknown' else 'Unknown/Other'
            ax.scatter(df.loc[mask, v1_col], df.loc[mask, v2_col], 
                       c=colors.get(profile, 'gray'), label=label, 
                       alpha=0.6, s=30)
    
    # Add identity line
    lims = [
        np.min([ax.get_xlim()[0], ax.get_ylim()[0]]),
        np.max([ax.get_xlim()[1], ax.get_ylim()[1]])
    ]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Identity line')
    
    ax.set_xlabel(f'{METRICS[metric]} - Previously Cleared Version')
    ax.set_ylabel(f'{METRICS[metric]} - Version 2.0')
    ax.set_title(f'{METRICS[metric]}: Version Comparison')
    ax.legend()
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'scatter_{metric}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath


def plot_difference_distribution(df: pd.DataFrame, metric: str, output_dir: str) -> str:
    """
    Create histogram of paired differences.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        output_dir: Directory to save plot
        
    Returns:
        Path to saved figure
    """
    diff_col = f"{metric.replace('dka_index', 'dkai').replace('percent_cgm_gt_180', 'tar')}_diff"
    
    # Handle column name mapping
    if 'dka_index' in metric:
        diff_col = 'dkai_diff'
    elif 'percent_cgm_gt_180' in metric:
        diff_col = 'tar_diff'
    else:
        diff_col = 'lbgi_diff'
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.hist(df[diff_col], bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No change')
    ax.axvline(x=df[diff_col].mean(), color='blue', linestyle='-', linewidth=2, 
               label=f'Mean: {df[diff_col].mean():.3f}')
    
    ax.set_xlabel(f'Difference (V2.0 - Previously Cleared)')
    ax.set_ylabel('Count')
    ax.set_title(f'{METRICS[metric]}: Distribution of Paired Differences')
    ax.legend()
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'hist_diff_{metric}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath


def plot_bland_altman(df: pd.DataFrame, metric: str, output_dir: str) -> str:
    """
    Create Bland-Altman plot (mean vs difference with limits of agreement).
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        output_dir: Directory to save plot
        
    Returns:
        Path to saved figure
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    mean_vals = (df[v1_col] + df[v2_col]) / 2
    diff_vals = df[v2_col] - df[v1_col]
    
    mean_diff = diff_vals.mean()
    std_diff = diff_vals.std()
    upper_loa = mean_diff + 1.96 * std_diff
    lower_loa = mean_diff - 1.96 * std_diff
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(mean_vals, diff_vals, alpha=0.5, s=30)
    
    ax.axhline(y=mean_diff, color='blue', linestyle='-', label=f'Mean: {mean_diff:.3f}')
    ax.axhline(y=upper_loa, color='red', linestyle='--', 
               label=f'+1.96 SD: {upper_loa:.3f}')
    ax.axhline(y=lower_loa, color='red', linestyle='--', 
               label=f'-1.96 SD: {lower_loa:.3f}')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    ax.set_xlabel(f'Mean of Both Versions')
    ax.set_ylabel(f'Difference (V2.0 - Previously Cleared)')
    ax.set_title(f'{METRICS[metric]}: Bland-Altman Plot')
    ax.legend()
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'bland_altman_{metric}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath


def plot_profile_comparison_boxplot(df: pd.DataFrame, metric: str, output_dir: str) -> str:
    """
    Create side-by-side boxplots by patient profile and version.
    
    Args:
        df: Merged DataFrame
        metric: Metric column name
        output_dir: Directory to save plot
        
    Returns:
        Path to saved figure
    """
    v1_col = f"{metric}_v1"
    v2_col = f"{metric}_v2"
    
    # Filter out unknown profiles
    df_filtered = df[df['patient_profile'].isin(PATIENT_PROFILES)].copy()
    
    if len(df_filtered) == 0:
        # No valid profiles to plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.text(0.5, 0.5, 'No data with recognized patient profiles', 
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title(f'{METRICS[metric]}: Comparison by Patient Profile')
        filepath = os.path.join(output_dir, f'boxplot_profile_{metric}.png')
        plt.savefig(filepath, dpi=150)
        plt.close()
        return filepath
    
    # Reshape data for seaborn
    df_long = pd.DataFrame({
        'Patient Profile': list(df_filtered['patient_profile'].str.capitalize()) * 2,
        'Version': ['Previously Cleared'] * len(df_filtered) + ['Version 2.0'] * len(df_filtered),
        'Value': list(df_filtered[v1_col]) + list(df_filtered[v2_col])
    })
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    sns.boxplot(data=df_long, x='Patient Profile', y='Value', hue='Version', ax=ax)
    
    ax.set_xlabel('Patient Profile')
    ax.set_ylabel(METRICS[metric])
    ax.set_title(f'{METRICS[metric]}: Comparison by Patient Profile')
    
    # Add count annotation
    unknown_count = len(df) - len(df_filtered)
    if unknown_count > 0:
        ax.annotate(f'Note: {unknown_count} rows with unrecognized profiles excluded',
                    xy=(0.02, 0.98), xycoords='axes fraction', fontsize=8,
                    ha='left', va='top', style='italic', color='gray')
    
    plt.tight_layout()
    
    filepath = os.path.join(output_dir, f'boxplot_profile_{metric}.png')
    plt.savefig(filepath, dpi=150)
    plt.close()
    
    return filepath


def generate_all_visualizations(df: pd.DataFrame, output_dir: str) -> List[str]:
    """
    Generate all visualization figures.
    
    Args:
        df: Merged DataFrame
        output_dir: Directory to save plots
        
    Returns:
        List of paths to generated figures
    """
    os.makedirs(output_dir, exist_ok=True)
    
    figure_paths = []
    
    for metric in METRICS.keys():
        figure_paths.append(plot_combined_scatterplot(df, metric, output_dir))
        figure_paths.append(plot_difference_distribution(df, metric, output_dir))
        figure_paths.append(plot_bland_altman(df, metric, output_dir))
        figure_paths.append(plot_profile_comparison_boxplot(df, metric, output_dir))
    
    return figure_paths


# =============================================================================
# Module 7: Report Generation
# =============================================================================

def generate_table2(q1_results: Dict) -> pd.DataFrame:
    """
    Format Question 1 results into Table 2 structure.
    
    Args:
        q1_results: Results from run_question1_analysis()
        
    Returns:
        DataFrame formatted per Table 2 shell
    """
    rows = []
    
    for metric_col, results in q1_results.items():
        metric_name = results['metric_name']
        
        # Range
        rows.append({
            'Metric': metric_name,
            'Test': 'Range',
            'V1 Result': f"{results['range']['v1_min']:.3f} - {results['range']['v1_max']:.3f}",
            'V2 Result': f"{results['range']['v2_min']:.3f} - {results['range']['v2_max']:.3f}",
            'Difference': f"{results['range']['range_diff']:.3f}",
            'p-value': '-',
            'Acceptability': '-'
        })
        
        # Mean
        wilcoxon_sig = results['wilcoxon']['significant']
        mean_increased = results['mean']['mean_diff'] > 0
        accept = 'PASS' if not (wilcoxon_sig and mean_increased) else 'REVIEW'
        
        rows.append({
            'Metric': metric_name,
            'Test': 'Mean',
            'V1 Result': f"{results['mean']['v1_mean']:.3f}",
            'V2 Result': f"{results['mean']['v2_mean']:.3f}",
            'Difference': f"{results['mean']['mean_diff']:.3f}",
            'p-value': '-',
            'Acceptability': '-'
        })
        
        # Wilcoxon with rank-biserial r
        p_val = results['wilcoxon']['p_value']
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        pct_changed = results['wilcoxon'].get('pct_changed', 0)
        n_non_zero = results['wilcoxon'].get('n_non_zero', '-')
        wilcoxon_note = f"({pct_changed:.1f}% changed, n={n_non_zero})"
        rows.append({
            'Metric': metric_name,
            'Test': 'Wilcoxon signed-rank',
            'V1 Result': '-',
            'V2 Result': '-',
            'Difference': wilcoxon_note,
            'p-value': p_str,
            'Acceptability': accept
        })
        
        # Rank-biserial correlation (effect size for Wilcoxon)
        r_val = results['wilcoxon'].get('rank_biserial_r', np.nan)
        r_interp = results['wilcoxon'].get('r_interpretation', 'N/A')
        r_str = f"{r_val:.3f}" if not np.isnan(r_val) else "N/A"
        rows.append({
            'Metric': metric_name,
            'Test': 'Rank-biserial r (non-zero)',
            'V1 Result': '-',
            'V2 Result': '-',
            'Difference': f"{r_str} ({r_interp})",
            'p-value': '-',
            'Acceptability': '-'
        })
        
        # Cohen's d (all differences)
        rows.append({
            'Metric': metric_name,
            'Test': "Cohen's d (all pairs)",
            'V1 Result': '-',
            'V2 Result': '-',
            'Difference': f"{results['cohens_d']['cohens_d']:.3f} ({results['cohens_d']['interpretation']})",
            'p-value': '-',
            'Acceptability': '-'
        })
        
        # Cohen's d (non-zero differences only)
        d_nz = results['cohens_d'].get('d_nonzero', np.nan)
        d_nz_interp = results['cohens_d'].get('interpretation_nonzero', 'N/A')
        d_nz_str = f"{d_nz:.3f}" if not np.isnan(d_nz) else "N/A"
        rows.append({
            'Metric': metric_name,
            'Test': "Cohen's d (non-zero only)",
            'V1 Result': '-',
            'V2 Result': '-',
            'Difference': f"{d_nz_str} ({d_nz_interp})",
            'p-value': '-',
            'Acceptability': '-'
        })
        
        # Paired t-test
        p_val = results['ttest']['p_value']
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        rows.append({
            'Metric': metric_name,
            'Test': 'Paired t-test',
            'V1 Result': '-',
            'V2 Result': '-',
            'Difference': '-',
            'p-value': p_str,
            'Acceptability': '-'
        })
        
        # Std
        rows.append({
            'Metric': metric_name,
            'Test': 'Std Dev',
            'V1 Result': f"{results['std']['v1_std']:.3f}",
            'V2 Result': f"{results['std']['v2_std']:.3f}",
            'Difference': f"{results['std']['std_diff']:.3f}",
            'p-value': '-',
            'Acceptability': '-'
        })
        
        # Median
        rows.append({
            'Metric': metric_name,
            'Test': 'Median',
            'V1 Result': f"{results['median']['v1_median']:.3f}",
            'V2 Result': f"{results['median']['v2_median']:.3f}",
            'Difference': f"{results['median']['median_diff']:.3f}",
            'p-value': '-',
            'Acceptability': '-'
        })
    
    return pd.DataFrame(rows)


def generate_table3(q2_results: Dict) -> pd.DataFrame:
    """
    Format Question 2 results into Table 3 structure.
    
    Includes Mean, Median, and Wilcoxon test for each profile/metric.
    Median is particularly relevant for LBGI (hypoglycemia) due to right skew.
    
    Args:
        q2_results: Results from run_question2_analysis()
        
    Returns:
        DataFrame formatted per Table 3 shell
    """
    rows = []
    
    for profile in PATIENT_PROFILES:
        if profile not in q2_results or 'error' in q2_results.get(profile, {}):
            continue
        
        profile_data = q2_results[profile]
        
        for metric_col, metric_name in METRICS.items():
            if metric_col not in profile_data:
                continue
            
            results = profile_data[metric_col]
            
            # Determine acceptability based on Wilcoxon significance and direction
            wilcoxon_sig = results['wilcoxon']['significant']
            mean_increased = results['mean']['mean_diff'] > 0
            accept = 'PASS' if not (wilcoxon_sig and mean_increased) else 'REVIEW'
            
            # Mean row
            rows.append({
                'Patient Profile': profile.capitalize(),
                'Metric': metric_name,
                'Test': 'Mean',
                'V1 Result': f"{results['mean']['v1_mean']:.3f}",
                'V2 Result': f"{results['mean']['v2_mean']:.3f}",
                'Difference': f"{results['mean']['mean_diff']:.3f}",
                'p-value': '-',
                'Acceptability': '-'
            })
            
            # Median row (important for skewed distributions like LBGI)
            rows.append({
                'Patient Profile': profile.capitalize(),
                'Metric': metric_name,
                'Test': 'Median',
                'V1 Result': f"{results['median']['v1_median']:.3f}",
                'V2 Result': f"{results['median']['v2_median']:.3f}",
                'Difference': f"{results['median']['median_diff']:.3f}",
                'p-value': '-',
                'Acceptability': '-'
            })
            
            # Wilcoxon row (tests ranks, aligns with median)
            p_val = results['wilcoxon']['p_value']
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            pct_changed = results['wilcoxon'].get('pct_changed', 0)
            n_non_zero = results['wilcoxon'].get('n_non_zero', '-')
            wilcoxon_note = f"({pct_changed:.1f}% changed, n={n_non_zero})"
            
            rows.append({
                'Patient Profile': profile.capitalize(),
                'Metric': metric_name,
                'Test': 'Wilcoxon',
                'V1 Result': '-',
                'V2 Result': '-',
                'Difference': wilcoxon_note,
                'p-value': p_str,
                'Acceptability': accept
            })
    
    return pd.DataFrame(rows)


def generate_table4(q3_results: pd.DataFrame) -> pd.DataFrame:
    """
    Format Question 3 results into Table 4 structure.
    
    Shows aggregated severity by risk_name and scenario_type.
    
    Args:
        q3_results: DataFrame from run_question3_analysis()
        
    Returns:
        DataFrame formatted per Table 4 shell
    """
    if len(q3_results) == 0:
        return pd.DataFrame(columns=['TLR Key', 'Scenario Type', 'N Profiles',
                                      'Harm V1', 'Severity V1', 'Harm V2', 
                                      'Severity V2', 'Severity Change', 'Change Type'])
    
    # Create output table
    table = q3_results[['risk_name', 'scenario_type', 'n_profiles',
                        'harm_v1', 'severity_v1', 'harm_v2', 'severity_v2',
                        'severity_change', 'change_types']].copy()
    
    table.columns = ['TLR Key', 'Scenario Type', 'N Profiles',
                     'Harm V1', 'Severity V1', 'Harm V2', 'Severity V2',
                     'Severity Change', 'Change Type']
    
    # Sort by TLR Key and Scenario Type
    scenario_order = {'pre-mitigation': 0, 'noLoop': 1, 'post-mitigation': 2}
    table['_sort'] = table['Scenario Type'].map(scenario_order)
    table = table.sort_values(['TLR Key', '_sort']).drop(columns=['_sort'])
    
    return table.reset_index(drop=True)


def export_unmatched_scenarios_to_csv(validation_report: Dict, output_dir: str) -> Optional[str]:
    """
    Export unmatched scenarios to a CSV file for review.
    
    Args:
        validation_report: Validation report from validate_pairwise_matching()
        output_dir: Output directory
        
    Returns:
        Path to exported file, or None if no unmatched scenarios
    """
    rows = []
    
    # Add V1-only scenarios
    for scenario in validation_report.get('unmatched_v1_details', []):
        rows.append({
            'Source': 'V1 (Previously Cleared)',
            'Risk Name': scenario['risk_name'],
            'Scenario Type': scenario['scenario_type'],
            'Patient Profile': scenario['patient_profile'],
            'Scenario Name': scenario['scenario_name']
        })
    
    # Add V2-only scenarios
    for scenario in validation_report.get('unmatched_v2_details', []):
        rows.append({
            'Source': 'V2 (Version 2.0)',
            'Risk Name': scenario['risk_name'],
            'Scenario Type': scenario['scenario_type'],
            'Patient Profile': scenario['patient_profile'],
            'Scenario Name': scenario['scenario_name']
        })
    
    if not rows:
        return None
    
    df = pd.DataFrame(rows)
    filepath = os.path.join(output_dir, 'unmatched_scenarios.csv')
    df.to_csv(filepath, index=False)
    
    return filepath


def export_table4_excel_with_jira(
    table4: pd.DataFrame,
    output_dir: str,
    env_path: Optional[str] = None,
    api_version: str = DEFAULT_API_VERSION,
) -> str:
    """
    Write Table 4 to Sheet 1 and Jira probability data to Sheet 2 of an Excel file.

    Reads unique base TLR keys from *table4*, fetches
    ``boja_prop_issue.risk_probability_value`` and
    ``boja_prop_issue.risk_residual_probability_value`` from the Jira REST API,
    then writes a two-sheet workbook:

    * **Sheet 1 – "Hazardous Situations"**: identical to the Table 4 CSV output.
    * **Sheet 2 – "Jira Probabilities"**: one row per unique base TLR key with
      Initial Probability and Residual Probability columns.

    Requires ``openpyxl`` (``pip install openpyxl``) and Jira credentials in
    environment variables (see ``jira_risk_probabilities.fetch_probabilities``).

    Args:
        table4: DataFrame produced by ``generate_table4()``.
        output_dir: Directory in which to write the ``.xlsx`` file.
        env_path: Optional path to a ``.env`` credentials file.

    Returns:
        Path to the written Excel file.
    """
    raw_keys = table4["TLR Key"].dropna().tolist()
    df_probs = fetch_probabilities(raw_keys, env_path=env_path, api_version=api_version)

    xlsx_path = os.path.join(output_dir, "table4_with_jira_probabilities.xlsx")
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        table4.to_excel(writer, sheet_name="Hazardous Situations", index=False)
        df_probs.to_excel(writer, sheet_name="Jira Probabilities", index=False)

    return xlsx_path


def export_results_to_csv(q1_results: Dict, q2_results: Dict, q3_results: pd.DataFrame,
                          acceptance: Dict, output_dir: str,
                          validation_report: Optional[Dict] = None) -> List[str]:
    """
    Export all results to CSV files.
    
    Args:
        q1_results: Question 1 results
        q2_results: Question 2 results
        q3_results: Question 3 results DataFrame
        acceptance: Acceptance criteria results
        output_dir: Output directory
        validation_report: Optional validation report for unmatched scenarios
        
    Returns:
        List of exported file paths
    """
    os.makedirs(output_dir, exist_ok=True)
    
    exported_files = []
    
    # Table 2 - Question 1
    table2 = generate_table2(q1_results)
    table2_path = os.path.join(output_dir, 'table2_overall_risk_profile.csv')
    table2.to_csv(table2_path, index=False)
    exported_files.append(table2_path)
    
    # Table 3 - Question 2
    table3 = generate_table3(q2_results)
    table3_path = os.path.join(output_dir, 'table3_patient_profile_stratification.csv')
    table3.to_csv(table3_path, index=False)
    exported_files.append(table3_path)
    
    # Table 4 - Question 3
    table4 = generate_table4(q3_results)
    table4_path = os.path.join(output_dir, 'table4_individual_hazardous_situations.csv')
    table4.to_csv(table4_path, index=False)
    exported_files.append(table4_path)
    
    # Unmatched scenarios (if any)
    if validation_report:
        unmatched_path = export_unmatched_scenarios_to_csv(validation_report, output_dir)
        if unmatched_path:
            exported_files.append(unmatched_path)
    
    return exported_files


def generate_summary_report(validation_report: Dict, q1_results: Dict, q2_results: Dict,
                            q3_results: pd.DataFrame, acceptance: Dict,
                            output_dir: str) -> str:
    """
    Generate comprehensive text summary report.
    
    Args:
        validation_report: Data validation results
        q1_results: Question 1 results
        q2_results: Question 2 results
        q3_results: Question 3 results
        acceptance: Acceptance criteria results
        output_dir: Output directory
        
    Returns:
        Path to generated report file
    """
    lines = []
    
    lines.append("=" * 70)
    lines.append("ALGORITHM RISK PROFILE COMPARISON ANALYSIS REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 70)
    
    # Data Validation
    lines.append("\n\n1. DATA VALIDATION")
    lines.append("-" * 40)
    lines.append(f"Total scenarios in V1: {validation_report['total_v1']}")
    lines.append(f"Total scenarios in V2: {validation_report['total_v2']}")
    lines.append(f"Matched pairs: {validation_report['matched']}")
    lines.append(f"Unmatched in V1 only: {len(validation_report['only_in_v1'])}")
    lines.append(f"Unmatched in V2 only: {len(validation_report['only_in_v2'])}")
    lines.append(f"Validation status: {'PASS' if validation_report['is_valid'] else 'ISSUES DETECTED'}")
    
    # List unmatched scenarios with details
    if validation_report.get('unmatched_v1_details'):
        lines.append("\nScenarios in V1 (previously cleared) without V2 match:")
        for scenario in validation_report['unmatched_v1_details']:
            lines.append(f"  - {scenario['risk_name']} | {scenario['scenario_type']} | "
                        f"{scenario['patient_profile']} | {scenario['scenario_name']}")
    
    if validation_report.get('unmatched_v2_details'):
        lines.append("\nScenarios in V2 (version 2.0) without V1 match:")
        for scenario in validation_report['unmatched_v2_details']:
            lines.append(f"  - {scenario['risk_name']} | {scenario['scenario_type']} | "
                        f"{scenario['patient_profile']} | {scenario['scenario_name']}")
    
    # Question 1 Summary
    lines.append("\n\n2. QUESTION 1: OVERALL RISK PROFILE")
    lines.append("-" * 40)
    
    for metric_col, results in q1_results.items():
        lines.append(f"\n{results['metric_name']}:")
        lines.append(f"  N pairs: {results['n_pairs']}")
        lines.append(f"  Mean V1: {results['mean']['v1_mean']:.4f}")
        lines.append(f"  Mean V2: {results['mean']['v2_mean']:.4f}")
        lines.append(f"  Mean difference: {results['mean']['mean_diff']:.4f}")
        
        # Wilcoxon with context
        p_val = results['wilcoxon']['p_value']
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        pct_changed = results['wilcoxon'].get('pct_changed', 0)
        n_non_zero = results['wilcoxon'].get('n_non_zero', 'N/A')
        lines.append(f"  Wilcoxon p-value: {p_str}")
        lines.append(f"    ({pct_changed:.1f}% of pairs changed, n={n_non_zero} non-zero differences)")
        
        # Effect sizes - Two-part approach for zero-inflated data
        lines.append(f"  Effect sizes:")
        
        # Rank-biserial (uses same non-zero subset as Wilcoxon)
        r_val = results['wilcoxon'].get('rank_biserial_r', np.nan)
        r_interp = results['wilcoxon'].get('r_interpretation', 'N/A')
        r_str = f"{r_val:.4f}" if not np.isnan(r_val) else "N/A"
        lines.append(f"    Rank-biserial r: {r_str} ({r_interp}) [non-zero pairs only]")
        
        # Cohen's d - all pairs
        d_all = results['cohens_d']['cohens_d']
        d_all_interp = results['cohens_d']['interpretation']
        lines.append(f"    Cohen's d (all): {d_all:.4f} ({d_all_interp})")
        
        # Cohen's d - non-zero only
        d_nz = results['cohens_d'].get('d_nonzero', np.nan)
        d_nz_interp = results['cohens_d'].get('interpretation_nonzero', 'N/A')
        d_nz_str = f"{d_nz:.4f}" if not np.isnan(d_nz) else "N/A"
        lines.append(f"    Cohen's d (non-zero): {d_nz_str} ({d_nz_interp})")
        
    # Question 2 Summary
    lines.append("\n\n3. QUESTION 2: PATIENT PROFILE STRATIFICATION")
    lines.append("-" * 40)
    lines.append("\nNote: Median is more robust for skewed distributions (e.g., LBGI).")
    lines.append("Wilcoxon tests ranks (aligned with median), not means.")
    
    for profile in PATIENT_PROFILES:
        if profile not in q2_results or 'error' in q2_results.get(profile, {}):
            continue
        
        lines.append(f"\n{profile.capitalize()} (n={q2_results[profile]['n_pairs']}):")
        
        for metric_col, metric_name in METRICS.items():
            if metric_col not in q2_results[profile]:
                continue
            
            results = q2_results[profile][metric_col]
            p_val = results['wilcoxon']['p_value']
            p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
            
            mean_diff = results['mean']['mean_diff']
            median_diff = results['median']['median_diff']
            
            lines.append(f"  {metric_name}:")
            lines.append(f"    Mean diff: {mean_diff:.4f}, Median diff: {median_diff:.4f}")
            lines.append(f"    Wilcoxon p={p_str}")
    
    # Question 3 Summary
    lines.append("\n\n4. QUESTION 3: AGGREGATED SEVERITY BY RISK AND SCENARIO TYPE")
    lines.append("-" * 40)
    lines.append(f"Total risk/scenario combinations analyzed: {len(q3_results)}")
    
    if len(q3_results) > 0:
        # Count significant changes
        sig_changes = q3_results[q3_results['has_significant_change'] == True]
        lines.append(f"Combinations with significant changes: {len(sig_changes)}")
        
        if len(sig_changes) > 0:
            lines.append("\nSignificant changes detected:")
            for _, row in sig_changes.iterrows():
                lines.append(f"  - {row['risk_name']} | {row['scenario_type']}: "
                            f"Severity {row['severity_v1']} -> {row['severity_v2']} "
                            f"({row['change_types']})")
        
        # Summary statistics
        lines.append("\nSeverity change summary (V2 - V1):")
        lines.append(f"  Mean change: {q3_results['severity_change'].mean():.2f}")
        lines.append(f"  Increases: {(q3_results['severity_change'] > 0).sum()}")
        lines.append(f"  Decreases: {(q3_results['severity_change'] < 0).sum()}")
        lines.append(f"  No change: {(q3_results['severity_change'] == 0).sum()}")
    
    # Acceptance Criteria
    lines.append("\n\n5. ACCEPTANCE CRITERIA EVALUATION")
    lines.append("-" * 40)
    lines.append(generate_acceptance_summary(acceptance))
    
    # Write report
    report_path = os.path.join(output_dir, 'analysis_report.txt')
    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    
    return report_path


# =============================================================================
# Module 9: Hazardous Situation Comparison Report
# =============================================================================

# Openpyxl-compatible hex fill colours for acceptability categories
_ACCEPT_FILL_COLORS = {
    'Acceptable': 'C6EFCE',
    'Conditionally acceptable': 'FFEB9C',
    'Unacceptable': 'FFC7CE',
    'Unknown': 'FFFFFF',
}

# Ordering rank: higher = more risk (used to detect direction of change)
_ACCEPT_RANK = {
    'Acceptable': 0,
    'Conditionally acceptable': 1,
    'Unacceptable': 2,
    'Unknown': -1,
}

# Column definitions per sheet (order matters for cell-colour indexing)
_SHEET_COLUMNS: Dict[str, List[str]] = {
    'Changes to Harm': [
        'TLR Key', 'Summary', 'Hazard Category', 'Evaluation Stage',
        'V1 Harm', 'V2 Harm', 'V1 Risk Score', 'V2 Risk Score',
    ],
    'Severity Increased': [
        'TLR Key', 'Summary', 'Hazard Category', 'Evaluation Stage',
        'Harm', 'V1 Risk Score', 'V2 Risk Score',
    ],
    'Severity Decreased': [
        'TLR Key', 'Summary', 'Hazard Category', 'Evaluation Stage',
        'Harm', 'V1 Risk Score', 'V2 Risk Score',
    ],
    'Risk Score Increased': [
        'TLR Key', 'Summary', 'Hazard Category', 'Evaluation Stage',
        'Harm', 'V1 Risk Score', 'V2 Risk Score',
    ],
    'Risk Score Decreased': [
        'TLR Key', 'Summary', 'Hazard Category', 'Evaluation Stage',
        'Harm', 'V1 Risk Score', 'V2 Risk Score',
    ],
}


def parse_probability(value: Any) -> Optional[float]:
    """
    Convert a Jira probability field value to a float.

    Jira stores Initial and Residual probability values as plain integer
    strings (e.g. '1', '3', '5').  Returns None if the value is absent or
    cannot be parsed, allowing downstream risk-score calculations to propagate
    None cleanly.

    Args:
        value: Raw probability value from the Jira property (str, int, or None).

    Returns:
        Float probability value, or None on failure.
    """
    if value is None:
        return None
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def compute_risk_score(
    severity: int,
    initial_prob: Optional[float],
    residual_prob: Optional[float],
    scenario_type: str,
) -> Optional[float]:
    """
    Compute risk score (Severity x Probability) for a given evaluation stage.

    Probability source depends on scenario type:
      - pre-mitigation  -> Initial Probability
      - noLoop          -> Initial Probability
      - post-mitigation -> Residual Probability

    Args:
        severity: Integer severity score (0-5).
        initial_prob: Parsed initial probability (float or None).
        residual_prob: Parsed residual probability (float or None).
        scenario_type: One of 'pre-mitigation', 'noLoop', 'post-mitigation'.

    Returns:
        Float risk score, or None if any required input is None.
    """
    prob = residual_prob if scenario_type == 'post-mitigation' else initial_prob
    if severity is None or prob is None:
        return None
    return float(severity) * prob


def get_acceptability(
    risk_score: Optional[float], severity: int
) -> Tuple[str, str]:
    """
    Return (acceptability label, hex fill colour) for a risk score.

    Acceptability rules (requirement section 2):
      - 1-3             -> Acceptable (green)
      - 4, sev 1-3      -> Acceptable (green)
      - 4, sev >= 4     -> Conditionally acceptable (yellow)
      - 5-9             -> Conditionally acceptable (yellow)
      - >= 10           -> Unacceptable (red)
      - None            -> Unknown (white; no probability data available)

    Args:
        risk_score: Computed risk score, or None.
        severity: Integer severity used for the risk = 4 boundary case.

    Returns:
        Tuple of (label_string, hex_colour_string).
    """
    if risk_score is None:
        return ('Unknown', _ACCEPT_FILL_COLORS['Unknown'])

    if risk_score <= 3:
        label = 'Acceptable'
    elif risk_score == 4:
        label = 'Acceptable' if severity <= 3 else 'Conditionally acceptable'
    elif risk_score <= 9:
        label = 'Conditionally acceptable'
    else:
        label = 'Unacceptable'

    return (label, _ACCEPT_FILL_COLORS[label])


def acceptability_rank(label: str) -> int:
    """
    Return the ordinal rank for an acceptability label.

    Higher rank = greater risk: Acceptable(0) < Conditionally acceptable(1) < Unacceptable(2).
    Unknown returns -1 so it never triggers directional comparisons.

    Args:
        label: Acceptability label string.

    Returns:
        Integer rank (0, 1, 2, or -1 for unknown).
    """
    return _ACCEPT_RANK.get(label, -1)


def _format_risk_score_cell(risk_score: Optional[float], label: str) -> str:
    """
    Format a risk-score cell value as '{score}, {acceptability}'.

    Integer-valued scores are displayed without a decimal point.
    Returns 'N/A' when risk_score is None.

    Args:
        risk_score: Numeric risk score or None.
        label: Acceptability label string.

    Returns:
        Formatted string, e.g. '6, Conditionally acceptable'.
    """
    if risk_score is None:
        return 'N/A'
    score_str = (
        str(int(risk_score))
        if risk_score == int(risk_score)
        else f'{risk_score:.2f}'
    )
    return f'{score_str}, {label}'


def enrich_with_jira_data(q3_df: pd.DataFrame, jira_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join Question 3 results with Jira metadata and compute risk scores.

    Left-joins on the normalised TLR base key so that scenario-variant suffixes
    (e.g. '_01_025') are stripped before matching.  Adds columns:

      Summary, Hazard Category, No Automation,
      Initial Probability, Residual Probability,
      risk_score_v1, risk_score_v2,
      accept_label_v1, accept_color_v1,
      accept_label_v2, accept_color_v2

    Args:
        q3_df: DataFrame produced by run_question3_analysis().
        jira_df: DataFrame produced by fetch_probabilities().

    Returns:
        Enriched copy of q3_df with added columns.
    """
    enriched = q3_df.copy()
    enriched['_base_key'] = enriched['risk_name'].apply(
        lambda k: normalize_tlr_key(str(k)) or str(k)
    )

    jira_keyed = jira_df.rename(columns={'TLR Key': '_base_key'})
    keep_cols = [
        '_base_key', 'Summary', 'Hazard Category', 'No Automation',
        'Initial Probability', 'Residual Probability',
    ]
    enriched = enriched.merge(jira_keyed[keep_cols], on='_base_key', how='left')

    enriched['_init_prob'] = enriched['Initial Probability'].apply(parse_probability)
    enriched['_resid_prob'] = enriched['Residual Probability'].apply(parse_probability)

    enriched['risk_score_v1'] = enriched.apply(
        lambda r: compute_risk_score(
            r['severity_v1'], r['_init_prob'], r['_resid_prob'], r['scenario_type']
        ),
        axis=1,
    )
    enriched['risk_score_v2'] = enriched.apply(
        lambda r: compute_risk_score(
            r['severity_v2'], r['_init_prob'], r['_resid_prob'], r['scenario_type']
        ),
        axis=1,
    )

    accept_v1 = enriched.apply(
        lambda r: get_acceptability(r['risk_score_v1'], int(r['severity_v1'])), axis=1
    )
    accept_v2 = enriched.apply(
        lambda r: get_acceptability(r['risk_score_v2'], int(r['severity_v2'])), axis=1
    )
    enriched['accept_label_v1'] = accept_v1.apply(lambda x: x[0])
    enriched['accept_color_v1'] = accept_v1.apply(lambda x: x[1])
    enriched['accept_label_v2'] = accept_v2.apply(lambda x: x[0])
    enriched['accept_color_v2'] = accept_v2.apply(lambda x: x[1])

    enriched.drop(columns=['_init_prob', '_resid_prob', '_base_key'], inplace=True, errors='ignore')
    return enriched


def classify_hazard_rows(enriched_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flag columns indicating which comparison sheets each row qualifies for.

    Flag logic (per requirements section 1):

      flag_harm_change   – 'harm_type_change' present in change_types
                           (criterion 5 – no noLoop/no_automation filter)
      flag_sev_increase  – 'severity_threshold_crossing' in change_types AND
                           severity_change > 0
      flag_sev_decrease  – 'severity_threshold_crossing' in change_types AND
                           severity_change < 0
      flag_risk_increase – acceptability rank rose (V1 -> V2);
                           noLoop rows only count when no_automation = True
      flag_risk_decrease – acceptability rank fell (V1 -> V2);
                           noLoop rows only count when no_automation = True

    Also sets 'in_scope' = True when any flag is True.

    Args:
        enriched_df: DataFrame from enrich_with_jira_data().

    Returns:
        Copy of enriched_df with five flag columns and 'in_scope' added.
    """
    df = enriched_df.copy()

    def _noloop_eligible(row: pd.Series) -> bool:
        if row['scenario_type'] == 'noLoop':
            return bool(row.get('No Automation', False))
        return True

    df['flag_harm_change'] = df['change_types'].apply(
        lambda ct: 'harm_type_change' in ct
    )

    df['flag_sev_increase'] = df.apply(
        lambda r: (
            'severity_threshold_crossing' in r['change_types']
            and r['severity_change'] > 0
        ),
        axis=1,
    )
    df['flag_sev_decrease'] = df.apply(
        lambda r: (
            'severity_threshold_crossing' in r['change_types']
            and r['severity_change'] < 0
        ),
        axis=1,
    )

    rank_v1 = df['accept_label_v1'].apply(acceptability_rank)
    rank_v2 = df['accept_label_v2'].apply(acceptability_rank)
    eligible = df.apply(_noloop_eligible, axis=1)

    df['flag_risk_increase'] = (
        (rank_v2 > rank_v1) & (rank_v1 >= 0) & (rank_v2 >= 0)
    ) & eligible
    df['flag_risk_decrease'] = (
        (rank_v2 < rank_v1) & (rank_v1 >= 0) & (rank_v2 >= 0)
    ) & eligible

    df['in_scope'] = (
        df['flag_harm_change']
        | df['flag_sev_increase']
        | df['flag_sev_decrease']
        | df['flag_risk_increase']
        | df['flag_risk_decrease']
    )

    return df


def assign_hazard_rows_to_sheets(
    classified_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """
    Apply deduplication rules and assign each qualifying row to exactly one sheet.

    Priority order (first matching rule wins):
      1. Changes to Harm      (flag_harm_change)
      2. Risk Score Increased (flag_risk_increase; overrides Severity Increased)
      3. Risk Score Decreased (flag_risk_decrease; overrides Severity Decreased)
      4. Severity Increased   (flag_sev_increase; only if not already assigned)
      5. Severity Decreased   (flag_sev_decrease; only if not already assigned)

    All five sheet keys are always present in the returned dict; unqualified
    sheets contain an empty DataFrame.

    Args:
        classified_df: DataFrame from classify_hazard_rows().

    Returns:
        Dict mapping sheet name -> DataFrame of qualifying rows.
    """
    df = classified_df.copy()
    assigned = pd.Series('', index=df.index, dtype=str)

    assigned[df['flag_harm_change']] = 'Changes to Harm'
    assigned[(assigned == '') & df['flag_risk_increase']] = 'Risk Score Increased'
    assigned[(assigned == '') & df['flag_risk_decrease']] = 'Risk Score Decreased'
    assigned[(assigned == '') & df['flag_sev_increase']] = 'Severity Increased'
    assigned[(assigned == '') & df['flag_sev_decrease']] = 'Severity Decreased'

    df['_sheet'] = assigned

    sheet_names = [
        'Changes to Harm',
        'Severity Increased',
        'Severity Decreased',
        'Risk Score Increased',
        'Risk Score Decreased',
    ]
    return {
        name: df[df['_sheet'] == name].drop(columns=['_sheet']).reset_index(drop=True)
        for name in sheet_names
    }


def compute_hazard_summary_stats(
    q3_df: pd.DataFrame, sheet_data: Dict[str, pd.DataFrame]
) -> Dict[str, Any]:
    """
    Compute summary statistics for the Summary sheet.

    Aggregates across ALL rows in q3_df (not just in-scope rows):
      - Overall severity score change: sum(severity_v2) - sum(severity_v1)
      - Total scenarios with severity decreased
      - Total scenarios with severity increased
      - Per-sheet practical-difference counts (including 0)

    Args:
        q3_df: Full Question 3 results DataFrame (before any filtering).
        sheet_data: Dict from assign_hazard_rows_to_sheets().

    Returns:
        Dictionary of computed statistics.
    """
    severity_change_sum = int(q3_df['severity_change'].sum())
    n_decreased = int((q3_df['severity_change'] < 0).sum())
    n_increased = int((q3_df['severity_change'] > 0).sum())
    practical_counts = {name: len(df) for name, df in sheet_data.items()}

    return {
        'severity_change_sum': severity_change_sum,
        'n_sev_decreased': n_decreased,
        'n_sev_increased': n_increased,
        'practical_counts': practical_counts,
    }


def _build_sheet_rows(
    df: pd.DataFrame, sheet_name: str
) -> List[Tuple[List[Any], str, str]]:
    """
    Convert an enriched DataFrame to (row_values, v1_colour, v2_colour) tuples.

    Used internally by write_hazardous_situation_comparison_excel().
    The 'Harm' column on sheets other than 'Changes to Harm' uses harm_v2
    (the current version's assessment).

    Args:
        df: Enriched, classified DataFrame for a single sheet.
        sheet_name: Name of the target sheet (determines column layout).

    Returns:
        List of (row_values, accept_color_v1, accept_color_v2) tuples.
    """
    rows = []
    for _, r in df.iterrows():
        v1_rs = _format_risk_score_cell(r['risk_score_v1'], r['accept_label_v1'])
        v2_rs = _format_risk_score_cell(r['risk_score_v2'], r['accept_label_v2'])

        if sheet_name == 'Changes to Harm':
            row_vals: List[Any] = [
                r['risk_name'],
                r.get('Summary') or '',
                r.get('Hazard Category') or '',
                r['scenario_type'],
                r['harm_v1'],
                r['harm_v2'],
                v1_rs,
                v2_rs,
            ]
        else:
            row_vals = [
                r['risk_name'],
                r.get('Summary') or '',
                r.get('Hazard Category') or '',
                r['scenario_type'],
                r['harm_v2'],
                v1_rs,
                v2_rs,
            ]

        rows.append((row_vals, str(r['accept_color_v1']), str(r['accept_color_v2'])))
    return rows


def write_hazardous_situation_comparison_excel(
    sheet_data: Dict[str, pd.DataFrame],
    summary_stats: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Write the six-sheet hazardous situation comparison workbook.

    Sheet 1 (Summary): overall severity delta, severity-change counts, and
    per-sheet practical-difference counts.

    Sheets 2-6: one per comparison category.  Risk-score cells are
    colour-coded (green / yellow / red) to reflect acceptability, and
    formatted as '{score}, {label}'.  If no rows qualify for a sheet,
    cell A2 is populated with 'No hazardous situations meeting criteria'.

    Requires openpyxl (pip install openpyxl).

    Args:
        sheet_data: Dict from assign_hazard_rows_to_sheets().
        summary_stats: Dict from compute_hazard_summary_stats().
        output_path: Destination .xlsx file path.

    Returns:
        output_path (unchanged; returned for chaining / logging).
    """
    from openpyxl import Workbook
    from openpyxl.styles import PatternFill, Font
    from openpyxl.utils import get_column_letter

    wb = Workbook()
    bold = Font(bold=True)

    # ── Sheet 1: Summary ─────────────────────────────────────────────────────
    ws = wb.active
    ws.title = 'Summary'
    ws.append(['Metric', 'Value'])
    for cell in ws[1]:
        cell.font = bold

    summary_rows: List[Tuple[str, Any]] = [
        ('Overall Severity Score Change (V2 - V1, all scenarios)',
         summary_stats['severity_change_sum']),
        ('Total Scenarios with Severity Decreased (all)',
         summary_stats['n_sev_decreased']),
        ('Total Scenarios with Severity Increased (all)',
         summary_stats['n_sev_increased']),
        ('', ''),
    ]
    for name, count in summary_stats['practical_counts'].items():
        summary_rows.append((f'Total {name}', count))

    for label, val in summary_rows:
        ws.append([label, val])

    ws.column_dimensions['A'].width = 58
    ws.column_dimensions['B'].width = 12

    # ── Sheets 2-6 ───────────────────────────────────────────────────────────
    _COL_WIDTHS: Dict[str, int] = {
        'TLR Key': 16,
        'Summary': 50,
        'Hazard Category': 22,
        'Evaluation Stage': 18,
        'V1 Harm': 16, 'V2 Harm': 16, 'Harm': 16,
        'V1 Risk Score': 28, 'V2 Risk Score': 28,
    }

    sheet_order = [
        'Changes to Harm',
        'Severity Increased',
        'Severity Decreased',
        'Risk Score Increased',
        'Risk Score Decreased',
    ]

    for sheet_name in sheet_order:
        ws = wb.create_sheet(title=sheet_name)
        cols = _SHEET_COLUMNS[sheet_name]
        ws.append(cols)
        for cell in ws[1]:
            cell.font = bold

        df = sheet_data.get(sheet_name, pd.DataFrame())

        if len(df) == 0:
            ws['A2'] = 'No hazardous situations meeting criteria'
            for col_idx, col_name in enumerate(cols, start=1):
                ws.column_dimensions[get_column_letter(col_idx)].width = (
                    _COL_WIDTHS.get(col_name, 18)
                )
            continue

        v1_col_idx = cols.index('V1 Risk Score') + 1
        v2_col_idx = cols.index('V2 Risk Score') + 1

        for row_vals, color_v1, color_v2 in _build_sheet_rows(df, sheet_name):
            ws.append(row_vals)
            row_idx = ws.max_row
            ws.cell(row=row_idx, column=v1_col_idx).fill = PatternFill(
                start_color=color_v1, end_color=color_v1, fill_type='solid'
            )
            ws.cell(row=row_idx, column=v2_col_idx).fill = PatternFill(
                start_color=color_v2, end_color=color_v2, fill_type='solid'
            )

        for col_idx, col_name in enumerate(cols, start=1):
            ws.column_dimensions[get_column_letter(col_idx)].width = (
                _COL_WIDTHS.get(col_name, 18)
            )

    wb.save(output_path)
    return output_path


def generate_hazardous_situation_comparison(
    q3_results: pd.DataFrame,
    output_dir: str,
    env_path: Optional[str] = None,
    api_version: str = DEFAULT_API_VERSION,
) -> Optional[str]:
    """
    Generate the hazardous situation comparison Excel workbook.

    End-to-end entry point for Module 9.  Fetches Jira metadata and
    probability values, enriches Q3 results, classifies and deduplicates
    rows, then writes a six-sheet Excel file to output_dir.

    Args:
        q3_results: DataFrame from run_question3_analysis().
        output_dir: Directory in which to write the output file.
        env_path: Optional path to a .env credentials file for Jira.
        api_version: Jira REST API version string.

    Returns:
        Path to the written .xlsx file, or None if q3_results is empty.

    Raises:
        EnvironmentError: Propagated from fetch_probabilities() if Jira
            credentials are missing.
        ConnectionError: Propagated if Jira authentication fails.
    """
    if len(q3_results) == 0:
        print('   No Q3 results; skipping hazardous situation comparison.')
        return None

    raw_keys = q3_results['risk_name'].dropna().tolist()
    jira_df = fetch_probabilities(raw_keys, env_path=env_path, api_version=api_version)

    enriched = enrich_with_jira_data(q3_results, jira_df)
    classified = classify_hazard_rows(enriched)
    sheet_data = assign_hazard_rows_to_sheets(classified)
    summary_stats = compute_hazard_summary_stats(q3_results, sheet_data)

    output_path = os.path.join(output_dir, 'hazardous_situation_comparison.xlsx')
    write_hazardous_situation_comparison_excel(sheet_data, summary_stats, output_path)
    print(f'   Hazardous situation comparison saved to: {output_path}')
    return output_path


# =============================================================================
# Module 8: Main Entry Point
# =============================================================================

def main(filepath_v1: str, filepath_v2: str, output_dir: str, jira: bool = False,
         env_path: Optional[str] = None,
         api_version: str = DEFAULT_API_VERSION,
         hazard_comparison: bool = False) -> Dict[str, Any]:
    """
    Main entry point for the analysis.
    
    Args:
        filepath_v1: Path to previously cleared version results CSV
        filepath_v2: Path to version 2.0 results CSV
        output_dir: Directory for output files
        
    Returns:
        Dictionary containing acceptance criteria results
    """
    print("=" * 70)
    print("ALGORITHM RISK PROFILE COMPARISON ANALYSIS")
    print("=" * 70)
    
    # 1. Load and preprocess
    print("\n1. Loading and merging data...")
    df_merged, validation_report = load_and_merge_data(filepath_v1, filepath_v2)
    print(f"   Matched pairs: {validation_report['matched']}")
    
    if not validation_report['is_valid']:
        print(f"   WARNING: {len(validation_report['only_in_v1'])} unmatched in V1, "
              f"{len(validation_report['only_in_v2'])} unmatched in V2")
        
        if validation_report.get('unmatched_v1_details'):
            print("\n   Scenarios in V1 without V2 match:")
            for scenario in validation_report['unmatched_v1_details'][:10]:  # Show first 10
                print(f"     - {scenario['risk_name']} | {scenario['scenario_type']} | "
                      f"{scenario['patient_profile']}")
            if len(validation_report['unmatched_v1_details']) > 10:
                print(f"     ... and {len(validation_report['unmatched_v1_details']) - 10} more")
        
        if validation_report.get('unmatched_v2_details'):
            print("\n   Scenarios in V2 without V1 match:")
            for scenario in validation_report['unmatched_v2_details'][:10]:  # Show first 10
                print(f"     - {scenario['risk_name']} | {scenario['scenario_type']} | "
                      f"{scenario['patient_profile']}")
            if len(validation_report['unmatched_v2_details']) > 10:
                print(f"     ... and {len(validation_report['unmatched_v2_details']) - 10} more")
    
    # 2. Question 1: Overall risk profile
    print("\n2. Running Question 1 analysis (overall risk profile)...")
    q1_results = run_question1_analysis(df_merged)
    
    for metric_col, results in q1_results.items():
        p_val = results['wilcoxon']['p_value']
        p_str = f"{p_val:.4f}" if not np.isnan(p_val) else "N/A"
        print(f"   {results['metric_name']}: mean_diff={results['mean']['mean_diff']:.4f}, p={p_str}")
    
    # 3. Question 2: Patient profile stratification
    print("\n3. Running Question 2 analysis (patient profile stratification)...")
    q2_results = run_question2_analysis(df_merged)
    
    for profile in PATIENT_PROFILES:
        if profile in q2_results and 'n_pairs' in q2_results[profile]:
            print(f"   {profile.capitalize()}: n={q2_results[profile]['n_pairs']}")
    
    # 4. Question 3: Individual hazardous situations
    print("\n4. Running Question 3 analysis (aggregated severity by risk and scenario)...")
    q3_results = run_question3_analysis(df_merged)
    print(f"   Risk/scenario combinations: {len(q3_results)}")
    if len(q3_results) > 0:
        sig_changes = q3_results[q3_results['has_significant_change'] == True]
        print(f"   Combinations with significant changes: {len(sig_changes)}")
    
    # 5. Acceptance criteria
    print("\n5. Evaluating acceptance criteria...")
    acceptance = evaluate_acceptance_criteria(q1_results, q2_results)
    print(f"   Overall determination: {acceptance['overall']['summary']}")
    
    # 6. Generate outputs
    print("\n6. Generating outputs...")
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualizations
    print("   Generating visualizations...")
    figure_paths = generate_all_visualizations(df_merged, output_dir)
    print(f"   Created {len(figure_paths)} figures")
    
    # CSV exports
    print("   Exporting CSV tables...")
    csv_paths = export_results_to_csv(q1_results, q2_results, q3_results, acceptance,
                                       output_dir, validation_report)
    print(f"   Created {len(csv_paths)} CSV files")

    # Optional Jira-enriched Excel output
    if jira:
        print("   Fetching Jira probability values and writing Excel report...")
        try:
            table4 = generate_table4(q3_results)
            xlsx_path = export_table4_excel_with_jira(table4, output_dir, env_path=env_path, api_version=api_version)
            print(f"   Excel report saved to: {xlsx_path}")
        except EnvironmentError as exc:
            print(f"   WARNING: Jira export skipped – {exc}")
        except Exception as exc:
            print(f"   WARNING: Jira export failed – {exc}")

    # Optional hazardous situation comparison
    if hazard_comparison:
        print("   Generating hazardous situation comparison report...")
        try:
            generate_hazardous_situation_comparison(
                q3_results, output_dir, env_path=env_path, api_version=api_version
            )
        except EnvironmentError as exc:
            print(f"   WARNING: Hazardous situation comparison skipped – {exc}")
        except Exception as exc:
            print(f"   WARNING: Hazardous situation comparison failed – {exc}")

    # Summary report
    print("   Generating summary report...")
    report_path = generate_summary_report(validation_report, q1_results, q2_results, 
                                          q3_results, acceptance, output_dir)
    print(f"   Report saved to: {report_path}")
    
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nOutput directory: {output_dir}")
    print(f"Overall result: {acceptance['overall']['summary']}")
    
    return acceptance


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Compare risk profiles between two algorithm versions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    python algorithm_risk_comparison.py \\
        --v1 /path/to/Risk_Results_v1.csv \\
        --v2 /path/to/Risk_Results_v2.csv \\
        --output /path/to/output_directory
        """
    )
    
    parser.add_argument('--v1', required=True,
                        help='Path to previously cleared version results CSV')
    parser.add_argument('--v2', required=True,
                        help='Path to version 2.0 results CSV')
    parser.add_argument('--output', '-o', required=True,
                        help='Output directory for results')
    parser.add_argument('--jira', action='store_true', default=False,
                        help='Fetch Jira probability values and write a two-sheet Excel report '
                             '(requires JIRA_BASE_URL, JIRA_USERNAME, JIRA_API_TOKEN env vars)')
    parser.add_argument('--env', metavar='FILE', default=None,
                        help='Path to .env file containing Jira credentials '
                             '(default: .env in current working directory)')
    parser.add_argument('--jira-api-version', metavar='VERSION', default=DEFAULT_API_VERSION,
                        help=f'Jira REST API version (default: {DEFAULT_API_VERSION})')
    parser.add_argument('--hazard-comparison', action='store_true', default=False,
                        help='Generate hazardous situation comparison Excel report '
                             '(requires Jira credentials: --env / env vars)')

    args = parser.parse_args()
    
    if not os.path.exists(args.v1):
        print(f"Error: V1 file not found: {args.v1}")
        exit(1)
    
    if not os.path.exists(args.v2):
        print(f"Error: V2 file not found: {args.v2}")
        exit(1)
    
    main(args.v1, args.v2, args.output, jira=args.jira, env_path=args.env,
         api_version=args.jira_api_version, hazard_comparison=args.hazard_comparison)
