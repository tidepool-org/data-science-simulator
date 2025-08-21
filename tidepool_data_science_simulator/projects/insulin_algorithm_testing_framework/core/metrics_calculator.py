"""
Functional metrics calculator for insulin algorithm testing.

This module provides functional approaches to calculating comprehensive metrics for 
comparing insulin delivery algorithms, including time in range, hypoglycemia metrics, 
glucose variability, and insulin delivery.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Tuple
from dataclasses import dataclass
from functools import partial

# Import Tidepool metrics
from tidepool_data_science_metrics.glucose.glucose import (
    percent_values_ge_70_le_180, percent_values_lt_70, percent_values_lt_54,
    percent_values_gt_180, percent_values_gt_250, blood_glucose_risk_index,
    lbgi_risk_score
)

logger = logging.getLogger(__name__)


@dataclass
class MetricsResult:
    """Container for calculated metrics."""
    
    # Time in range metrics
    time_in_range_70_180: float
    time_below_70: float
    time_below_54: float
    time_above_180: float
    time_above_250: float
    
    # Glucose statistics
    mean_glucose: float
    median_glucose: float
    std_glucose: float
    cv_glucose: float
    
    # Risk indices
    lbgi: float
    hbgi: float
    bgri: float
    lbgi_risk_score: float
    
    # Insulin delivery
    cumulative_insulin: float
    basal_insulin: float
    bolus_insulin: float
    
    # Additional metrics
    glucose_management_indicator: Optional[float] = None
    time_in_tight_range_70_140: Optional[float] = None
    coefficient_of_variation: Optional[float] = None
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            'time_in_range_70_180': self.time_in_range_70_180,
            'time_below_70': self.time_below_70,
            'time_below_54': self.time_below_54,
            'time_above_180': self.time_above_180,
            'time_above_250': self.time_above_250,
            'mean_glucose': self.mean_glucose,
            'median_glucose': self.median_glucose,
            'std_glucose': self.std_glucose,
            'cv_glucose': self.cv_glucose,
            'lbgi': self.lbgi,
            'hbgi': self.hbgi,
            'bgri': self.bgri,
            'lbgi_risk_score': self.lbgi_risk_score,
            'cumulative_insulin': self.cumulative_insulin,
            'basal_insulin': self.basal_insulin,
            'bolus_insulin': self.bolus_insulin,
            'glucose_management_indicator': self.glucose_management_indicator,
            'time_in_tight_range_70_140': self.time_in_tight_range_70_140,
            'coefficient_of_variation': self.coefficient_of_variation
        }


# Core data extraction functions
def extract_data_slice(
    results_df: pd.DataFrame,
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Extract glucose and insulin data for the specified time slice.
    
    Args:
        results_df: Simulation results DataFrame
        start_hours: Start time in hours from simulation start
        duration_hours: Duration to analyze (None for all remaining)
        
    Returns:
        Tuple of (glucose_data, insulin_data)
    """
    # Calculate indices (5-minute intervals)
    start_idx = int(start_hours * 12)
    end_idx = start_idx + int(duration_hours * 12) if duration_hours else len(results_df)
    
    # Extract active data only
    active_data = results_df[results_df['active'] == 1]
    
    if len(active_data) == 0:
        return np.array([]), {'basal': 0.0, 'bolus': 0.0, 'total': 0.0}
    
    # Slice the data
    sliced_data = active_data.iloc[start_idx:end_idx]
    
    # Extract glucose data
    glucose_data = sliced_data['bg'].values
    
    # Extract insulin data
    basal_insulin = sliced_data['delivered_basal_insulin'].sum()
    bolus_insulin = sliced_data['true_bolus'].sum()
    
    insulin_data = {
        'basal': basal_insulin,
        'bolus': bolus_insulin,
        'total': basal_insulin + bolus_insulin
    }
    
    return glucose_data, insulin_data


def clip_glucose_values(glucose_data: np.ndarray) -> np.ndarray:
    """Clip glucose values to valid range for risk calculations."""
    return np.clip(glucose_data, 1, 401)


# Glucose metrics functions
def calculate_time_in_ranges(glucose_data: np.ndarray) -> Dict[str, float]:
    """Calculate time in range metrics."""
    if len(glucose_data) == 0:
        return {
            'time_in_range_70_180': 0.0,
            'time_below_70': 0.0,
            'time_below_54': 0.0,
            'time_above_180': 0.0,
            'time_above_250': 0.0,
            'time_in_tight_range_70_140': 0.0
        }
    
    glucose_clipped = clip_glucose_values(glucose_data)
    
    return {
        'time_in_range_70_180': percent_values_ge_70_le_180(glucose_clipped),
        'time_below_70': percent_values_lt_70(glucose_clipped),
        'time_below_54': percent_values_lt_54(glucose_clipped),
        'time_above_180': percent_values_gt_180(glucose_clipped),
        'time_above_250': percent_values_gt_250(glucose_clipped),
        'time_in_tight_range_70_140': np.mean((glucose_data >= 70) & (glucose_data <= 140)) * 100
    }


def calculate_glucose_statistics(glucose_data: np.ndarray) -> Dict[str, float]:
    """Calculate basic glucose statistics."""
    if len(glucose_data) == 0:
        return {
            'mean_glucose': 0.0,
            'median_glucose': 0.0,
            'std_glucose': 0.0,
            'cv_glucose': 0.0,
            'coefficient_of_variation': 0.0
        }
    
    mean_glucose = np.mean(glucose_data)
    std_glucose = np.std(glucose_data)
    cv_glucose = (std_glucose / mean_glucose) * 100 if mean_glucose > 0 else 0.0
    
    return {
        'mean_glucose': mean_glucose,
        'median_glucose': np.median(glucose_data),
        'std_glucose': std_glucose,
        'cv_glucose': cv_glucose,
        'coefficient_of_variation': cv_glucose
    }


def calculate_risk_indices(glucose_data: np.ndarray) -> Dict[str, float]:
    """Calculate glucose risk indices."""
    if len(glucose_data) == 0:
        return {
            'lbgi': 0.0,
            'hbgi': 0.0,
            'bgri': 0.0,
            'lbgi_risk_score': 0.0
        }
    
    glucose_clipped = clip_glucose_values(glucose_data)
    lbgi, hbgi, bgri = blood_glucose_risk_index(glucose_clipped)
    
    return {
        'lbgi': lbgi,
        'hbgi': hbgi,
        'bgri': bgri,
        'lbgi_risk_score': lbgi_risk_score(lbgi)
    }


def calculate_glucose_management_indicator(glucose_data: np.ndarray) -> float:
    """
    Calculate Glucose Management Indicator (GMI).
    GMI = 3.31 + 0.02392 * mean_glucose
    """
    if len(glucose_data) == 0:
        return 0.0
    return 3.31 + 0.02392 * np.mean(glucose_data)


def calculate_glucose_metrics(glucose_data: np.ndarray) -> Dict[str, float]:
    """Calculate all glucose-related metrics."""
    time_ranges = calculate_time_in_ranges(glucose_data)
    statistics = calculate_glucose_statistics(glucose_data)
    risk_indices = calculate_risk_indices(glucose_data)
    
    return {**time_ranges, **statistics, **risk_indices}


# Insulin metrics functions
def calculate_insulin_metrics(insulin_data: Dict[str, float]) -> Dict[str, float]:
    """Calculate insulin delivery metrics."""
    return {
        'cumulative_insulin': insulin_data['total'],
        'basal_insulin': insulin_data['basal'],
        'bolus_insulin': insulin_data['bolus']
    }


# Main calculation functions
def create_empty_metrics() -> MetricsResult:
    """Create empty metrics result for error cases."""
    return MetricsResult(
        time_in_range_70_180=0.0,
        time_below_70=0.0,
        time_below_54=0.0,
        time_above_180=0.0,
        time_above_250=0.0,
        mean_glucose=0.0,
        median_glucose=0.0,
        std_glucose=0.0,
        cv_glucose=0.0,
        lbgi=0.0,
        hbgi=0.0,
        bgri=0.0,
        lbgi_risk_score=0.0,
        cumulative_insulin=0.0,
        basal_insulin=0.0,
        bolus_insulin=0.0,
        glucose_management_indicator=0.0,
        time_in_tight_range_70_140=0.0,
        coefficient_of_variation=0.0
    )


def calculate_all_metrics(
    results_df: pd.DataFrame,
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> MetricsResult:
    """
    Calculate all metrics for a simulation result.
    
    Args:
        results_df: Simulation results DataFrame
        start_hours: Start time in hours from simulation start
        duration_hours: Duration to analyze (None for all remaining)
        
    Returns:
        MetricsResult object with all calculated metrics
    """
    # Extract data
    glucose_data, insulin_data = extract_data_slice(results_df, start_hours, duration_hours)
    
    if len(glucose_data) == 0:
        logger.warning("No data available for metrics calculation")
        return create_empty_metrics()
    
    # Calculate all metric groups
    glucose_metrics = calculate_glucose_metrics(glucose_data)
    insulin_metrics = calculate_insulin_metrics(insulin_data)
    
    # Add GMI calculation
    glucose_metrics['glucose_management_indicator'] = calculate_glucose_management_indicator(glucose_data)
    
    # Combine all metrics
    return MetricsResult(**glucose_metrics, **insulin_metrics)


# Batch processing functions
def calculate_metrics_batch(
    results_dict: Dict[str, pd.DataFrame],
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> Dict[str, MetricsResult]:
    """
    Calculate metrics for multiple simulation results.
    
    Args:
        results_dict: Dictionary of simulation_id -> results_df
        start_hours: Start time in hours from simulation start
        duration_hours: Duration to analyze (None for all remaining)
        
    Returns:
        Dictionary of simulation_id -> MetricsResult
    """
    metrics_dict = {}
    
    for sim_id, results_df in results_dict.items():
        try:
            metrics = calculate_all_metrics(results_df, start_hours, duration_hours)
            metrics_dict[sim_id] = metrics
        except Exception as e:
            logger.error(f"Error calculating metrics for {sim_id}: {e}")
            metrics_dict[sim_id] = create_empty_metrics()
    
    return metrics_dict


def calculate_time_series_metrics(
    results_df: pd.DataFrame,
    time_windows: List[int] = [1, 2, 4, 6, 8]
) -> Dict[int, MetricsResult]:
    """
    Calculate metrics for different time windows.
    
    Args:
        results_df: Simulation results DataFrame
        time_windows: List of time windows in hours
        
    Returns:
        Dictionary of time_window -> MetricsResult
    """
    return {
        hours: calculate_all_metrics(results_df, start_hours=0, duration_hours=hours)
        for hours in time_windows
    }


def calculate_paired_metrics(
    reference_results: pd.DataFrame,
    comparison_results: pd.DataFrame,
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> Tuple[MetricsResult, MetricsResult, Dict[str, float]]:
    """
    Calculate metrics for paired simulations and their differences.
    
    Args:
        reference_results: Reference algorithm results
        comparison_results: Comparison algorithm results
        start_hours: Start time in hours from simulation start
        duration_hours: Duration to analyze (None for all remaining)
        
    Returns:
        Tuple of (reference_metrics, comparison_metrics, differences)
    """
    ref_metrics = calculate_all_metrics(reference_results, start_hours, duration_hours)
    comp_metrics = calculate_all_metrics(comparison_results, start_hours, duration_hours)
    
    ref_dict = ref_metrics.to_dict()
    comp_dict = comp_metrics.to_dict()
    
    differences = {}
    for metric in ref_dict:
        if ref_dict[metric] is not None and comp_dict[metric] is not None:
            differences[f'{metric}_diff'] = comp_dict[metric] - ref_dict[metric]
            
            # Calculate relative difference for non-zero reference values
            if ref_dict[metric] != 0:
                differences[f'{metric}_rel_diff'] = (
                    (comp_dict[metric] - ref_dict[metric]) / ref_dict[metric] * 100
                )
    
    return ref_metrics, comp_metrics, differences


# Utility functions
def parse_simulation_id(sim_id: str) -> Dict[str, Any]:
    """Parse simulation information from simulation ID."""
    info = {}
    
    try:
        # Expected format: alg=tempbasal_patient=1_ibg=100_meal=20g_paf=0.4_isf=1.0_cir=1.0_basal=1.0
        parts = sim_id.split('_')
        
        for part in parts:
            if '=' in part:
                key, value = part.split('=', 1)
                
                # Convert to appropriate type
                try:
                    if '.' in value:
                        info[key] = float(value)
                    elif value.isdigit():
                        info[key] = int(value)
                    else:
                        # Remove 'g' suffix from meal size
                        if key == 'meal' and value.endswith('g'):
                            info[key] = int(value[:-1])
                        else:
                            info[key] = value
                except ValueError:
                    info[key] = value
    
    except Exception as e:
        logger.warning(f"Could not parse simulation ID {sim_id}: {e}")
    
    return info


def create_metrics_dataframe(
    metrics_dict: Dict[str, MetricsResult],
    include_simulation_info: bool = True
) -> pd.DataFrame:
    """
    Create a DataFrame from metrics results.
    
    Args:
        metrics_dict: Dictionary of simulation_id -> MetricsResult
        include_simulation_info: Whether to parse simulation info from IDs
        
    Returns:
        DataFrame with metrics for each simulation
    """
    data = []
    
    for sim_id, metrics in metrics_dict.items():
        row = {'simulation_id': sim_id}
        row.update(metrics.to_dict())
        
        # Parse simulation info from ID if requested
        if include_simulation_info:
            sim_info = parse_simulation_id(sim_id)
            row.update(sim_info)
        
        data.append(row)
    
    return pd.DataFrame(data)


# Metric extractors
def extract_safety_metrics(metrics: MetricsResult) -> Dict[str, float]:
    """Extract safety-focused metrics."""
    return {
        'time_below_70': metrics.time_below_70,
        'time_below_54': metrics.time_below_54,
        'lbgi': metrics.lbgi,
        'lbgi_risk_score': metrics.lbgi_risk_score
    }


def extract_efficacy_metrics(metrics: MetricsResult) -> Dict[str, float]:
    """Extract efficacy-focused metrics."""
    return {
        'time_in_range_70_180': metrics.time_in_range_70_180,
        'time_above_180': metrics.time_above_180,
        'mean_glucose': metrics.mean_glucose,
        'cv_glucose': metrics.cv_glucose,
        'hbgi': metrics.hbgi
    }


def extract_insulin_metrics(metrics: MetricsResult) -> Dict[str, float]:
    """Extract insulin delivery metrics."""
    return {
        'cumulative_insulin': metrics.cumulative_insulin,
        'basal_insulin': metrics.basal_insulin,
        'bolus_insulin': metrics.bolus_insulin
    }

