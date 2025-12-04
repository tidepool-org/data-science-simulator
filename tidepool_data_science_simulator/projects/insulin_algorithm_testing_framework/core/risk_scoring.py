"""
Functional risk scoring for iCGM sensitivity analysis and regulatory compliance.

This module provides pure functions for calculating LBGI-based risk scores
and event probabilities following FDA AI Letter methodology.
"""

import logging
import warnings
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np
import pandas as pd

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index
from tidepool_data_science_simulator.models.sensor_icgm import DexcomG6ValueModel

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration (immutable dataclasses)
# ============================================================================

@dataclass(frozen=True)
class SeverityBand:
    """Immutable severity band definition for risk scoring."""
    name: str
    min_lbgi: float
    max_lbgi: float
    safety_threshold: float  # events per 100,000 person-years


@dataclass(frozen=True)
class RiskConfig:
    """Immutable risk scoring configuration."""
    severity_bands: Tuple[SeverityBand, ...]
    bg_ranges: Tuple[Tuple[int, int], ...]
    p_correction_bolus: float = 3.0 / 288.0  # 1/48
    cgm_per_100k_person_years: int = 288 * 365 * 100000
    population_type: str = "adult"


# ============================================================================
# Pure Functions (no side effects)
# ============================================================================

def calculate_lbgi(bg_values: np.ndarray) -> float:
    """
    Calculate LBGI from blood glucose array.
    
    Args:
        bg_values: Array of blood glucose values (mg/dL)
        
    Returns:
        LBGI score
    """
    bg_clean = np.copy(bg_values)
    bg_clean[bg_clean < 1] = 1  # Avoid log(0)
    lbgi, _, _ = blood_glucose_risk_index(bg_clean)
    return lbgi


def extract_lbgi_from_results(
    results_df: pd.DataFrame,
    start_index: int = 137,
    from_first_action: bool = False
) -> float:
    """
    Extract LBGI from simulation results DataFrame.
    
    Args:
        results_df: Simulation results DataFrame with 'bg', 'true_bolus', 'temp_basal' columns
        start_index: Standard start index (skip warm-up period)
        from_first_action: If True, calculate from first Loop action instead
        
    Returns:
        LBGI score
    """
    true_bg = results_df['bg'].values
    
    if from_first_action:
        # Find first insulin action (bolus or basal)
        bolus = results_df['true_bolus'].fillna(0).values
        basal = results_df['temp_basal'].fillna(0).values
        
        first_bolus = len(bolus) if not np.any(bolus > 0) else np.argmax(bolus > 0)
        first_basal = len(basal) if not np.any(basal > 0) else np.argmax(basal > 0)
        start_index = min(first_bolus, first_basal)
    
    return calculate_lbgi(true_bg[start_index:])


def get_severity_band_index(lbgi: float, severity_bands: Tuple[SeverityBand, ...]) -> int:
    """
    Get severity band index for given LBGI value.
    
    Args:
        lbgi: LBGI score
        severity_bands: Tuple of severity band definitions
        
    Returns:
        Index of matching severity band
    """
    for idx, band in enumerate(severity_bands):
        if band.min_lbgi <= lbgi < band.max_lbgi:
            return idx
    return len(severity_bands) - 1  # Return highest severity if no match


def calculate_concurrency_square_probabilities(
    summary_df: pd.DataFrame,
    true_range: Tuple[int, int],
    sensor_range: Tuple[int, int],
    severity_bands: Tuple[SeverityBand, ...]
) -> np.ndarray:
    """
    Calculate probability distribution across severity bands for one BG range square.
    
    In the concurrency table (true BG vs sensor BG grid), each "square" represents
    one combination of true and sensor BG ranges. This function calculates what
    fraction of simulations in that square fall into each severity band.
    
    Args:
        summary_df: Simulation summary with true_start_bg, sensor_start_bg, lbgi columns
        true_range: (low, high) true BG range in mg/dL
        sensor_range: (low, high) sensor BG range in mg/dL
        severity_bands: Severity band definitions
        
    Returns:
        Array of probabilities for each severity band (sums to 1.0 or 0.0 if no data)
    """
    low_true, high_true = true_range
    low_sensor, high_sensor = sensor_range
    
    # Mask simulations in this square
    in_square = (
        (summary_df['tbg'] >= low_true) & 
        (summary_df['tbg'] <= high_true) &
        (summary_df['sbg'] >= low_sensor) & 
        (summary_df['sbg'] <= high_sensor)
    )
    
    square_data = summary_df[in_square]
    
    if len(square_data) == 0:
        return np.zeros(len(severity_bands))
    
    # Count simulations in each severity band
    probabilities = np.zeros(len(severity_bands))
    for idx, band in enumerate(severity_bands):
        in_band = (
            (square_data['lbgi'] >= band.min_lbgi) & 
            (square_data['lbgi'] < band.max_lbgi)
        )
        probabilities[idx] = in_band.sum() / len(square_data)
    
    return probabilities


def compute_risk_table(
    summary_df: pd.DataFrame,
    risk_config: RiskConfig
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Compute risk scores and event probabilities using FDA AI Letter methodology.
    
    Algorithm:
    1. Divide true BG x sensor BG space into grid of ranges (concurrency table)
    2. For each grid square:
       - Calculate probability of each severity band (based on simulations)
       - Weight by Dexcom concurrency table probability (how often this error occurs)
       - Calculate: P(risk event) = P(severity|error) * P(bolus|error) * P(error)
    3. Aggregate across all squares to get events per 100k person-years
    
    Args:
        summary_df: Simulation summaries with columns:
            - true_start_bg: True BG at t0 (mg/dL)
            - sensor_start_bg: Sensor reading at t0 (mg/dL)
            - lbgi_from_start: LBGI from standard start index
            - lbgi_from_first_action: LBGI from first Loop action
        risk_config: Risk scoring configuration
        
    Returns:
        Tuple of (severity_df, analysis_arrays):
        - severity_df: DataFrame with event probabilities per severity band
        - analysis_arrays: Dict with intermediate arrays for visualization
    """
    # Suppress warnings during batch processing
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        
        # Initialize Dexcom concurrency model
        dexcom_model = DexcomG6ValueModel(concurrency_table=risk_config.population_type)
        832-696-5487
        # Storage for results
        num_severity_bands = len(risk_config.severity_bands)
        severity_event_counts = np.zeros(num_severity_bands)
        
        # Analysis arrays for visualization
        true_axis = []
        sensor_axis = []
        severity_probs_by_square = []
        joint_probs = []
        
        # Iterate over all BG range combinations
        for true_range in risk_config.bg_ranges:
            for sensor_range in risk_config.bg_ranges:
                low_true, _ = true_range
                low_sensor, _ = sensor_range
                
                # Store for visualization
                true_axis.append(low_true)
                sensor_axis.append(low_sensor)
                
                # Special case: Skip very low BG edge cases to avoid invalid calculations
                if low_true == 40 and low_sensor in [40, 61]:
                    severity_probs_by_square.append(np.zeros(num_severity_bands))
                    joint_probs.append(0)
                    continue
                
                # Calculate severity probabilities for this square
                severity_probs = calculate_concurrency_square_probabilities(
                    summary_df, true_range, sensor_range, 
                    risk_config.severity_bands
                )
                severity_probs_by_square.append(severity_probs)
                
                # Get Dexcom concurrency probability (how often this error occurs in real data)
                p_error = dexcom_model.get_joint_probability(low_true, low_sensor)
                joint_probs.append(p_error)
                
                # Calculate event counts for each severity band
                for severity_idx, severity_prob in enumerate(severity_probs):
                    # P(risk event) = P(severity|error) * P(correction bolus|error) * P(error)
                    risk_prob = severity_prob * risk_config.p_correction_bolus * p_error
                    
                    # Convert to events per 100k person-years
                    events = risk_prob * risk_config.cgm_per_100k_person_years
                    severity_event_counts[severity_idx] += events
        
        # Create results DataFrame
        severity_event_probs = severity_event_counts / risk_config.cgm_per_100k_person_years
        
        severity_df = pd.DataFrame({
            'severity_band': [b.name for b in risk_config.severity_bands],
            'min_lbgi': [b.min_lbgi for b in risk_config.severity_bands],
            'max_lbgi': [b.max_lbgi for b in risk_config.severity_bands],
            'events_per_100k_years': severity_event_counts,
            'probability': severity_event_probs,
            'safety_threshold': [b.safety_threshold for b in risk_config.severity_bands],
            'passes_threshold': severity_event_probs < [b.safety_threshold for b in risk_config.severity_bands]
        })
        
        # Package analysis arrays for visualization
        analysis_arrays = {
            'true_bg': np.array(true_axis),
            'sensor_bg': np.array(sensor_axis),
            'severity_probs': np.array(severity_probs_by_square),
            'joint_probs': np.array(joint_probs)
        }
        
        logger.info(f"Risk table computed: {severity_event_counts.sum():.2e} total events")
        
        return severity_df, analysis_arrays


def calculate_risk_scores(severity_df: pd.DataFrame) -> np.ndarray:
    """
    Map event probabilities to FDA risk index scores (1-5).
    
    Risk indices are determined by probability thresholds:
    - Index 1: probability < 1e-6
    - Index 2: 1e-6 <= probability < 1e-4
    - Index 3: 1e-4 <= probability < 1e-2
    - Index 4: 1e-2 <= probability < 1e-1
    - Index 5: probability >= 1e-1
    
    Risk scores are: index * severity_weight where severity_weights = [1,2,3,4,5]
    
    Args:
        severity_df: Output from compute_risk_table()
        
    Returns:
        Array of risk scores weighted by severity
    """
    probability_bins = [0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0]
    risk_indices = pd.cut(
        severity_df['probability'],
        bins=probability_bins,
        labels=[1, 2, 3, 4, 5],
        include_lowest=True
    ).astype(int)
    
    severity_weights = np.array([1, 2, 3, 4, 5])
    return risk_indices.values * severity_weights


def generate_risk_report(severity_df: pd.DataFrame) -> str:
    """
    Generate human-readable risk report for regulatory review.
    
    Args:
        severity_df: Output from compute_risk_table()
        
    Returns:
        Formatted risk report string
    """
    risk_scores = calculate_risk_scores(severity_df)
    
    lines = [
        "=" * 80,
        "FDA AI LETTER RISK ANALYSIS REPORT",
        "=" * 80,
        "",
        "Severity Band Analysis",
        "-" * 80,
    ]
    
    for idx, row in severity_df.iterrows():
        status = "✓ PASS" if row['passes_threshold'] else "✗ FAIL"
        lines.extend([
            f"\n{row['severity_band'].upper()} (LBGI {row['min_lbgi']:.1f}-{row['max_lbgi']:.1f})",
            f"  Events: {row['events_per_100k_years']:.2e} per 100k person-years",
            f"  Probability: {row['probability']:.2e}",
            f"  Safety Threshold: {row['safety_threshold']:.2e}",
            f"  Risk Score: {risk_scores[idx]:.0f}",
            f"  Status: {status}"
        ])
    
    lines.extend([
        "",
        "=" * 80,
        f"TOTAL RISK SCORE: {risk_scores.sum():.0f}",
        "=" * 80,
    ])
    
    return "\n".join(lines)


# ============================================================================
# Configuration Factories
# ============================================================================

def create_fda_risk_config(population_type: str = "adult") -> RiskConfig:
    """
    Create standard FDA AI Letter risk configuration.
    
    Args:
        population_type: "adult" or "pediatric" (affects Dexcom concurrency table)
        
    Returns:
        RiskConfig with standard FDA severity bands and BG ranges
    """
    severity_bands = (
        SeverityBand("minimal_risk", 0.0, 2.5, 1.0),
        SeverityBand("low_risk", 2.5, 5.0, 1e-1),
        SeverityBand("moderate_risk", 5.0, 10.0, 1e-2),
        SeverityBand("high_risk", 10.0, 20.0, 1e-4),
        SeverityBand("critical_risk", 20.0, np.inf, 1e-6),
    )
    
    bg_ranges = (
        (40, 60), (61, 80), (81, 120), (121, 160), (161, 200),
        (201, 250), (251, 300), (301, 350), (351, 400)
    )
    
    return RiskConfig(
        severity_bands=severity_bands,
        bg_ranges=bg_ranges,
        population_type=population_type
    )


# ============================================================================
# Convenience Pipeline
# ============================================================================

def analyze_icgm_risk(
    summary_df: pd.DataFrame,
    population_type: str = "adult",
) -> Tuple[pd.DataFrame, Dict[str, np.ndarray], str]:
    """
    Complete risk analysis pipeline in one function call.
    
    This is a convenience function that runs the full risk analysis workflow:
    1. Create standard FDA risk configuration
    2. Compute risk table from simulation data
    3. Generate human-readable report
    
    Args:
        summary_df: Simulation summaries with true_start_bg, sensor_start_bg, lbgi columns
        population_type: "adult" or "pediatric"
        
    Returns:
        Tuple of (severity_df, analysis_arrays, report_text)
        
    Example:
        >>> summary_df = pd.read_csv('icgm_results.tsv', sep='\\t')
        >>> severity_df, arrays, report = analyze_icgm_risk(summary_df)
        >>> print(report)
    """
    config = create_fda_risk_config(population_type)
    severity_df, arrays = compute_risk_table(summary_df, config)
    report = generate_risk_report(severity_df)
    
    return severity_df, arrays, report
