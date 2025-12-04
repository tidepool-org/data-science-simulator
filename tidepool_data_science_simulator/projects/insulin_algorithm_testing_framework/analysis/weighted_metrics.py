"""
Weighted statistical analysis for population-representative metrics.

This module provides functions for calculating weighted statistics using
initial blood glucose (IBG) distribution weights. This ensures metrics
reflect real-world population distributions rather than uniform sampling.
"""

import logging
from typing import Dict, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ============================================================================
# Weighted Statistical Functions (Pure)
# ============================================================================

def weighted_percentile(
    values: np.ndarray, 
    weights: np.ndarray, 
    percentiles: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """
    Calculate weighted percentile(s).
    
    Args:
        values: Data values
        weights: Weights for each value (should sum to 1 or will be normalized)
        percentiles: Percentile(s) to calculate (0-100)
        
    Returns:
        Percentile value(s)
        
    Example:
        >>> values = np.array([1, 2, 3, 4, 5])
        >>> weights = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        >>> weighted_percentile(values, weights, 50)
        3.0
    """
    # Sort by values
    sorted_idx = np.argsort(values)
    sorted_values = values[sorted_idx]
    sorted_weights = weights[sorted_idx]
    
    # Compute cumulative weights
    cum_weights = np.cumsum(sorted_weights)
    total_weight = cum_weights[-1]
    
    # Normalize to percentages
    cum_percentages = 100 * cum_weights / total_weight
    
    # Interpolate to find percentile values
    percentiles_array = np.atleast_1d(percentiles)
    result = np.interp(percentiles_array, cum_percentages, sorted_values)
    
    # Return scalar if input was scalar
    if np.isscalar(percentiles):
        return float(result[0])
    return result


def weighted_iqr(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted interquartile range (Q3 - Q1).
    
    Args:
        values: Data values
        weights: Weights for each value
        
    Returns:
        Weighted IQR
    """
    q75, q25 = weighted_percentile(values, weights, [75, 25])
    return q75 - q25


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted mean.
    
    Args:
        values: Data values
        weights: Weights for each value
        
    Returns:
        Weighted mean
    """
    return np.average(values, weights=weights)


def weighted_std(
    values: np.ndarray, 
    weights: np.ndarray,
    ddof: int = 0
) -> float:
    """
    Calculate weighted standard deviation.
    
    Args:
        values: Data values
        weights: Weights for each value
        ddof: Delta degrees of freedom (0 for population, 1 for sample)
        
    Returns:
        Weighted standard deviation
    """
    mean = weighted_mean(values, weights)
    variance = np.average((values - mean) ** 2, weights=weights)
    
    # Apply degrees of freedom correction if requested
    if ddof > 0:
        # Bessel's correction for weighted samples
        sum_weights = np.sum(weights)
        sum_squared_weights = np.sum(weights ** 2)
        variance *= sum_weights / (sum_weights - ddof * sum_squared_weights / sum_weights)
    
    return np.sqrt(variance)


def weighted_mean_std(
    values: np.ndarray, 
    weights: np.ndarray
) -> Tuple[float, float]:
    """
    Calculate weighted mean and standard deviation together.
    
    Args:
        values: Data values
        weights: Weights for each value
        
    Returns:
        Tuple of (mean, std)
    """
    mean = weighted_mean(values, weights)
    std = weighted_std(values, weights)
    return mean, std


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    """
    Calculate weighted median (equivalent to 50th percentile).
    
    Args:
        values: Data values
        weights: Weights for each value
        
    Returns:
        Weighted median
    """
    return weighted_percentile(values, weights, 50)


# ============================================================================
# IBG Histogram Loading
# ============================================================================

def load_ibg_histogram(
    histogram_path: Union[str, Path],
    ibg_column: str = 'ibg',
    proportion_column: str = 'proportion'
) -> Dict[float, float]:
    """
    Load initial blood glucose (IBG) histogram weights from CSV.
    
    The histogram represents the distribution of initial BG values in a
    reference population (e.g., from clinical data). This allows computing
    population-weighted metrics rather than treating all IBG values equally.
    
    Args:
        histogram_path: Path to CSV file with IBG distribution
        ibg_column: Name of column containing IBG values
        proportion_column: Name of column containing proportions/weights
        
    Returns:
        Dictionary mapping IBG value -> proportion
        
    Example CSV format:
        ibg,proportion
        70,0.05
        80,0.10
        90,0.15
        ...
    """
    histogram_path = Path(histogram_path)
    
    if not histogram_path.exists():
        raise FileNotFoundError(f"IBG histogram not found: {histogram_path}")
    
    df = pd.read_csv(histogram_path)
    
    if ibg_column not in df.columns or proportion_column not in df.columns:
        raise ValueError(
            f"CSV must contain '{ibg_column}' and '{proportion_column}' columns. "
            f"Found: {df.columns.tolist()}"
        )
    
    # Create dictionary mapping IBG -> proportion
    ibg_weights = {
        row[ibg_column]: row[proportion_column] 
        for _, row in df.iterrows()
    }
    
    # Validate proportions sum to approximately 1
    total_proportion = sum(ibg_weights.values())
    if not np.isclose(total_proportion, 1.0, atol=0.01):
        logger.warning(
            f"IBG histogram proportions sum to {total_proportion:.4f}, not 1.0. "
            "Consider normalizing the histogram."
        )
    
    logger.info(f"Loaded IBG histogram with {len(ibg_weights)} values from {histogram_path}")
    return ibg_weights


def normalize_ibg_weights(ibg_weights: Dict[float, float]) -> Dict[float, float]:
    """
    Normalize IBG weights to sum to 1.0.
    
    Args:
        ibg_weights: Dictionary of IBG -> proportion
        
    Returns:
        Normalized dictionary
    """
    total = sum(ibg_weights.values())
    return {ibg: weight / total for ibg, weight in ibg_weights.items()}


# ============================================================================
# Metrics DataFrame Weighting
# ============================================================================

def add_weights_to_metrics(
    metrics_df: pd.DataFrame,
    ibg_weights: Dict[float, float],
    ibg_column: str = 'initial_bg',
    weight_column: str = 'weight'
) -> pd.DataFrame:
    """
    Add weight column to metrics DataFrame based on IBG values.
    
    Args:
        metrics_df: DataFrame with metrics (must have initial_bg column)
        ibg_weights: Dictionary mapping IBG value -> weight
        ibg_column: Name of IBG column in metrics_df
        weight_column: Name for new weight column
        
    Returns:
        DataFrame with added weight column
    """
    metrics_df = metrics_df.copy()
    
    # Map IBG values to weights
    metrics_df[weight_column] = metrics_df[ibg_column].map(ibg_weights)
    
    # Check for missing weights
    missing_weights = metrics_df[weight_column].isna().sum()
    if missing_weights > 0:
        logger.warning(
            f"{missing_weights} rows have no matching IBG weight. "
            "These will be excluded from weighted calculations."
        )
        # Fill missing with 0 so they don't contribute
        metrics_df[weight_column] = metrics_df[weight_column].fillna(0)
    
    return metrics_df


def calculate_weighted_summary(
    metrics_df: pd.DataFrame,
    metric_columns: list,
    weight_column: str = 'weight',
    statistics: list = ['mean', 'std', 'median', 'q25', 'q75', 'iqr']
) -> pd.DataFrame:
    """
    Calculate weighted summary statistics for multiple metrics.
    
    Args:
        metrics_df: DataFrame with metrics and weights
        metric_columns: List of metric column names to summarize
        weight_column: Name of weight column
        statistics: List of statistics to compute
        
    Returns:
        DataFrame with weighted statistics
    """
    results = []
    
    for metric_col in metric_columns:
        # Extract values and weights, removing NaN
        mask = metrics_df[metric_col].notna() & metrics_df[weight_column].notna()
        values = metrics_df.loc[mask, metric_col].values
        weights = metrics_df.loc[mask, weight_column].values
        
        if len(values) == 0:
            logger.warning(f"No valid data for metric: {metric_col}")
            continue
        
        # Calculate requested statistics
        stats = {'metric': metric_col}
        
        if 'mean' in statistics:
            stats['mean'] = weighted_mean(values, weights)
        
        if 'std' in statistics:
            stats['std'] = weighted_std(values, weights)
        
        if 'median' in statistics:
            stats['median'] = weighted_median(values, weights)
        
        if 'q25' in statistics:
            stats['q25'] = weighted_percentile(values, weights, 25)
        
        if 'q75' in statistics:
            stats['q75'] = weighted_percentile(values, weights, 75)
        
        if 'iqr' in statistics:
            stats['iqr'] = weighted_iqr(values, weights)
        
        if 'min' in statistics:
            stats['min'] = np.min(values)
        
        if 'max' in statistics:
            stats['max'] = np.max(values)
        
        results.append(stats)
    
    return pd.DataFrame(results)


# ============================================================================
# Paired Comparison with Weights
# ============================================================================

def calculate_weighted_paired_difference(
    metrics_df: pd.DataFrame,
    metric_column: str,
    algorithm_column: str = 'algorithm',
    reference_algorithm: str = 'tempbasal',
    comparison_algorithm: str = 'autobolus',
    pairing_columns: list = None,
    weight_column: str = 'weight'
) -> pd.DataFrame:
    """
    Calculate weighted paired differences between algorithms.
    
    This function assumes each row in the pairing columns represents a
    matched scenario (e.g., same patient, same initial BG, same settings).
    
    Args:
        metrics_df: DataFrame with metrics for both algorithms
        metric_column: Name of metric to compare
        algorithm_column: Name of column indicating algorithm
        reference_algorithm: Reference algorithm name
        comparison_algorithm: Comparison algorithm name
        pairing_columns: Columns that define paired scenarios
        weight_column: Name of weight column
        
    Returns:
        DataFrame with paired differences and weights
    """
    if pairing_columns is None:
        pairing_columns = ['initial_bg', 'patient_id']
    
    # Split into reference and comparison
    ref_df = metrics_df[metrics_df[algorithm_column] == reference_algorithm].copy()
    comp_df = metrics_df[metrics_df[algorithm_column] == comparison_algorithm].copy()
    
    # Merge on pairing columns
    paired_df = ref_df.merge(
        comp_df,
        on=pairing_columns,
        suffixes=('_ref', '_comp')
    )
    
    # Calculate difference
    paired_df['difference'] = (
        paired_df[f'{metric_column}_comp'] - 
        paired_df[f'{metric_column}_ref']
    )
    
    # Use weight from reference (should be same for both)
    paired_df['weight'] = paired_df[f'{weight_column}_ref']
    
    return paired_df


# ============================================================================
# Convenience Functions
# ============================================================================

def weighted_comparison_summary(
    metrics_df: pd.DataFrame,
    metric_columns: list,
    algorithm_column: str = 'algorithm',
    weight_column: str = 'weight'
) -> pd.DataFrame:
    """
    Generate weighted summary comparison across algorithms.
    
    Args:
        metrics_df: DataFrame with metrics for multiple algorithms
        metric_columns: Metrics to summarize
        algorithm_column: Column indicating algorithm
        weight_column: Weight column name
        
    Returns:
        DataFrame with algorithm x metric summary statistics
    """
    summaries = []
    
    for algorithm in metrics_df[algorithm_column].unique():
        algo_df = metrics_df[metrics_df[algorithm_column] == algorithm]
        
        summary = calculate_weighted_summary(
            algo_df, 
            metric_columns, 
            weight_column
        )
        summary['algorithm'] = algorithm
        summaries.append(summary)
    
    return pd.concat(summaries, ignore_index=True)
