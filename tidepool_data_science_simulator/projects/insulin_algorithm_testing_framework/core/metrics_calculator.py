"""
Functional metrics calculator for insulin algorithm testing.
"""

import logging
import os
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from multiprocessing import Pool
from functools import partial

from tidepool_data_science_metrics.glucose.glucose import (
    percent_values_ge_70_le_180, percent_values_lt_70, percent_values_lt_54,
    percent_values_gt_180, percent_values_gt_250, blood_glucose_risk_index,
    lbgi_risk_score
)

logger = logging.getLogger(__name__)


@dataclass
class PointMetrics:
    """Container for calculated point metrics."""
    
    # Time in range
    time_in_range_70_180: float
    time_below_70: float
    time_below_54: float
    time_above_180: float
    time_above_250: float
    time_in_tight_range_70_140: float
    
    # Glucose statistics
    mean_glucose: float
    median_glucose: float
    std_glucose: float
    cv_glucose: float
    gmi: float
    
    # Risk indices
    lbgi: float
    hbgi: float
    bgri: float
    lbgi_risk_score: float
    
    # Insulin delivery
    total_insulin: float
    basal_insulin: float
    bolus_insulin: float
    max_bolus_delivered: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {k: v for k, v in self.__dict__.items()}


# Keep old name as alias if needed for compatibility
MetricsResult = PointMetrics


# =============================================================================
# Data extraction - just slicing, no aggregation
# =============================================================================

def slice_active_data(
    results_df: pd.DataFrame,
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> pd.DataFrame:
    """
    Slice results to active data within the specified time window.
    
    Args:
        results_df: Simulation results DataFrame
        start_hours: Start time in hours (5-min intervals assumed)
        duration_hours: Duration to analyze (None for all remaining)
        
    Returns:
        Sliced DataFrame containing only active rows in the time window
    """
    active_data = results_df[results_df['active'] == 1]
    
    start_idx = int(start_hours * 12)
    end_idx = start_idx + int(duration_hours * 12) if duration_hours else len(active_data)
    
    return active_data.iloc[start_idx:end_idx]


# =============================================================================
# Glucose metrics
# =============================================================================

def calculate_time_in_ranges(bg: np.ndarray) -> Dict[str, float]:
    """Calculate time in range metrics from glucose array."""
    if len(bg) == 0:
        return dict.fromkeys([
            'time_in_range_70_180', 'time_below_70', 'time_below_54',
            'time_above_180', 'time_above_250', 'time_in_tight_range_70_140'
        ], 0.0)
    
    bg_clipped = np.clip(bg, 1, 401)
    
    return {
        'time_in_range_70_180': percent_values_ge_70_le_180(bg_clipped),
        'time_below_70': percent_values_lt_70(bg_clipped),
        'time_below_54': percent_values_lt_54(bg_clipped),
        'time_above_180': percent_values_gt_180(bg_clipped),
        'time_above_250': percent_values_gt_250(bg_clipped),
        'time_in_tight_range_70_140': np.mean((bg >= 70) & (bg <= 140)) * 100
    }


def calculate_glucose_stats(bg: np.ndarray) -> Dict[str, float]:
    """Calculate basic glucose statistics."""
    if len(bg) == 0:
        return dict.fromkeys(['mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'gmi'], 0.0)
    
    mean_bg = np.mean(bg)
    std_bg = np.std(bg)
    
    return {
        'mean_glucose': mean_bg,
        'median_glucose': np.median(bg),
        'std_glucose': std_bg,
        'cv_glucose': (std_bg / mean_bg * 100) if mean_bg > 0 else 0.0,
        'gmi': 3.31 + 0.02392 * mean_bg
    }


def calculate_risk_indices(bg: np.ndarray) -> Dict[str, float]:
    """Calculate glucose risk indices."""
    if len(bg) == 0:
        return dict.fromkeys(['lbgi', 'hbgi', 'bgri', 'lbgi_risk_score'], 0.0)
    
    bg_clipped = np.clip(bg, 1, 401)
    lbgi, hbgi, bgri = blood_glucose_risk_index(bg_clipped)
    
    return {
        'lbgi': lbgi,
        'hbgi': hbgi,
        'bgri': bgri,
        'lbgi_risk_score': lbgi_risk_score(lbgi)
    }


# =============================================================================
# Insulin metrics
# =============================================================================

def calculate_insulin_totals(sliced_df: pd.DataFrame) -> Dict[str, float]:
    """Calculate insulin delivery totals from sliced data."""
    if len(sliced_df) == 0:
        return {'total_insulin': 0.0, 'basal_insulin': 0.0, 'bolus_insulin': 0.0, 'max_bolus_delivered': 0.0}
    
    basal = sliced_df['delivered_basal_insulin'].sum()
    bolus = sliced_df['true_bolus'].sum()
    max_bolus = sliced_df['true_bolus'].max()
    
    return {
        'basal_insulin': basal,
        'bolus_insulin': bolus,
        'total_insulin': basal + bolus,
        'max_bolus_delivered': max_bolus if not np.isnan(max_bolus) else 0.0
    }


def calculate_insulin_cumsum(sliced_df: pd.DataFrame) -> np.ndarray:
    """Calculate cumulative insulin delivery over time."""
    if len(sliced_df) == 0:
        return np.array([])
    
    basal_cumsum = sliced_df['delivered_basal_insulin'].fillna(0).cumsum().values
    bolus_cumsum = sliced_df['true_bolus'].fillna(0).cumsum().values
    
    return basal_cumsum + bolus_cumsum


# =============================================================================
# Main calculation functions
# =============================================================================

def calculate_point_metrics(
    results_df: pd.DataFrame,
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> PointMetrics:
    """Calculate all point metrics for a simulation result."""
    sliced = slice_active_data(results_df, start_hours, duration_hours)
    
    if len(sliced) == 0:
        logger.warning("No data available for metrics calculation")
        return _empty_point_metrics()
    
    bg = sliced['bg'].values
    
    metrics = {
        **calculate_time_in_ranges(bg),
        **calculate_glucose_stats(bg),
        **calculate_risk_indices(bg),
        **calculate_insulin_totals(sliced)
    }
    
    return PointMetrics(**metrics)


def calculate_timeseries_metrics(
    results_df: pd.DataFrame,
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> np.ndarray:
    """Calculate cumulative insulin timeseries."""
    sliced = slice_active_data(results_df, start_hours, duration_hours)
    return calculate_insulin_cumsum(sliced)


def _empty_point_metrics() -> PointMetrics:
    """Create empty metrics for error cases."""
    return PointMetrics(
        **dict.fromkeys([
            'time_in_range_70_180', 'time_below_70', 'time_below_54',
            'time_above_180', 'time_above_250', 'time_in_tight_range_70_140',
            'mean_glucose', 'median_glucose', 'std_glucose', 'cv_glucose', 'gmi',
            'lbgi', 'hbgi', 'bgri', 'lbgi_risk_score',
            'total_insulin', 'basal_insulin', 'bolus_insulin', 'max_bolus_delivered'
        ], 0.0)
    )


# =============================================================================
# Batch processing
# =============================================================================

def calculate_metrics_batch(
    results_dict: Dict[str, pd.DataFrame],
    start_hours: float = 0,
    duration_hours: Optional[float] = None,
    include_timeseries: bool = True
) -> Tuple[Dict[str, PointMetrics], Dict[str, np.ndarray]]:
    """
    Calculate metrics for multiple simulations.
    
    Returns:
        Tuple of (point_metrics_dict, timeseries_dict)
    """
    point_metrics = {}
    timeseries = {}
    
    for sim_id, df in results_dict.items():
        try:
            point_metrics[sim_id] = calculate_point_metrics(df, start_hours, duration_hours)
            if include_timeseries:
                timeseries[sim_id] = calculate_timeseries_metrics(df, start_hours, duration_hours)
        except Exception as e:
            logger.error(f"Error calculating metrics for {sim_id}: {e}")
            point_metrics[sim_id] = _empty_point_metrics()
            timeseries[sim_id] = np.array([])
    
    return point_metrics, timeseries


# =============================================================================
# Output helpers
# =============================================================================

def parse_simulation_id(sim_id: str) -> Dict[str, Any]:
    """Parse simulation info from ID string like 'alg=tempbasal_patient=1_ibg=100'."""
    info = {}
    
    for part in sim_id.split('_'):
        if '=' not in part:
            continue
        key, value = part.split('=', 1)
        
        # Handle meal size suffix
        if key == 'meal' and value.endswith('g'):
            value = value[:-1]
        
        # Convert to numeric if possible
        try:
            info[key] = float(value) if '.' in value else int(value)
        except ValueError:
            info[key] = value
    
    return info


def metrics_to_dataframe(
    metrics_dict: Dict[str, PointMetrics],
    parse_sim_ids: bool = True
) -> pd.DataFrame:
    """Convert metrics dict to DataFrame."""
    rows = []
    for sim_id, metrics in metrics_dict.items():
        row = {'simulation_id': sim_id, **metrics.to_dict()}
        if parse_sim_ids:
            row.update(parse_simulation_id(sim_id))
        rows.append(row)
    
    return pd.DataFrame(rows)


def timeseries_to_array(
    timeseries_dict: Dict[str, np.ndarray],
    sim_ids: List[str]
) -> np.ndarray:
    """
    Stack timeseries into 2D array (n_sims, max_timepoints), NaN-padded.
    
    Args:
        timeseries_dict: {sim_id: 1D array}
        sim_ids: Ordered list of simulation IDs (determines row order)
    """
    if not timeseries_dict or not sim_ids:
        return np.array([])
    
    max_len = max(
        (len(timeseries_dict.get(sid, [])) for sid in sim_ids),
        default=0
    )
    
    if max_len == 0:
        return np.array([])
    
    result = np.full((len(sim_ids), max_len), np.nan)
    
    for i, sim_id in enumerate(sim_ids):
        arr = timeseries_dict.get(sim_id, np.array([]))
        if len(arr) > 0:
            result[i, :len(arr)] = arr
    
    return result


# =============================================================================
# Parquet file processing with parallel metrics calculation
# =============================================================================

def _process_single_simulation(
    args: Tuple[str, pd.DataFrame],
    start_hours: float = 0,
    duration_hours: Optional[float] = None
) -> Tuple[str, PointMetrics]:
    """
    Worker function to process a single simulation's metrics.
    
    Designed for use with multiprocessing.Pool.
    
    Args:
        args: Tuple of (sim_id, simulation_dataframe)
        start_hours: Start time offset for metrics calculation
        duration_hours: Duration to analyze (None for all)
        
    Returns:
        Tuple of (sim_id, PointMetrics)
    """
    sim_id, sim_df = args
    try:
        metrics = calculate_point_metrics(sim_df, start_hours, duration_hours)
        return (sim_id, metrics)
    except Exception as e:
        logger.error(f"Error processing simulation {sim_id}: {e}")
        return (sim_id, _empty_point_metrics())


def calculate_metrics_from_parquet(
    parquet_path: str,
    n_processes: Optional[int] = None,
    start_hours: float = 0,
    duration_hours: Optional[float] = None,
    show_progress: bool = True
) -> Tuple[Dict[str, PointMetrics], Optional[Dict[str, Any]]]:
    """
    Load parquet file and calculate metrics for all simulations in parallel.
    
    This function provides an efficient way to process large parquet files
    containing multiple simulations. It uses multiprocessing to calculate
    metrics in parallel across CPU cores.
    
    Args:
        parquet_path: Path to combined_results.parquet file
        n_processes: Number of parallel workers (default: cpu_count)
        start_hours: Start time offset for metrics calculation
        duration_hours: Duration to analyze (None for all remaining)
        show_progress: Whether to log progress updates
        
    Returns:
        Tuple of:
        - point_metrics_dict: {sim_id: PointMetrics}
        - metadata: Simulation metadata dict keyed by sim_id (or None)
        
    Example:
        >>> metrics, metadata = calculate_metrics_from_parquet(
        ...     'results/combined_results.parquet',
        ...     n_processes=8
        ... )
        >>> df = metrics_to_dataframe(metrics)
    """
    from tidepool_data_science_simulator.utils import load_streaming_parquet_with_metadata
    
    # Load parquet file
    if show_progress:
        logger.info(f"Loading parquet file: {parquet_path}")
    
    df, metadata = load_streaming_parquet_with_metadata(parquet_path)
    
    # Get unique simulation IDs
    sim_ids = df['sim_id'].unique()
    n_sims = len(sim_ids)
    
    if show_progress:
        logger.info(f"Found {n_sims} simulations to process")
    
    # Create list of (sim_id, df) tuples for processing
    sim_groups = [(sim_id, df[df['sim_id'] == sim_id].copy()) for sim_id in sim_ids]
    
    # Set up parallel processing
    n_processes = n_processes or os.cpu_count()
    n_processes = min(n_processes, n_sims)  # Don't use more workers than simulations
    
    if show_progress:
        logger.info(f"Processing metrics with {n_processes} parallel workers...")
    
    # Create worker function with fixed parameters
    worker_fn = partial(
        _process_single_simulation,
        start_hours=start_hours,
        duration_hours=duration_hours
    )
    
    # Process in parallel
    if n_processes > 1 and n_sims > 1:
        with Pool(n_processes) as pool:
            results = pool.map(worker_fn, sim_groups)
    else:
        # Single process for debugging or small batches
        results = [worker_fn(sg) for sg in sim_groups]
    
    # Collect results into dict
    point_metrics = {sim_id: metrics for sim_id, metrics in results}
    
    if show_progress:
        logger.info(f"Completed metrics calculation for {len(point_metrics)} simulations")
    
    return point_metrics, metadata


def calculate_metrics_from_parquet_streaming(
    parquet_path: str,
    n_processes: Optional[int] = None,
    start_hours: float = 0,
    duration_hours: Optional[float] = None,
    batch_size: int = 100,
    show_progress: bool = True
) -> Tuple[Dict[str, PointMetrics], Optional[Dict[str, Any]]]:
    """
    Process parquet file in batches to limit memory usage.
    
    For very large parquet files, this function processes simulations in
    batches to avoid loading everything into memory at once.
    
    Args:
        parquet_path: Path to combined_results.parquet file
        n_processes: Number of parallel workers (default: cpu_count)
        start_hours: Start time offset for metrics calculation
        duration_hours: Duration to analyze (None for all remaining)
        batch_size: Number of simulations to process per batch
        show_progress: Whether to log progress updates
        
    Returns:
        Tuple of (point_metrics_dict, metadata_dict)
    """
    import pyarrow.parquet as pq
    import json
    
    # Get metadata first
    metadata_path = parquet_path.replace('.parquet', '_metadata.parquet')
    metadata = None
    if os.path.exists(metadata_path):
        metadata_table = pq.read_table(metadata_path)
        metadata_df = metadata_table.to_pandas()
        metadata = {}
        for _, row in metadata_df.iterrows():
            metadata[row['sim_id']] = json.loads(row['metadata_json'])
    
    # Read parquet file to get unique sim_ids
    parquet_file = pq.ParquetFile(parquet_path)
    
    # First pass: get all unique sim_ids
    all_sim_ids = set()
    for batch in parquet_file.iter_batches(columns=['sim_id']):
        all_sim_ids.update(batch.to_pandas()['sim_id'].unique())
    
    sim_ids = list(all_sim_ids)
    n_sims = len(sim_ids)
    
    if show_progress:
        logger.info(f"Found {n_sims} simulations to process in streaming mode")
    
    # Set up parallel processing
    n_processes = n_processes or os.cpu_count()
    n_processes = min(n_processes, batch_size)
    
    worker_fn = partial(
        _process_single_simulation,
        start_hours=start_hours,
        duration_hours=duration_hours
    )
    
    # Process in batches
    point_metrics = {}
    n_batches = (n_sims + batch_size - 1) // batch_size
    
    for batch_idx in range(n_batches):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, n_sims)
        batch_sim_ids = sim_ids[batch_start:batch_end]
        
        if show_progress:
            logger.info(f"Processing batch {batch_idx + 1}/{n_batches} "
                       f"(simulations {batch_start + 1}-{batch_end}/{n_sims})")
        
        # Load data for this batch of sim_ids
        df = pq.read_table(parquet_path).to_pandas()
        batch_df = df[df['sim_id'].isin(batch_sim_ids)]
        
        # Create groups for this batch
        sim_groups = [
            (sim_id, batch_df[batch_df['sim_id'] == sim_id].copy())
            for sim_id in batch_sim_ids
        ]
        
        # Process batch in parallel
        if n_processes > 1 and len(sim_groups) > 1:
            with Pool(n_processes) as pool:
                results = pool.map(worker_fn, sim_groups)
        else:
            results = [worker_fn(sg) for sg in sim_groups]
        
        # Collect results
        for sim_id, metrics in results:
            point_metrics[sim_id] = metrics
        
        # Clear memory
        del batch_df, df, sim_groups, results
    
    if show_progress:
        logger.info(f"Completed streaming metrics calculation for {len(point_metrics)} simulations")
    
    return point_metrics, metadata
