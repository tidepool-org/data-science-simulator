"""
Data processing utilities for insulin algorithm testing framework.

This module provides utilities for processing, aggregating, and transforming
simulation data and results.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
import json
import pickle

logger = logging.getLogger(__name__)


def format_duration(seconds: float) -> str:
    """
    Format a duration in seconds into a human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Human-readable duration string (e.g., "2d 3h 45m 12.5s")
    """
    if seconds < 0:
        return "0s"
    
    # Define time units in seconds
    units = [
        ('d', 86400),   # days
        ('h', 3600),    # hours
        ('m', 60),      # minutes
        ('s', 1)        # seconds
    ]
    
    parts = []
    remaining = seconds
    
    for unit_name, unit_seconds in units:
        if remaining >= unit_seconds:
            if unit_name == 's':
                # For seconds, show decimal places if less than 60 seconds total
                if seconds < 60:
                    value = remaining
                    parts.append(f"{value:.1f}{unit_name}")
                else:
                    value = int(remaining)
                    if value > 0:
                        parts.append(f"{value}{unit_name}")
            else:
                value = int(remaining // unit_seconds)
                parts.append(f"{value}{unit_name}")
                remaining = remaining % unit_seconds
    
    # If no parts, it's less than 1 second
    if not parts:
        return f"{seconds:.1f}s"
    
    return " ".join(parts)


class DataProcessor:
    """
    Utility class for processing simulation data.
    
    Provides methods for cleaning, transforming, and preparing data
    for analysis and visualization.
    """
    
    @staticmethod
    def clean_simulation_results(
        results_df: pd.DataFrame,
        remove_inactive: bool = True,
        interpolate_missing: bool = True
    ) -> pd.DataFrame:
        """
        Clean simulation results DataFrame.
        
        Args:
            results_df: Raw simulation results
            remove_inactive: Whether to remove inactive time periods
            interpolate_missing: Whether to interpolate missing values
            
        Returns:
            Cleaned DataFrame
        """
        df = results_df.copy()
        
        # Remove inactive periods if requested
        if remove_inactive and 'active' in df.columns:
            df = df[df['active'] == 1].copy()
            logger.debug(f"Removed inactive periods, {len(df)} rows remaining")
        
        # Interpolate missing values if requested
        if interpolate_missing:
            numeric_columns = df.select_dtypes(include=[np.number]).columns
            df[numeric_columns] = df[numeric_columns].interpolate(method='linear')
            logger.debug("Interpolated missing values in numeric columns")
        
        # Ensure time index is properly formatted
        if 'time' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
            try:
                df['time'] = pd.to_datetime(df['time'])
                df = df.set_index('time')
            except Exception as e:
                logger.warning(f"Could not convert time column to datetime: {e}")
        
        return df
    
    @staticmethod
    def extract_glucose_segments(
        results_df: pd.DataFrame,
        segment_length_hours: float = 2.0,
        overlap_hours: float = 0.5
    ) -> List[np.ndarray]:
        """
        Extract overlapping glucose segments for analysis.
        
        Args:
            results_df: Simulation results DataFrame
            segment_length_hours: Length of each segment in hours
            overlap_hours: Overlap between segments in hours
            
        Returns:
            List of glucose segments as numpy arrays
        """
        if 'bg' not in results_df.columns:
            logger.warning("No 'bg' column found in results")
            return []
        
        # Clean data
        df = DataProcessor.clean_simulation_results(results_df)
        glucose_values = df['bg'].values
        
        # Calculate segment parameters (assuming 5-minute intervals)
        segment_length_points = int(segment_length_hours * 12)
        overlap_points = int(overlap_hours * 12)
        step_size = segment_length_points - overlap_points
        
        segments = []
        start_idx = 0
        
        while start_idx + segment_length_points <= len(glucose_values):
            segment = glucose_values[start_idx:start_idx + segment_length_points]
            segments.append(segment)
            start_idx += step_size
        
        logger.debug(f"Extracted {len(segments)} glucose segments")
        return segments
    
    @staticmethod
    def calculate_glucose_statistics(
        glucose_data: np.ndarray,
        percentiles: List[float] = [5, 25, 50, 75, 95]
    ) -> Dict[str, float]:
        """
        Calculate comprehensive glucose statistics.
        
        Args:
            glucose_data: Array of glucose values
            percentiles: List of percentiles to calculate
            
        Returns:
            Dictionary of statistics
        """
        if len(glucose_data) == 0:
            return {}
        
        stats = {
            'mean': np.mean(glucose_data),
            'std': np.std(glucose_data),
            'min': np.min(glucose_data),
            'max': np.max(glucose_data),
            'cv': (np.std(glucose_data) / np.mean(glucose_data)) * 100,
            'range': np.max(glucose_data) - np.min(glucose_data)
        }
        
        # Add percentiles
        for p in percentiles:
            stats[f'p{p}'] = np.percentile(glucose_data, p)
        
        # Add time in ranges
        stats['time_in_range_70_180'] = np.mean((glucose_data >= 70) & (glucose_data <= 180)) * 100
        stats['time_in_range_70_140'] = np.mean((glucose_data >= 70) & (glucose_data <= 140)) * 100
        stats['time_below_70'] = np.mean(glucose_data < 70) * 100
        stats['time_below_54'] = np.mean(glucose_data < 54) * 100
        stats['time_above_180'] = np.mean(glucose_data > 180) * 100
        stats['time_above_250'] = np.mean(glucose_data > 250) * 100
        
        return stats
    
    @staticmethod
    def resample_timeseries(
        df: pd.DataFrame,
        target_frequency: str = '5T',
        method: str = 'linear'
    ) -> pd.DataFrame:
        """
        Resample time series data to target frequency.
        
        Args:
            df: DataFrame with datetime index
            target_frequency: Target frequency (e.g., '5T' for 5 minutes)
            method: Interpolation method
            
        Returns:
            Resampled DataFrame
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not datetime, cannot resample")
            return df
        
        # Resample numeric columns
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        resampled = df[numeric_columns].resample(target_frequency).mean()
        
        # Interpolate missing values
        resampled = resampled.interpolate(method=method)
        
        logger.debug(f"Resampled data to {target_frequency} frequency")
        return resampled
    
    @staticmethod
    def align_simulation_results(
        results_dict: Dict[str, pd.DataFrame],
        reference_sim: Optional[str] = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Align multiple simulation results to common time grid.
        
        Args:
            results_dict: Dictionary of simulation_id -> results_df
            reference_sim: Reference simulation for time alignment
            
        Returns:
            Dictionary of aligned results
        """
        if not results_dict:
            return {}
        
        # Determine reference time grid
        if reference_sim and reference_sim in results_dict:
            ref_df = results_dict[reference_sim]
        else:
            # Use the simulation with the most data points
            ref_sim = max(results_dict.keys(), key=lambda k: len(results_dict[k]))
            ref_df = results_dict[ref_sim]
            logger.debug(f"Using {ref_sim} as reference for alignment")
        
        # Get reference time index
        if isinstance(ref_df.index, pd.DatetimeIndex):
            ref_time = ref_df.index
        elif 'time' in ref_df.columns:
            ref_time = pd.to_datetime(ref_df['time'])
        else:
            logger.warning("Cannot determine time index for alignment")
            return results_dict
        
        aligned_results = {}
        
        for sim_id, df in results_dict.items():
            try:
                # Clean and prepare data
                clean_df = DataProcessor.clean_simulation_results(df)
                
                # Align to reference time grid
                if isinstance(clean_df.index, pd.DatetimeIndex):
                    aligned_df = clean_df.reindex(ref_time, method='nearest')
                else:
                    # Fallback: interpolate to same length
                    aligned_df = clean_df.copy()
                    if len(aligned_df) != len(ref_time):
                        # Simple linear interpolation to match length
                        from scipy.interpolate import interp1d
                        numeric_cols = aligned_df.select_dtypes(include=[np.number]).columns
                        
                        for col in numeric_cols:
                            if col in aligned_df.columns:
                                old_x = np.linspace(0, 1, len(aligned_df))
                                new_x = np.linspace(0, 1, len(ref_time))
                                f = interp1d(old_x, aligned_df[col].values, 
                                           kind='linear', fill_value='extrapolate')
                                aligned_df[col] = f(new_x)
                
                aligned_results[sim_id] = aligned_df
                
            except Exception as e:
                logger.warning(f"Could not align simulation {sim_id}: {e}")
                aligned_results[sim_id] = df
        
        logger.debug(f"Aligned {len(aligned_results)} simulation results")
        return aligned_results


class ResultsAggregator:
    """
    Utility class for aggregating and summarizing simulation results.
    
    Provides methods for combining results across multiple simulations
    and creating summary statistics.
    """
    
    def __init__(self):
        """Initialize the results aggregator."""
        self.results_cache = {}
        logger.debug("Initialized ResultsAggregator")
    
    def aggregate_metrics(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        group_by: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Aggregate metrics across simulations.
        
        Args:
            metrics_dict: Dictionary of simulation_id -> metrics_dict
            group_by: Column to group by for aggregation
            
        Returns:
            Aggregated metrics DataFrame
        """
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        if group_by and group_by in metrics_df.columns:
            # Group and aggregate
            aggregated = metrics_df.groupby(group_by).agg({
                col: ['mean', 'std', 'min', 'max', 'count'] 
                for col in metrics_df.select_dtypes(include=[np.number]).columns
            })
            
            # Flatten column names
            aggregated.columns = [f'{col[0]}_{col[1]}' for col in aggregated.columns]
            
        else:
            # Overall aggregation
            numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
            aggregated = metrics_df[numeric_cols].agg(['mean', 'std', 'min', 'max', 'count']).T
        
        logger.debug(f"Aggregated metrics for {len(metrics_dict)} simulations")
        return aggregated
    
    def create_summary_report(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        algorithms: List[str],
        key_metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Create a comprehensive summary report.
        
        Args:
            metrics_dict: Dictionary of simulation_id -> metrics_dict
            algorithms: List of algorithms to compare
            key_metrics: List of key metrics to highlight
            
        Returns:
            Summary report dictionary
        """
        if key_metrics is None:
            key_metrics = [
                'time_in_range_70_180', 'time_below_70', 'time_below_54',
                'mean_glucose', 'cv_glucose', 'cumulative_insulin'
            ]
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Parse algorithm from simulation ID
        def extract_algorithm(sim_id):
            for alg in algorithms:
                if alg in sim_id:
                    return alg
            return 'unknown'
        
        metrics_df['algorithm'] = metrics_df.index.map(extract_algorithm)
        
        report = {
            'total_simulations': len(metrics_dict),
            'algorithms': algorithms,
            'key_metrics': key_metrics,
            'summary_by_algorithm': {},
            'overall_summary': {}
        }
        
        # Summary by algorithm
        for algorithm in algorithms:
            alg_data = metrics_df[metrics_df['algorithm'] == algorithm]
            
            if len(alg_data) > 0:
                alg_summary = {}
                for metric in key_metrics:
                    if metric in alg_data.columns:
                        alg_summary[metric] = {
                            'mean': float(alg_data[metric].mean()),
                            'std': float(alg_data[metric].std()),
                            'median': float(alg_data[metric].median()),
                            'min': float(alg_data[metric].min()),
                            'max': float(alg_data[metric].max()),
                            'count': int(alg_data[metric].count())
                        }
                
                report['summary_by_algorithm'][algorithm] = alg_summary
        
        # Overall summary
        numeric_cols = metrics_df.select_dtypes(include=[np.number]).columns
        overall_stats = metrics_df[numeric_cols].describe()
        report['overall_summary'] = overall_stats.to_dict()
        
        logger.debug(f"Created summary report for {len(algorithms)} algorithms")
        return report
    
    def compare_algorithms(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        reference_algorithm: str,
        comparison_algorithms: List[str],
        metrics: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Compare algorithms against a reference.
        
        Args:
            metrics_dict: Dictionary of simulation_id -> metrics_dict
            reference_algorithm: Reference algorithm name
            comparison_algorithms: List of algorithms to compare
            metrics: List of metrics to compare
            
        Returns:
            Comparison results dictionary
        """
        if metrics is None:
            metrics = ['time_in_range_70_180', 'time_below_70', 'mean_glucose']
        
        # Convert to DataFrame
        metrics_df = pd.DataFrame.from_dict(metrics_dict, orient='index')
        
        # Extract algorithm from simulation ID
        def extract_algorithm(sim_id):
            for alg in [reference_algorithm] + comparison_algorithms:
                if alg in sim_id:
                    return alg
            return 'unknown'
        
        metrics_df['algorithm'] = metrics_df.index.map(extract_algorithm)
        
        comparison_results = {
            'reference_algorithm': reference_algorithm,
            'comparison_algorithms': comparison_algorithms,
            'metrics_compared': metrics,
            'comparisons': {}
        }
        
        # Get reference data
        ref_data = metrics_df[metrics_df['algorithm'] == reference_algorithm]
        
        if len(ref_data) == 0:
            logger.warning(f"No data found for reference algorithm: {reference_algorithm}")
            return comparison_results
        
        # Compare each algorithm
        for comp_alg in comparison_algorithms:
            comp_data = metrics_df[metrics_df['algorithm'] == comp_alg]
            
            if len(comp_data) == 0:
                logger.warning(f"No data found for comparison algorithm: {comp_alg}")
                continue
            
            alg_comparison = {}
            
            for metric in metrics:
                if metric in ref_data.columns and metric in comp_data.columns:
                    ref_values = ref_data[metric].dropna()
                    comp_values = comp_data[metric].dropna()
                    
                    if len(ref_values) > 0 and len(comp_values) > 0:
                        # Calculate differences
                        ref_mean = ref_values.mean()
                        comp_mean = comp_values.mean()
                        
                        absolute_diff = comp_mean - ref_mean
                        relative_diff = (absolute_diff / ref_mean) * 100 if ref_mean != 0 else 0
                        
                        alg_comparison[metric] = {
                            'reference_mean': float(ref_mean),
                            'comparison_mean': float(comp_mean),
                            'absolute_difference': float(absolute_diff),
                            'relative_difference_percent': float(relative_diff),
                            'reference_std': float(ref_values.std()),
                            'comparison_std': float(comp_values.std())
                        }
            
            comparison_results['comparisons'][comp_alg] = alg_comparison
        
        logger.debug(f"Compared {len(comparison_algorithms)} algorithms against {reference_algorithm}")
        return comparison_results
    
    def export_summary(
        self,
        summary_data: Dict[str, Any],
        output_path: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Export summary data to file.
        
        Args:
            summary_data: Summary data dictionary
            output_path: Output file path
            format: Export format ('json', 'pickle')
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format.lower() == 'json':
            with open(output_path, 'w') as f:
                json.dump(summary_data, f, indent=2, default=str)
        
        elif format.lower() == 'pickle':
            with open(output_path, 'wb') as f:
                pickle.dump(summary_data, f)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        logger.info(f"Exported summary data to {output_path}")
