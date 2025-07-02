"""
Comparison plotting utilities for insulin algorithm testing framework.

This module provides functions for creating comprehensive comparison plots
between different insulin delivery algorithms.
"""

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

from ..config.experiment_config import ExperimentConfig

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ComparisonPlotter:
    """
    Creates comparison plots for insulin algorithm analysis.
    
    Provides methods for visualizing metrics comparisons, glucose traces,
    and statistical analysis results.
    """
    
    def __init__(self, config: ExperimentConfig):
        """
        Initialize the comparison plotter.
        
        Args:
            config: Experiment configuration
        """
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Set up plotting parameters
        self.colors = {
            'temp_basal': '#1f77b4',  # Blue
            'autobolus': '#ff7f0e',   # Orange
            'reference': '#2ca02c',    # Green
            'comparison': '#d62728'    # Red
        }
        
        self.markers = {
            'temp_basal': 'o',
            'autobolus': 's',
            'reference': '^',
            'comparison': 'v'
        }
        
        logger.info("Initialized ComparisonPlotter")
    
    def plot_algorithm_comparison(
        self,
        metrics_df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        show_individual_points: bool = True,
        figsize: Tuple[int, int] = (15, 10)
    ) -> None:
        """
        Create comprehensive algorithm comparison plots.
        
        Args:
            metrics_df: DataFrame with metrics for each simulation
            metrics: List of metrics to plot (None for default set)
            save_path: Path to save the plot
            show_individual_points: Whether to show individual data points
            figsize: Figure size tuple
        """
        if metrics is None:
            metrics = [
                'time_in_range_70_180', 'time_below_70', 'time_below_54',
                'mean_glucose', 'cv_glucose', 'cumulative_insulin'
            ]
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in metrics_df.columns]
        
        if not available_metrics:
            logger.warning("No valid metrics found for plotting")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        n_cols = 3
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle('Algorithm Comparison: Key Metrics', fontsize=16, fontweight='bold')
        
        for i, metric in enumerate(available_metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            self._plot_metric_comparison(metrics_df, metric, ax, show_individual_points)
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Algorithm comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def _plot_metric_comparison(
        self,
        metrics_df: pd.DataFrame,
        metric: str,
        ax: plt.Axes,
        show_individual_points: bool = True
    ) -> None:
        """Plot comparison for a single metric."""
        
        if 'alg' not in metrics_df.columns:
            logger.warning(f"Algorithm column 'alg' not found in metrics_df")
            return
        
        # Create box plot
        algorithms = metrics_df['alg'].unique()
        
        if show_individual_points:
            # Box plot with individual points
            sns.boxplot(data=metrics_df, x='alg', y=metric, ax=ax, 
                       palette=[self.colors.get(alg, 'gray') for alg in algorithms],
                       hue='alg')
            sns.stripplot(data=metrics_df, x='alg', y=metric, ax=ax, 
                         color='black', alpha=0.5, size=3)
        else:
            # Just box plot
            sns.boxplot(data=metrics_df, x='alg', y=metric, ax=ax,
                       palette=[self.colors.get(alg, 'gray') for alg in algorithms])
        
        # Formatting
        ax.set_title(self._format_metric_name(metric), fontweight='bold')
        ax.set_xlabel('Algorithm')
        ax.set_ylabel(self._get_metric_units(metric))
        ax.grid(True, alpha=0.3)
        
        # Add statistical annotations if possible
        self._add_statistical_annotations(metrics_df, metric, ax)
    
    def plot_glucose_traces_sample(
        self,
        results_dict: Dict[str, pd.DataFrame],
        n_samples: int = 6,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (15, 12)
    ) -> None:
        """
        Plot sample glucose traces for visual comparison.
        
        Args:
            results_dict: Dictionary of simulation_id -> results_df
            n_samples: Number of sample traces to plot
            save_path: Path to save the plot
            figsize: Figure size tuple
        """
        # Sample simulations
        sim_ids = list(results_dict.keys())
        if len(sim_ids) > n_samples:
            np.random.seed(42)  # For reproducibility
            sim_ids = np.random.choice(sim_ids, n_samples, replace=False)
        
        # Group by algorithm
        algorithm_traces = {}
        for sim_id in sim_ids:
            # Extract algorithm from simulation ID
            if 'temp_basal' in sim_id:
                alg = 'temp_basal'
            elif 'autobolus' in sim_id:
                alg = 'autobolus'
            else:
                alg = 'unknown'
            
            if alg not in algorithm_traces:
                algorithm_traces[alg] = []
            
            algorithm_traces[alg].append((sim_id, results_dict[sim_id]))
        
        # Create plots
        n_algorithms = len(algorithm_traces)
        fig, axes = plt.subplots(n_algorithms, 1, figsize=figsize, sharex=True)
        
        if n_algorithms == 1:
            axes = [axes]
        
        fig.suptitle('Sample Glucose Traces by Algorithm', fontsize=16, fontweight='bold')
        
        for i, (algorithm, traces) in enumerate(algorithm_traces.items()):
            ax = axes[i]
            
            for j, (sim_id, results_df) in enumerate(traces):
                # Extract active data
                active_data = results_df[results_df['active'] == 1]
                
                if len(active_data) == 0:
                    continue
                
                # Plot glucose trace
                time_hours = np.arange(len(active_data)) / 12  # 5-minute intervals
                
                ax.plot(time_hours, active_data['bg'].values, 
                       color=self.colors.get(algorithm, 'gray'),
                       alpha=0.7, linewidth=1.5,
                       label=f'{algorithm}' if j == 0 else '')
                
                # Add target ranges
                if j == 0:
                    ax.axhspan(70, 180, alpha=0.1, color='green', label='Target Range (70-180)')
                    ax.axhspan(70, 140, alpha=0.1, color='lightgreen', label='Tight Range (70-140)')
                    ax.axhline(70, color='red', linestyle='--', alpha=0.5, label='Hypoglycemia (<70)')
                    ax.axhline(250, color='orange', linestyle='--', alpha=0.5, label='Severe Hyperglycemia (>250)')
            
            ax.set_title(f'{algorithm.replace("_", " ").title()} Algorithm', fontweight='bold')
            ax.set_ylabel('Blood Glucose (mg/dL)')
            ax.set_ylim(50, 350)
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend(loc='upper right', fontsize=8)
        
        axes[-1].set_xlabel('Time (hours)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Glucose traces plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_paired_comparison(
        self,
        metrics_df: pd.DataFrame,
        reference_algorithm: str = 'temp_basal',
        comparison_algorithm: str = 'autobolus',
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 8)
    ) -> None:
        """
        Create paired comparison plots (scatter plots with identity line).
        
        Args:
            metrics_df: DataFrame with metrics for each simulation
            reference_algorithm: Reference algorithm name
            comparison_algorithm: Comparison algorithm name
            metrics: List of metrics to plot
            save_path: Path to save the plot
            figsize: Figure size tuple
        """
        if metrics is None:
            metrics = ['time_in_range_70_180', 'time_below_70', 'mean_glucose', 'cv_glucose']
        
        # Filter for paired simulations
        ref_data = metrics_df[metrics_df['alg'] == reference_algorithm].copy()
        comp_data = metrics_df[metrics_df['alg'] == comparison_algorithm].copy()
        
        if len(ref_data) == 0 or len(comp_data) == 0:
            logger.warning(f"No data found for algorithms: {reference_algorithm}, {comparison_algorithm}")
            return
        
        # Create matching key for pairing
        def create_match_key(df):
            key_cols = ['patient', 'ibg', 'meal']  # Adjust based on your ID format
            available_cols = [col for col in key_cols if col in df.columns]
            if available_cols:
                return df[available_cols].apply(lambda x: '_'.join(map(str, x)), axis=1)
            else:
                # Fallback: use index
                return df.index
        
        ref_data['match_key'] = create_match_key(ref_data)
        comp_data['match_key'] = create_match_key(comp_data)
        
        # Merge for paired comparison
        paired_data = pd.merge(ref_data, comp_data, on='match_key', suffixes=('_ref', '_comp'))
        
        if len(paired_data) == 0:
            logger.warning("No paired simulations found")
            return
        
        # Create subplots
        n_metrics = len(metrics)
        n_cols = 2
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle(f'Paired Comparison: {reference_algorithm} vs {comparison_algorithm}', 
                    fontsize=14, fontweight='bold')
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col]
            
            ref_col = f'{metric}_ref'
            comp_col = f'{metric}_comp'
            
            if ref_col not in paired_data.columns or comp_col not in paired_data.columns:
                ax.text(0.5, 0.5, f'Metric {metric}\nnot available', 
                       ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Scatter plot
            ax.scatter(paired_data[ref_col], paired_data[comp_col], 
                      alpha=0.6, s=50, color=self.colors.get(comparison_algorithm, 'orange'))
            
            # Identity line
            min_val = min(paired_data[ref_col].min(), paired_data[comp_col].min())
            max_val = max(paired_data[ref_col].max(), paired_data[comp_col].max())
            ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Identity')
            
            # Formatting
            ax.set_xlabel(f'{reference_algorithm} {self._format_metric_name(metric)}')
            ax.set_ylabel(f'{comparison_algorithm} {self._format_metric_name(metric)}')
            ax.set_title(self._format_metric_name(metric), fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Add correlation
            correlation = paired_data[ref_col].corr(paired_data[comp_col])
            ax.text(0.05, 0.95, f'r = {correlation:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Paired comparison plot saved to {save_path}")
        else:
            plt.show()
    
    def plot_metrics_heatmap(
        self,
        metrics_df: pd.DataFrame,
        group_by: str = 'alg',
        metrics: Optional[List[str]] = None,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 6)
    ) -> None:
        """
        Create a heatmap of metrics by algorithm or other grouping.
        
        Args:
            metrics_df: DataFrame with metrics for each simulation
            group_by: Column to group by (e.g., 'alg', 'patient')
            metrics: List of metrics to include
            save_path: Path to save the plot
            figsize: Figure size tuple
        """
        if metrics is None:
            metrics = [
                'time_in_range_70_180', 'time_below_70', 'time_below_54',
                'mean_glucose', 'cv_glucose', 'cumulative_insulin'
            ]
        
        # Filter available metrics
        available_metrics = [m for m in metrics if m in metrics_df.columns]
        
        if not available_metrics or group_by not in metrics_df.columns:
            logger.warning(f"Invalid metrics or grouping column: {group_by}")
            return
        
        # Calculate mean values by group
        heatmap_data = metrics_df.groupby(group_by)[available_metrics].mean()
        
        # Normalize for better visualization (z-score)
        heatmap_data_norm = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()
        
        # Create heatmap
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Raw values
        sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlBu_r', 
                   ax=ax1, cbar_kws={'label': 'Raw Value'})
        ax1.set_title('Raw Metric Values', fontweight='bold')
        ax1.set_ylabel(group_by.title())
        
        # Normalized values
        sns.heatmap(heatmap_data_norm, annot=True, fmt='.2f', cmap='RdBu_r', 
                   center=0, ax=ax2, cbar_kws={'label': 'Z-Score'})
        ax2.set_title('Normalized Metric Values', fontweight='bold')
        ax2.set_ylabel('')
        
        # Format metric names
        for ax in [ax1, ax2]:
            ax.set_xticklabels([self._format_metric_name(m) for m in available_metrics], 
                              rotation=45, ha='right')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(self.output_dir / save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Metrics heatmap saved to {save_path}")
        else:
            plt.show()
    
    def _format_metric_name(self, metric: str) -> str:
        """Format metric name for display."""
        
        format_map = {
            'time_in_range_70_180': 'Time in Range\n(70-180 mg/dL)',
            'time_below_70': 'Time Below 70\n(mg/dL)',
            'time_below_54': 'Time Below 54\n(mg/dL)',
            'time_above_180': 'Time Above 180\n(mg/dL)',
            'time_above_250': 'Time Above 250\n(mg/dL)',
            'mean_glucose': 'Mean Glucose\n(mg/dL)',
            'cv_glucose': 'Glucose CV\n(%)',
            'cumulative_insulin': 'Total Insulin\n(U)',
            'lbgi': 'LBGI',
            'hbgi': 'HBGI',
            'bgri': 'BGRI'
        }
        
        return format_map.get(metric, metric.replace('_', ' ').title())
    
    def _get_metric_units(self, metric: str) -> str:
        """Get units for metric."""
        
        units_map = {
            'time_in_range_70_180': '%',
            'time_below_70': '%',
            'time_below_54': '%',
            'time_above_180': '%',
            'time_above_250': '%',
            'mean_glucose': 'mg/dL',
            'cv_glucose': '%',
            'cumulative_insulin': 'U',
            'lbgi': 'index',
            'hbgi': 'index',
            'bgri': 'index'
        }
        
        return units_map.get(metric, '')
    
    def _add_statistical_annotations(
        self,
        metrics_df: pd.DataFrame,
        metric: str,
        ax: plt.Axes
    ) -> None:
        """Add statistical significance annotations to plot."""
        
        try:
            from scipy import stats
            
            algorithms = metrics_df['alg'].unique()
            if len(algorithms) == 2:
                alg1, alg2 = algorithms
                data1 = metrics_df[metrics_df['alg'] == alg1][metric].dropna()
                data2 = metrics_df[metrics_df['alg'] == alg2][metric].dropna()
                
                if len(data1) > 0 and len(data2) > 0:
                    # Perform t-test
                    statistic, p_value = stats.ttest_ind(data1, data2)
                    
                    # Add annotation
                    if p_value < 0.001:
                        sig_text = '***'
                    elif p_value < 0.01:
                        sig_text = '**'
                    elif p_value < 0.05:
                        sig_text = '*'
                    else:
                        sig_text = 'ns'
                    
                    # Add text annotation
                    ax.text(0.5, 0.95, f'p = {p_value:.3f} {sig_text}', 
                           transform=ax.transAxes, ha='center',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                           fontsize=8)
        
        except ImportError:
            logger.debug("scipy not available for statistical annotations")
        except Exception as e:
            logger.debug(f"Error adding statistical annotations: {e}")
