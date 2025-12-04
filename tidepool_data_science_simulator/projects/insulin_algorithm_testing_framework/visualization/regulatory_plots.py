"""
Regulatory-compliant visualization for FDA 510k submissions.

This module provides publication-quality plots for iCGM sensitivity analysis,
risk scoring, and algorithm comparisons following FDA AI Letter guidelines.
"""

import logging
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

logger = logging.getLogger(__name__)


# ============================================================================
# Weighted KDE Comparison Plots
# ============================================================================

def plot_weighted_kde_comparison(
    data_1: np.ndarray,
    data_2: np.ndarray,
    weights: np.ndarray,
    title: str = "Algorithm Comparison",
    ylabel: str = "Value",
    label_1: str = "Dataset 1",
    label_2: str = "Dataset 2",
    color_1: str = "#627cff",
    color_2: str = "#271b45",
    bw_method: float = 0.2,
    violin_width: float = 0.4,
    box_width: float = 0.05,
    figsize: Tuple[int, int] = (6, 6),
    font_size: int = 12
) -> Tuple[Figure, Axes]:
    """
    Create weighted KDE comparison plot with boxplots.
    
    This creates a split violin plot with KDE distributions and overlaid
    boxplots, weighted by population distribution (e.g., IBG histogram).
    
    Args:
        data_1: First dataset values
        data_2: Second dataset values
        weights: Population weights for each data point
        title: Plot title
        ylabel: Y-axis label
        label_1: Label for first dataset
        label_2: Label for second dataset
        color_1: Color for first dataset
        color_2: Color for second dataset
        bw_method: KDE bandwidth
        violin_width: Maximum width of violin plots
        box_width: Width of boxplots
        figsize: Figure size (width, height)
        font_size: Base font size
        
    Returns:
        Tuple of (figure, axes)
        
    Example:
        >>> fig, ax = plot_weighted_kde_comparison(
        ...     tir_tempbasal, tir_autobolus, weights,
        ...     title="Time in Range Comparison",
        ...     ylabel="TIR (%)",
        ...     label_1="Temp Basal", label_2="Autobolus"
        ... )
        >>> plt.savefig('tir_comparison.png')
    """
    # Data range for KDE
    x_min = min(data_1.min(), data_2.min())
    x_max = max(data_1.max(), data_2.max())
    x_grid = np.linspace(x_min, x_max, 500)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    try:
        # Calculate KDEs
        kde_1 = gaussian_kde(data_1, weights=weights, bw_method=bw_method)
        kde_2 = gaussian_kde(data_2, weights=weights, bw_method=bw_method)
        
        density_1 = kde_1(x_grid)
        density_2 = kde_2(x_grid)
        
        # Find modes
        peaks_1, _ = find_peaks(density_1)
        modes_1 = x_grid[peaks_1]
        
        peaks_2, _ = find_peaks(density_2)
        modes_2 = x_grid[peaks_2]
        
        # Normalize KDEs
        density_1 /= density_1.max()
        density_2 /= density_2.max()
        
        # Add modes to labels
        if len(modes_1) > 0:
            label_1_with_mode = f"{label_1} (mode: {modes_1[0]:.1f})"
        else:
            label_1_with_mode = label_1
        
        if len(modes_2) > 0:
            label_2_with_mode = f"{label_2} (mode: {modes_2[0]:.1f})"
        else:
            label_2_with_mode = label_2
        
        # Plot KDE violins
        ax.fill_betweenx(
            x_grid, 
            (-density_1 * violin_width) - 0.1, 
            -0.1, 
            facecolor=color_1, 
            alpha=1, 
            label=label_1_with_mode
        )
        ax.fill_betweenx(
            x_grid, 
            0.1, 
            (density_2 * violin_width) + 0.1, 
            facecolor=color_2, 
            alpha=1, 
            label=label_2_with_mode
        )
        
    except Exception as e:
        logger.warning(f"KDE calculation failed: {e}. Skipping KDE plots.")
    
    # Add boxplots
    # Weight data by repeating based on weights
    data_1_weighted = np.repeat(data_1, (weights * 10000).astype(int))
    data_2_weighted = np.repeat(data_2, (weights * 10000).astype(int))
    
    box_data = [data_1_weighted, data_2_weighted]
    positions = [-0.05, 0.05]
    box = ax.boxplot(
        box_data, 
        vert=True, 
        positions=positions, 
        widths=box_width, 
        patch_artist=True
    )
    
    # Style boxplots
    line_width = 2.5
    for i, color in enumerate([color_1, color_2]):
        box['boxes'][i].set_facecolor('none')
        box['boxes'][i].set_edgecolor(color)
        box['boxes'][i].set_linewidth(line_width)
        
        box['medians'][i].set_color(color)
        box['medians'][i].set_linewidth(line_width)
        
        # Whiskers and caps
        for j in [2*i, 2*i+1]:
            box['whiskers'][j].set_color(color)
            box['whiskers'][j].set_linewidth(line_width)
            box['caps'][j].set_color(color)
            box['caps'][j].set_linewidth(line_width)
        
        # Fliers
        if 'fliers' in box and i < len(box['fliers']):
            box['fliers'][i].set_markeredgecolor(color)
            box['fliers'][i].set_linewidth(line_width)
    
    # Style plot
    ax.set_xlabel('Density', fontsize=font_size)
    ax.set_ylabel(ylabel, fontsize=font_size)
    ax.set_xticks([])
    ax.set_xticklabels([])
    ax.set_title(title, fontsize=font_size + 2, fontweight='bold')
    ax.legend(frameon=False, fontsize=font_size - 1)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# Risk Score Heatmaps
# ============================================================================

def plot_risk_heatmap(
    true_bg: np.ndarray,
    sensor_bg: np.ndarray,
    risk_data: np.ndarray,
    title: str = "Risk Score Heatmap",
    xlabel: str = "True Blood Glucose (mg/dL)",
    ylabel: str = "Sensor Blood Glucose (mg/dL)",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    colormap: str = 'viridis',
    show_colorbar: bool = True,
    grid_linewidth: float = 0.5,
    figsize: Tuple[int, int] = (10, 8),
    font_size: int = 12
) -> Tuple[Figure, Axes]:
    """
    Create risk score heatmap for iCGM sensitivity analysis.
    
    Args:
        true_bg: Array of true BG values (1D, lower bounds of ranges)
        sensor_bg: Array of sensor BG values (1D, lower bounds of ranges)
        risk_data: 2D array of risk values (sensor_bg x true_bg)
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        vmin: Minimum value for color scale
        vmax: Maximum value for color scale
        colormap: Matplotlib colormap name
        show_colorbar: Whether to show colorbar
        grid_linewidth: Width of grid lines
        figsize: Figure size
        font_size: Base font size
        
    Returns:
        Tuple of (figure, axes)
    """
    # Reshape data to grid
    dim = int(np.sqrt(len(true_bg)))
    dims = (dim, dim)
    
    true_grid = np.reshape(true_bg, dims)
    sensor_grid = np.reshape(sensor_bg, dims)
    risk_grid = np.reshape(risk_data, dims)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot heatmap
    pcm = ax.pcolormesh(
        true_grid, 
        sensor_grid, 
        risk_grid,
        vmin=vmin,
        vmax=vmax,
        cmap=colormap,
        edgecolors='k',
        linewidths=grid_linewidth,
        shading='auto'
    )
    
    # Invert y-axis (sensor BG high at top)
    ax.invert_yaxis()
    
    # Labels and title
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight='bold')
    ax.set_title(title, fontsize=font_size + 2, fontweight='bold', pad=15)
    
    # Colorbar
    if show_colorbar:
        cbar = plt.colorbar(pcm, ax=ax)
        cbar.ax.tick_params(labelsize=font_size - 1)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=font_size - 1)
    
    plt.tight_layout()
    
    return fig, ax


def plot_risk_heatmap_grid(
    true_bg: np.ndarray,
    sensor_bg: np.ndarray,
    risk_data_dict: Dict[str, np.ndarray],
    severity_bands: List[str],
    safety_thresholds: Optional[List[float]] = None,
    shared_z_scale: bool = True,
    colormap: str = 'viridis',
    figsize: Tuple[int, int] = (18, 12),
    font_size: int = 12
) -> Tuple[Figure, List[Axes]]:
    """
    Create grid of risk heatmaps for all severity bands.
    
    Args:
        true_bg: Array of true BG values
        sensor_bg: Array of sensor BG values
        risk_data_dict: Dict mapping severity band name to risk data array
        severity_bands: List of severity band names (in order)
        safety_thresholds: Optional list of safety thresholds to overlay
        shared_z_scale: Use same color scale across all heatmaps
        colormap: Matplotlib colormap
        figsize: Figure size
        font_size: Base font size
        
    Returns:
        Tuple of (figure, list_of_axes)
    """
    n_bands = len(severity_bands)
    n_cols = 3
    n_rows = int(np.ceil(n_bands / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten()
    
    # Determine global vmin/vmax if shared scale
    if shared_z_scale:
        all_data = np.concatenate([risk_data_dict[band] for band in severity_bands])
        vmin = np.min(all_data)
        vmax = np.max(all_data)
    else:
        vmin = None
        vmax = None
    
    # Plot each severity band
    for idx, band_name in enumerate(severity_bands):
        ax = axes[idx]
        risk_data = risk_data_dict[band_name]
        
        # Reshape for heatmap
        dim = int(np.sqrt(len(true_bg)))
        dims = (dim, dim)
        
        true_grid = np.reshape(true_bg, dims)
        sensor_grid = np.reshape(sensor_bg, dims)
        risk_grid = np.reshape(risk_data, dims)
        
        # Plot
        pcm = ax.pcolormesh(
            true_grid,
            sensor_grid,
            risk_grid,
            vmin=vmin if shared_z_scale else None,
            vmax=vmax if shared_z_scale else None,
            cmap=colormap,
            edgecolors='k',
            linewidths=0.5
        )
        
        ax.invert_yaxis()
        
        # Add safety threshold line if provided
        if safety_thresholds and idx < len(safety_thresholds):
            threshold = safety_thresholds[idx]
            # This would need the actual risk values to draw threshold line
            # For now, just note in title
            title = f"{band_name}\n(threshold: {threshold:.2e})"
        else:
            title = band_name
        
        ax.set_title(title, fontsize=font_size, fontweight='bold')
        ax.set_xlabel("True BG (mg/dL)", fontsize=font_size - 1)
        ax.set_ylabel("Sensor BG (mg/dL)", fontsize=font_size - 1)
        
        # Colorbar for each subplot
        plt.colorbar(pcm, ax=ax)
    
    # Remove extra subplots
    for idx in range(n_bands, len(axes)):
        fig.delaxes(axes[idx])
    
    fig.suptitle(
        "Risk Score Heatmaps by Severity Band",
        fontsize=font_size + 4,
        fontweight='bold',
        y=0.98
    )
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig, axes[:n_bands]


# ============================================================================
# Risk vs TIR Scatter Plots
# ============================================================================

def plot_risk_vs_tir_scatter(
    risk_scores: np.ndarray,
    tir_values: np.ndarray,
    labels: List[str],
    colors: List[str],
    markers: List[str],
    title: str = "Risk Score vs Time in Range",
    xlabel: str = "Probability of Risk Event",
    ylabel: str = "Time in Range (%)",
    safety_threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (10, 6),
    font_size: int = 12
) -> Tuple[Figure, Axes]:
    """
    Create scatter plot of risk scores vs TIR for different algorithms/settings.
    
    Args:
        risk_scores: Array of risk scores
        tir_values: Array of TIR values
        labels: List of labels for each point
        colors: List of colors for each point
        markers: List of marker styles
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        safety_threshold: Optional safety threshold to draw as vertical line
        figsize: Figure size
        font_size: Base font size
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each point
    for risk, tir, label, color, marker in zip(risk_scores, tir_values, labels, colors, markers):
        ax.scatter(
            risk, tir,
            color=color,
            marker=marker,
            s=120,
            alpha=0.8,
            edgecolors='black',
            linewidth=2,
            label=label,
            zorder=3
        )
    
    # Add safety threshold line
    if safety_threshold is not None:
        ax.axvline(
            x=safety_threshold,
            color='darkred',
            linestyle='--',
            linewidth=3,
            alpha=0.9,
            label=f'Safety Threshold ({safety_threshold:.2e})',
            zorder=2
        )
    
    # Style
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight='bold')
    ax.set_title(title, fontsize=font_size + 2, fontweight='bold')
    ax.grid(True, alpha=0.3, zorder=1)
    ax.legend(fontsize=font_size - 2, frameon=True, fancybox=True, shadow=True)
    
    # Format x-axis for scientific notation if needed
    if np.max(risk_scores) < 0.01:
        ax.ticklabel_format(style='scientific', axis='x', scilimits=(0, 0))
    
    ax.tick_params(axis='both', labelsize=font_size - 1)
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# Metric Over Time Plots
# ============================================================================

def plot_metric_over_time(
    time_windows: np.ndarray,
    mean_values_1: np.ndarray,
    std_values_1: np.ndarray,
    mean_values_2: np.ndarray,
    std_values_2: np.ndarray,
    label_1: str = "Algorithm 1",
    label_2: str = "Algorithm 2",
    metric_name: str = "Time in Range",
    ylabel: str = "TIR (%)",
    xlabel: str = "Time Window (hours)",
    color_1: str = "blue",
    color_2: str = "orange",
    figsize: Tuple[int, int] = (10, 6),
    font_size: int = 12
) -> Tuple[Figure, Axes]:
    """
    Plot metric evolution over time with mean ± std bands.
    
    Args:
        time_windows: Array of time windows (x-axis)
        mean_values_1: Mean values for algorithm 1
        std_values_1: Std values for algorithm 1
        mean_values_2: Mean values for algorithm 2
        std_values_2: Std values for algorithm 2
        label_1: Label for algorithm 1
        label_2: Label for algorithm 2
        metric_name: Name of metric being plotted
        ylabel: Y-axis label
        xlabel: X-axis label
        color_1: Color for algorithm 1
        color_2: Color for algorithm 2
        figsize: Figure size
        font_size: Base font size
        
    Returns:
        Tuple of (figure, axes)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot algorithm 1
    ax.plot(
        time_windows, mean_values_1,
        label=f"{label_1} (Mean)",
        color=color_1,
        linewidth=2
    )
    ax.fill_between(
        time_windows,
        mean_values_1 - std_values_1,
        mean_values_1 + std_values_1,
        color=color_1,
        alpha=0.2,
        label=f"{label_1} ± Std"
    )
    
    # Plot algorithm 2
    ax.plot(
        time_windows, mean_values_2,
        label=f"{label_2} (Mean)",
        color=color_2,
        linewidth=2
    )
    ax.fill_between(
        time_windows,
        mean_values_2 - std_values_2,
        mean_values_2 + std_values_2,
        color=color_2,
        alpha=0.2,
        label=f"{label_2} ± Std"
    )
    
    # Style
    ax.set_title(f"{metric_name} Over Time", fontsize=font_size + 2, fontweight='bold')
    ax.set_xlabel(xlabel, fontsize=font_size, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=font_size, fontweight='bold')
    ax.legend(fontsize=font_size - 2)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=font_size - 1)
    
    plt.tight_layout()
    
    return fig, ax


# ============================================================================
# Regulatory Report Figure Generation
# ============================================================================

def save_regulatory_figure(
    fig: Figure,
    save_path: Union[str, Path],
    dpi: int = 300,
    formats: List[str] = ['png', 'pdf']
) -> List[Path]:
    """
    Save figure in regulatory-compliant formats.
    
    Args:
        fig: Matplotlib figure
        save_path: Base path for saving (without extension)
        dpi: Resolution in dots per inch
        formats: List of formats to save ('png', 'pdf', 'svg')
        
    Returns:
        List of saved file paths
    """
    save_path = Path(save_path)
    saved_paths = []
    
    for fmt in formats:
        output_path = save_path.with_suffix(f'.{fmt}')
        fig.savefig(
            output_path,
            dpi=dpi,
            bbox_inches='tight',
            facecolor='white',
            edgecolor='none'
        )
        saved_paths.append(output_path)
        logger.info(f"Saved figure: {output_path}")
    
    return saved_paths
