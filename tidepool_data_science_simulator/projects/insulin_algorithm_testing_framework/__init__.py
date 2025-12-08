"""
Insulin Algorithm Testing Framework

A comprehensive framework for comparing insulin delivery algorithms,
specifically designed for temp basal vs autobolus comparisons.

This framework provides:
- Core simulation integration with Tidepool simulator
- Functional simulation building and scenario generation
- Metrics calculation and statistical analysis
- Visualization tools for results
- Regulatory compliance support

Example usage:
    from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework import (
        build_simulation,
        build_simulations,
        generate_simulations,
        calculate_point_metrics,
        metrics_to_dataframe
    )
    
    # Build simulations from scenario dictionaries
    simulations = build_simulations(config, scenarios)
    
    # Or generate iCGM simulations directly
    sim_generator, num_sims = generate_simulations(config, patient_configs, true_bg_range, sensor_bg_range)
"""

__version__ = "1.0.0"
__author__ = "Tidepool Data Science Team"

# Core simulation building functions
from .core.simulation_builder import (
    generate_simulations,
    count_simulations
)

# Metrics calculation functions
from .core.metrics_calculator import (
    calculate_point_metrics,
    calculate_metrics_batch,
    metrics_to_dataframe,
    PointMetrics,
    MetricsResult
)

# Data loading
from .core.data_loader import DataLoader

# Configuration
from .config.experiment_config import ExperimentConfig

__all__ = [
    # Simulation building
    'build_simulation',
    'build_simulations',
    'generate_simulations',
    'count_simulations',
    # Metrics
    'calculate_point_metrics',
    'calculate_metrics_batch',
    'metrics_to_dataframe',
    'PointMetrics',
    'MetricsResult',
    # Data loading
    'DataLoader',
    # Configuration
    'ExperimentConfig',
]
