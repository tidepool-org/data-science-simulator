"""
Core components for the insulin algorithm testing framework.
"""

from .simulation_runner import SimulationRunner
from .scenario_generator import ScenarioGenerator
from .data_loader import DataLoader

# Import functional metrics components
from .metrics_calculator import (
    calculate_point_metrics,
    calculate_metrics_batch,
    metrics_to_dataframe,
    PointMetrics,
    MetricsResult  # Alias for backward compatibility
)

__all__ = [
    'SimulationRunner',
    'ScenarioGenerator', 
    'DataLoader',
    # Functional metrics components
    'calculate_point_metrics',
    'calculate_metrics_batch', 
    'metrics_to_dataframe',
    'PointMetrics',
    'MetricsResult'  # Alias for backward compatibility
]
