"""
Core components for the insulin algorithm testing framework.
"""

from .data_loader import DataLoader

# Import functional simulation building components
from .simulation_builder import (
    generate_simulations,
    count_simulations,
    configure_initial_glucose,
    configure_algorithm_settings,
    configure_settings_mismatches,
    create_meal_timeline,
    generate_simulation_id
)

# Import functional metrics components
from .metrics_calculator import (
    calculate_point_metrics,
    calculate_metrics_batch,
    metrics_to_dataframe,
    PointMetrics,
    MetricsResult  # Alias for backward compatibility
)

__all__ = [
    # Data loading
    'DataLoader',
    # Simulation building
    'build_simulation',
    'build_simulations',
    'generate_simulations',
    'count_simulations',
    'configure_initial_glucose',
    'configure_algorithm_settings',
    'configure_settings_mismatches',
    'create_meal_timeline',
    'generate_simulation_id',
    # Metrics calculation
    'calculate_point_metrics',
    'calculate_metrics_batch', 
    'metrics_to_dataframe',
    'PointMetrics',
    'MetricsResult'  # Alias for backward compatibility
]
