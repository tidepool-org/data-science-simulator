"""
Utilities module for insulin algorithm testing framework.

This module provides utility functions and helper classes for common tasks
in insulin algorithm testing and analysis.
"""

__version__ = "1.0.0"
__author__ = "Tidepool Data Science Team"

from .data_utils import DataProcessor, ResultsAggregator, format_duration

__all__ = [
    'DataProcessor',
    'ResultsAggregator',
    'format_duration'
]
