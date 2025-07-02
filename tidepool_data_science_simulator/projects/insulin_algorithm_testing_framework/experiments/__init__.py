"""
Experiments module for insulin algorithm testing framework.

This module contains high-level experiment runners and orchestrators
that combine the core framework components to run complete experiments.
"""

# Use lazy import to avoid dependency issues at module load time
def _get_experiment_runner():
    from .main_experiment import ExperimentRunner
    return ExperimentRunner

# Make ExperimentRunner available at module level with lazy loading
import sys
module = sys.modules[__name__]

class _LazyLoader:
    def __init__(self, loader_func):
        self._loader_func = loader_func
        self._loaded_class = None
    
    def __call__(self, *args, **kwargs):
        if self._loaded_class is None:
            self._loaded_class = self._loader_func()
        return self._loaded_class(*args, **kwargs)

# Set up lazy loading
setattr(module, 'ExperimentRunner', _LazyLoader(_get_experiment_runner))

__all__ = ['ExperimentRunner']
