"""
Insulin Algorithm Testing Framework

A comprehensive framework for comparing insulin delivery algorithms,
specifically designed for temp basal vs autobolus comparisons.

This framework provides:
- Core simulation integration with Tidepool simulator
- Scenario generation for comprehensive testing
- Metrics calculation and statistical analysis
- Visualization tools for results
- Regulatory compliance support
"""

__version__ = "1.0.0"
__author__ = "Tidepool Data Science Team"

# Use lazy imports to avoid dependency issues at module load time
def _get_experiment_runner():
    from .experiments.main_experiment import ExperimentRunner
    return ExperimentRunner

def _get_simulation_runner():
    from .core.scenario_runner import ScenarioRunner
    return ScenarioRunner

def _get_scenario_generator():
    from .core.scenario_generator import ScenarioGenerator
    return ScenarioGenerator


# Make classes available at module level
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
setattr(module, 'SimulationRunner', _LazyLoader(_get_simulation_runner))
setattr(module, 'ScenarioGenerator', _LazyLoader(_get_scenario_generator))
__all__ = [
    'ExperimentRunner',
    'SimulationRunner', 
    'ScenarioGenerator'
]
