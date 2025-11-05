"""
Configuration management for experiments.
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class AlgorithmConfig:
    """Configuration for a specific algorithm."""
    enabled: bool = True
    controller_id: str = "swift"
    include_positive_velocity_and_RC: bool = True
    use_mid_absorption_isf: bool = True
    max_basal_multiplier: float = 3.5
    partial_application_factors: List[float] = field(default_factory=lambda: [0.2, 0.3, 0.4, 0.5, 0.6])
    gradual_transition_thresholds: List[float] = field(default_factory=lambda: [20.0, 30.0, 40.0])  # in mg/dL per min
    minimum_autobolus: float = 0.1
    maximum_autobolus: float = 100.0


@dataclass
class ScenarioConfig:
    """Configuration for test scenarios."""
    initial_bg_range: List[int] = field(default_factory=lambda: [70, 180])
    initial_bg_step: int = 10
    unannounced_meals: List[int] = field(default_factory=lambda: [20, 40, 60])
    meal_timing: int = 0
    absorption_time: int = 240
    settings_multipliers: List[float] = field(default_factory=lambda: [0.5, 0.75, 1.0, 1.25, 1.5])
    settings_apply_to: List[str] = field(default_factory=lambda: ["isf", "cir", "basal"])
    patient_source: str = "icgm_patients"
    num_patients: Optional[int] = None


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    duration_hours: int = 8
    start_index: int = 137
    time_step_minutes: int = 5
    safety_threshold_hours: int = 3


@dataclass
class ProcessingConfig:
    """Configuration for processing parameters."""
    parallel_processes: int = os.cpu_count()
    batch_size: int = os.cpu_count()
    save_individual_results: bool = True
    save_summary_only: bool = False


@dataclass
class AnalysisConfig:
    """Configuration for statistical analysis."""
    statistical_tests: List[str] = field(default_factory=lambda: ["paired_t_test", "wilcoxon_signed_rank"])
    multiple_comparisons_method: str = "bonferroni"
    alpha: float = 0.05
    non_inferiority_enabled: bool = True
    safety_metrics: List[str] = field(default_factory=lambda: ["time_below_70", "time_below_54", "lbgi"])
    non_inferiority_margins: Dict[str, float] = field(default_factory=lambda: {
        "time_below_70": 1.0,
        "time_below_54": 0.5,
        "lbgi": 0.5
    })
    mixed_effects_enabled: bool = True
    random_effects: List[str] = field(default_factory=lambda: ["patient_id"])
    fixed_effects: List[str] = field(default_factory=lambda: ["algorithm", "initial_bg", "settings_mismatch"])


class ExperimentConfig:
    """
    Main configuration class for experiments.
    
    Handles loading, validation, and access to configuration parameters.
    """
    
    def __init__(self, config_path: Optional[str] = None, config_dict: Optional[Dict[str, Any]] = None):
        """
        Initialize configuration.
        
        Args:
            config_path: Path to YAML configuration file
            config_dict: Dictionary with configuration parameters
        """
        self.config_path = config_path
        self._config = {}
        
        if config_path:
            self.load_from_file(config_path)
        elif config_dict:
            self._config = config_dict
        else:
            self.load_default_config()
            
        self._validate_config()
        self._setup_logging()
    
    def load_from_file(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f)
            
        logger.info(f"Loaded configuration from {config_path}")
    
    def load_default_config(self) -> None:
        """Load default configuration."""
        default_config_path = Path(__file__).parent / "default_configs.yaml"
        self.load_from_file(default_config_path)
        logger.info("Loaded default configuration")
    
    def save_to_file(self, output_path: str) -> None:
        """Save current configuration to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(self._config, f, default_flow_style=False, indent=2)
            
        logger.info(f"Saved configuration to {output_path}")
    
    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        required_sections = ['experiment', 'algorithms', 'scenarios', 'simulation']
        
        for section in required_sections:
            if section not in self._config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate algorithm configurations
        algorithms = self._config.get('algorithms', {})
        if not any(alg.get('enabled', False) for alg in algorithms.values()):
            raise ValueError("At least one algorithm must be enabled")
        
        # Validate scenario parameters
        scenarios = self._config.get('scenarios', {})
        initial_bg = scenarios.get('initial_bg', {})
        if isinstance(initial_bg.get('range'), list) and len(initial_bg['range']) != 2:
            raise ValueError("initial_bg.range must be a list of [start, end]")
        
        logger.info("Configuration validation passed")
    
    def _setup_logging(self) -> None:
        """Setup logging based on configuration."""
        log_config = self._config.get('logging', {})
        
        level = getattr(logging, log_config.get('level', 'INFO').upper())
        format_str = log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Configure root logger
        logging.basicConfig(
            level=level,
            format=format_str,
            force=True
        )
        
        # Add file handler if specified
        if 'file' in log_config:
            file_handler = logging.FileHandler(log_config['file'])
            file_handler.setFormatter(logging.Formatter(format_str))
            logging.getLogger().addHandler(file_handler)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key (supports dot notation)."""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
                
        return value
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value by key (supports dot notation)."""
        keys = key.split('.')
        config = self._config
        
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
            
        config[keys[-1]] = value
    
    @property
    def experiment_name(self) -> str:
        """Get experiment name."""
        return self.get('experiment.name', 'unnamed_experiment')
    
    @property
    def output_dir(self) -> str:
        """Get output directory."""
        return self.get('experiment.output_dir', 'results')
    
    @property
    def random_seed(self) -> int:
        """Get random seed."""
        return self.get('experiment.random_seed', 42)
    
    def get_algorithm_config(self, algorithm_name: str) -> AlgorithmConfig:
        """Get configuration for specific algorithm."""
        alg_config = self.get(f'algorithms.{algorithm_name}', {})
        return AlgorithmConfig(**alg_config)
    
    def get_scenario_config(self) -> ScenarioConfig:
        """Get scenario configuration."""
        scenario_config = self.get('scenarios', {})
        
        # Handle nested structure
        config_dict = {
            'initial_bg_range': scenario_config.get('initial_bg', {}).get('range', [70, 180]),
            'initial_bg_step': scenario_config.get('initial_bg', {}).get('step', 10),
            'unannounced_meals': scenario_config.get('meal_scenarios', {}).get('unannounced_meals', [20, 40, 60]),
            'meal_timing': scenario_config.get('meal_scenarios', {}).get('meal_timing', 0),
            'absorption_time': scenario_config.get('meal_scenarios', {}).get('absorption_time', 240),
            'settings_multipliers': scenario_config.get('settings_mismatches', {}).get('multipliers', [0.5, 0.75, 1.0, 1.25, 1.5]),
            'settings_apply_to': scenario_config.get('settings_mismatches', {}).get('apply_to', ["isf", "cir", "basal"]),
            'patient_source': scenario_config.get('patient_parameters', {}).get('source', 'icgm_patients'),
            'num_patients': scenario_config.get('patient_parameters', {}).get('num_patients', None)
        }
        
        return ScenarioConfig(**config_dict)
    
    def get_simulation_config(self) -> SimulationConfig:
        """Get simulation configuration."""
        sim_config = self.get('simulation', {})
        return SimulationConfig(**sim_config)
    
    def get_processing_config(self) -> ProcessingConfig:
        """Get processing configuration."""
        proc_config = self.get('processing', {})
        return ProcessingConfig(**proc_config)
    
    def get_analysis_config(self) -> AnalysisConfig:
        """Get analysis configuration."""
        analysis_config = self.get('analysis', {})
        
        # Handle nested structure
        config_dict = {
            'statistical_tests': analysis_config.get('statistical_tests', ["paired_t_test", "wilcoxon_signed_rank"]),
            'multiple_comparisons_method': analysis_config.get('multiple_comparisons', {}).get('method', 'bonferroni'),
            'alpha': analysis_config.get('multiple_comparisons', {}).get('alpha', 0.05),
            'non_inferiority_enabled': analysis_config.get('non_inferiority', {}).get('enabled', True),
            'safety_metrics': analysis_config.get('non_inferiority', {}).get('safety_metrics', ["time_below_70", "time_below_54", "lbgi"]),
            'non_inferiority_margins': analysis_config.get('non_inferiority', {}).get('margins', {
                "time_below_70": 1.0,
                "time_below_54": 0.5,
                "lbgi": 0.5
            }),
            'mixed_effects_enabled': analysis_config.get('mixed_effects', {}).get('enabled', True),
            'random_effects': analysis_config.get('mixed_effects', {}).get('random_effects', ["patient_id"]),
            'fixed_effects': analysis_config.get('mixed_effects', {}).get('fixed_effects', ["algorithm", "initial_bg", "settings_mismatch"])
        }
        
        return AnalysisConfig(**config_dict)
    
    def get_enabled_algorithms(self) -> List[str]:
        """Get list of enabled algorithms."""
        algorithms = self.get('algorithms', {})
        return [name for name, config in algorithms.items() if config.get('enabled', False)]
    
    def get_primary_metrics(self) -> List[str]:
        """Get list of primary metrics."""
        return self.get('metrics.primary', [
            'time_in_range_70_180',
            'time_below_70',
            'time_below_54',
            'lbgi',
            'mean_glucose',
            'cv_glucose',
            'cumulative_insulin'
        ])
    
    def get_secondary_metrics(self) -> List[str]:
        """Get list of secondary metrics."""
        return self.get('metrics.secondary', [
            'time_above_180',
            'time_above_250',
            'hbgi',
            'bgri'
        ])
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self._config.copy()
    
    def __repr__(self) -> str:
        """String representation of configuration."""
        return f"ExperimentConfig(name='{self.experiment_name}', algorithms={self.get_enabled_algorithms()})"
