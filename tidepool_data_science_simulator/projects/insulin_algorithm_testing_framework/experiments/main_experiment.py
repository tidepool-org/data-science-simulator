"""
Main experiment runner for insulin algorithm testing framework.

This module provides the ExperimentRunner class which orchestrates
complete experimental workflows combining all framework components.
"""

import logging
import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np

from ..config.experiment_config import ExperimentConfig
from ..core.data_loader import DataLoader
from ..core.scenario_generator import ScenarioGenerator
from ..core.scenario_runner import ScenarioRunner
from ..core.metrics_calculator import (
    calculate_metrics_batch,
    create_metrics_dataframe
)
from ..analysis.statistical_analyzer import StatisticalAnalyzer

logger = logging.getLogger(__name__)


class ExperimentResults:
    """Container for experiment results."""
    
    def __init__(self):
        self.metrics_df: Optional[pd.DataFrame] = None
        self.full_results: Optional[Dict[str, Any]] = None
        self.summary_results: Optional[Dict[str, Any]] = None
        self.statistical_results: Optional[Dict[str, Any]] = None
        self.experiment_metadata: Dict[str, Any] = {}
        self.execution_time: float = 0.0
        
    def save_to_directory(self, output_dir: str) -> None:
        """Save all results to specified directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics DataFrame
        if self.metrics_df is not None:
            self.metrics_df.to_csv(output_path / 'experiment_metrics.csv', index=False)
            
        # Save statistical results
        if self.statistical_results is not None:
            with open(output_path / 'statistical_analysis.json', 'w') as f:
                json.dump(self._serialize_for_json(self.statistical_results), f, indent=2)
                
        # Save experiment metadata
        with open(output_path / 'experiment_metadata.json', 'w') as f:
            json.dump(self.experiment_metadata, f, indent=2)
            
        # Save summary results if available
        if self.summary_results is not None:
            with open(output_path / 'summary_results.json', 'w') as f:
                json.dump(self._serialize_for_json(self.summary_results), f, indent=2)
                
        logger.info(f"Results saved to {output_path}")
    
    def _serialize_for_json(self, obj: Any) -> Any:
        """Convert numpy types to native Python types for JSON serialization."""
        if isinstance(obj, dict):
            return {k: self._serialize_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._serialize_for_json(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy scalar
            return obj.item()
        elif hasattr(obj, 'tolist'):  # numpy array
            return obj.tolist()
        else:
            return obj


class ExperimentRunner:
    """
    High-level experiment runner that orchestrates the complete workflow.
    
    This class provides a simple interface for running insulin algorithm
    comparison experiments, handling all the complexity of coordinating
    the various framework components.
    """
    
    def __init__(self, config: Optional[ExperimentConfig] = None):
        """
        Initialize the experiment runner.
        
        Args:
            config: Experiment configuration. If None, uses default config.
        """
        self.config = config or ExperimentConfig()
        self.results = ExperimentResults()
        
        # Initialize framework components
        self.data_loader = DataLoader(self.config)
        self.scenario_generator = ScenarioGenerator(self.config)
        self.simulation_runner = ScenarioRunner(self.config)
        self.metrics_calculator = MetricsCalculator(self.config)
        self.statistical_analyzer = StatisticalAnalyzer(self.config)
        
        logger.info(f"Initialized ExperimentRunner: {self.config}")
    
    def run_basic_comparison(
        self,
        max_patients: Optional[int] = None,
        algorithms: Optional[List[str]] = None,
        save_results: bool = True
    ) -> ExperimentResults:
        """
        Run a basic algorithm comparison experiment.
        
        Args:
            max_patients: Maximum number of patients to include
            algorithms: List of algorithms to compare. If None, uses all enabled algorithms
            save_results: Whether to save results to disk
            
        Returns:
            ExperimentResults object containing all results
        """
        start_time = time.time()
        
        logger.info("Starting basic comparison experiment")
        
        try:
            # 1. Load patient data
            logger.info("Loading patient configurations...")
            patient_configs = self.data_loader.load_patient_configurations()
            if max_patients:
                patient_configs = patient_configs[:max_patients]
            
            logger.info(f"Loaded {len(patient_configs)} patient configurations")
            
            # 2. Generate scenarios
            logger.info("Generating scenarios...")
            if algorithms:
                scenarios = []
                for algorithm in algorithms:
                    alg_scenarios = list(self.scenario_generator.generate_scenarios_for_algorithm(
                        patient_configs, algorithm
                    ))
                    scenarios.extend(alg_scenarios)
            else:
                scenarios = list(self.scenario_generator.generate_all_scenarios(patient_configs))
            
            logger.info(f"Generated {len(scenarios)} scenarios")
            
            # 3. Run simulations
            logger.info("Running simulations...")
            full_results, summary_results = self._run_simulations(scenarios)
            
            # 4. Calculate metrics
            logger.info("Calculating metrics...")
            metrics_dict = self.metrics_calculator.calculate_metrics_batch(full_results)
            metrics_df = self.metrics_calculator.create_metrics_dataframe(metrics_dict)
            
            # 5. Statistical analysis
            logger.info("Performing statistical analysis...")
            statistical_results = self._perform_statistical_analysis(metrics_df)
            
            # 6. Store results
            self.results.metrics_df = metrics_df
            self.results.full_results = full_results
            self.results.summary_results = summary_results
            self.results.statistical_results = statistical_results
            self.results.execution_time = time.time() - start_time
            
            # 7. Save results if requested
            if save_results:
                self.results.save_to_directory(self.config.output_dir)
            
            logger.info(f"Basic comparison completed in {self.results.execution_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Experiment failed: {e}")
            raise
    
    def run_parameter_sweep(
        self,
        parameter_name: str,
        parameter_values: List[float],
        algorithm: str = 'autobolus',
        max_patients: Optional[int] = None,
        save_results: bool = True
    ) -> ExperimentResults:
        """
        Run a parameter sweep experiment.
        
        Args:
            parameter_name: Name of parameter to sweep (e.g., 'partial_application_factor')
            parameter_values: List of parameter values to test
            algorithm: Algorithm to test (default: 'autobolus')
            max_patients: Maximum number of patients to include
            save_results: Whether to save results to disk
            
        Returns:
            ExperimentResults object containing all results
        """
        start_time = time.time()
        
        logger.info(f"Starting parameter sweep: {parameter_name} = {parameter_values}")
        
        try:
            # Configure for parameter sweep
            if parameter_name == 'partial_application_factor':
                self.config.set(f'algorithms.{algorithm}.partial_application_factors', parameter_values)
            else:
                raise ValueError(f"Parameter sweep for '{parameter_name}' not implemented")
            
            # Load patient data
            logger.info("Loading patient configurations...")
            patient_configs = self.data_loader.load_patient_configurations()
            if max_patients:
                patient_configs = patient_configs[:max_patients]
            
            # Generate scenarios for the specified algorithm
            logger.info("Generating scenarios...")
            scenarios = list(self.scenario_generator.generate_scenarios_for_algorithm(
                patient_configs, algorithm
            ))
            
            logger.info(f"Generated {len(scenarios)} scenarios")
            
            # Run simulations
            logger.info("Running simulations...")
            full_results, summary_results = self._run_simulations(scenarios)
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics_dict = self.metrics_calculator.calculate_metrics_batch(full_results)
            metrics_df = self.metrics_calculator.create_metrics_dataframe(metrics_dict)
            
            # Analyze parameter sweep results
            logger.info("Analyzing parameter sweep results...")
            sweep_analysis = self._analyze_parameter_sweep(metrics_df, parameter_name)
            
            # Store results
            self.results.metrics_df = metrics_df
            self.results.full_results = full_results
            self.results.summary_results = summary_results
            self.results.statistical_results = sweep_analysis
            self.results.execution_time = time.time() - start_time
            
            # Save results if requested
            if save_results:
                self.results.save_to_directory(self.config.output_dir)
            
            logger.info(f"Parameter sweep completed in {self.results.execution_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Parameter sweep failed: {e}")
            raise
    
    def run_custom_experiment(
        self,
        scenarios: List[Dict[str, Any]],
        save_results: bool = True
    ) -> ExperimentResults:
        """
        Run a custom experiment with user-provided scenarios.
        
        Args:
            scenarios: List of scenario dictionaries
            save_results: Whether to save results to disk
            
        Returns:
            ExperimentResults object containing all results
        """
        start_time = time.time()
        
        logger.info(f"Starting custom experiment with {len(scenarios)} scenarios")
        
        try:
            # Run simulations
            logger.info("Running simulations...")
            full_results, summary_results = self._run_simulations(scenarios)
            
            # Calculate metrics
            logger.info("Calculating metrics...")
            metrics_dict = self.metrics_calculator.calculate_metrics_batch(full_results)
            metrics_df = self.metrics_calculator.create_metrics_dataframe(metrics_dict)
            
            # Statistical analysis if multiple algorithms
            statistical_results = None
            algorithms = metrics_df['alg'].unique() if 'alg' in metrics_df.columns else []
            if len(algorithms) > 1:
                logger.info("Performing statistical analysis...")
                statistical_results = self._perform_statistical_analysis(metrics_df)
            
            # Store results
            self.results.metrics_df = metrics_df
            self.results.full_results = full_results
            self.results.summary_results = summary_results
            self.results.statistical_results = statistical_results
            self.results.execution_time = time.time() - start_time
            
            # Save results if requested
            if save_results:
                self.results.save_to_directory(self.config.output_dir)
            
            logger.info(f"Custom experiment completed in {self.results.execution_time:.2f} seconds")
            
            return self.results
            
        except Exception as e:
            logger.error(f"Custom experiment failed: {e}")
            raise
    
    def _run_simulations(self, scenarios: List[Dict[str, Any]]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Run batch simulations for given scenarios."""
        # Create simulation objects
        simulations = {}
        for scenario in scenarios:
            simulation = self.simulation_runner.create_simulation_from_scenario(scenario)
            simulations[simulation.sim_id] = simulation
        
        # Run batch simulations
        full_results, summary_results = self.simulation_runner.run_batch_simulations(simulations)
        
        logger.info(f"Completed {len(full_results)} simulations")
        
        return full_results, summary_results
    
    def _perform_statistical_analysis(self, metrics_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform statistical analysis on metrics."""
        algorithms = metrics_df['alg'].unique()
        
        if len(algorithms) < 2:
            logger.warning("Statistical analysis requires at least 2 algorithms")
            return {}
        
        # Determine reference algorithm (prefer temp_basal if available)
        reference_algorithm = 'temp_basal' if 'temp_basal' in algorithms else algorithms[0]
        comparison_algorithms = [alg for alg in algorithms if alg != reference_algorithm]
        
        # Perform comparison
        comparison_results = self.statistical_analyzer.compare_algorithms(
            metrics_df,
            reference_algorithm=reference_algorithm,
            comparison_algorithms=comparison_algorithms
        )
        
        return comparison_results
    
    def _analyze_parameter_sweep(self, metrics_df: pd.DataFrame, parameter_name: str) -> Dict[str, Any]:
        """Analyze parameter sweep results."""
        if parameter_name == 'partial_application_factor':
            groupby_col = 'paf'
        else:
            groupby_col = parameter_name
        
        if groupby_col not in metrics_df.columns:
            logger.warning(f"Parameter column '{groupby_col}' not found in metrics")
            return {}
        
        # Group by parameter value and calculate summary statistics
        primary_metrics = self.config.get_primary_metrics()
        available_metrics = [m for m in primary_metrics if m in metrics_df.columns]
        
        sweep_results = metrics_df.groupby(groupby_col)[available_metrics].agg(['mean', 'std']).round(4)
        
        # Find optimal values
        optimal_values = {}
        for metric in available_metrics:
            if 'time_below' in metric or 'lbgi' in metric or 'hbgi' in metric:
                # Lower is better for safety metrics
                optimal_values[f'optimal_{metric}'] = sweep_results[(metric, 'mean')].idxmin()
            elif 'time_in_range' in metric:
                # Higher is better for time in range
                optimal_values[f'optimal_{metric}'] = sweep_results[(metric, 'mean')].idxmax()
            elif 'mean_glucose' in metric:
                # Closest to target (140 mg/dL) is better
                target_glucose = 140
                optimal_values[f'optimal_{metric}'] = sweep_results[(metric, 'mean')].apply(
                    lambda x: abs(x - target_glucose)
                ).idxmin()
        
        # Calculate correlations with parameter
        correlations = {}
        parameter_values = metrics_df[groupby_col].values
        for metric in available_metrics:
            metric_values = metrics_df[metric].values
            correlation = np.corrcoef(parameter_values, metric_values)[0, 1]
            correlations[metric] = correlation
        
        return {
            'parameter_name': parameter_name,
            'summary_statistics': sweep_results.to_dict(),
            'optimal_values': optimal_values,
            'correlations': correlations
        }
    
    def get_summary_report(self) -> str:
        """Generate a summary report of the experiment results."""
        if self.results.metrics_df is None:
            return "No experiment results available."
        
        report = []
        report.append("=" * 60)
        report.append("EXPERIMENT SUMMARY REPORT")
        report.append("=" * 60)
        
        # Basic statistics
        metrics_df = self.results.metrics_df
        report.append(f"\nTotal simulations: {len(metrics_df)}")
        report.append(f"Execution time: {self.results.execution_time:.2f} seconds")
        
        # Algorithm summary
        if 'alg' in metrics_df.columns:
            algorithms = metrics_df['alg'].unique()
            report.append(f"Algorithms tested: {', '.join(algorithms)}")
            
            # Key metrics by algorithm
            key_metrics = ['time_in_range_70_180', 'time_below_70', 'mean_glucose']
            available_metrics = [m for m in key_metrics if m in metrics_df.columns]
            
            if available_metrics:
                report.append("\nKey Metrics by Algorithm:")
                summary_stats = metrics_df.groupby('alg')[available_metrics].agg(['mean', 'std'])
                report.append(str(summary_stats.round(2)))
        
        # Statistical results
        if self.results.statistical_results and 'statistical_tests' in self.results.statistical_results:
            report.append("\nStatistical Test Results:")
            for metric, tests in self.results.statistical_results['statistical_tests'].items():
                report.append(f"\n{metric}:")
                for test_name, result in tests.items():
                    p_value = result.get('p_value', 'N/A')
                    significant = result.get('significant', False)
                    report.append(f"  {test_name}: p={p_value:.4f}, significant={significant}")
        
        return "\n".join(report)
    
    def create_visualizations(self, output_dir: Optional[str] = None) -> None:
        """Create standard visualizations for the experiment results."""
        if self.results.metrics_df is None:
            logger.warning("No results available for visualization")
            return
        
        try:
            from ..visualization.comparison_plots import ComparisonPlotter
            
            output_path = output_dir or self.config.output_dir
            plotter = ComparisonPlotter(self.config)
            
            # Algorithm comparison plots
            if 'alg' in self.results.metrics_df.columns:
                algorithms = self.results.metrics_df['alg'].unique()
                if len(algorithms) > 1:
                    plotter.plot_algorithm_comparison(
                        self.results.metrics_df,
                        save_path=f"{output_path}/algorithm_comparison.png"
                    )
            
            # Glucose traces sample
            if self.results.full_results:
                plotter.plot_glucose_traces_sample(
                    self.results.full_results,
                    n_samples=6,
                    save_path=f"{output_path}/glucose_traces_sample.png"
                )
            
            logger.info(f"Visualizations saved to {output_path}")
            
        except ImportError:
            logger.warning("Visualization module not available")
        except Exception as e:
            logger.error(f"Visualization failed: {e}")
