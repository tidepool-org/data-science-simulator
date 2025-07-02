#!/usr/bin/env python3
"""
Basic comparison example for insulin algorithm testing framework.

This example demonstrates how to:
1. Load configuration
2. Generate scenarios
3. Run simulations comparing temp basal vs autobolus
4. Calculate metrics and perform statistical analysis
5. Visualize results

Usage:
    python basic_comparison.py
"""

import logging
import json
from pathlib import Path

from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import ExperimentConfig
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.data_loader import DataLoader
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.scenario_generator import ScenarioGenerator
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.simulation_runner import SimulationRunner
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import MetricsCalculator
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.analysis.statistical_analyzer import StatisticalAnalyzer
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.visualization.comparison_plots import ComparisonPlotter

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run basic comparison example."""
    
    logger.info("Starting basic insulin algorithm comparison")
    
    # 1. Load configuration
    logger.info("Loading configuration...")
    config = ExperimentConfig()  # Uses default config
    
    config.set('experiment.output_dir', '/Users/mconn/data/simulator/processed_data')
    
    # Override some settings for this example
    config.set('experiment.name', 'basic_comparison_example')
    config.set('scenarios.initial_bg.range', [100, 125])  # Smaller range for example
    config.set('scenarios.initial_bg.step', 25)
    config.set('scenarios.meal_scenarios.unannounced_meals', [30])  # Two meal sizes
    config.set('scenarios.settings_mismatches.multipliers', [1.0])  # Three settings
    config.set('algorithms.autobolus.partial_application_factors', [0.4])
    config.set('processing.parallel_processes', 14)  # Reduce for example
    
    logger.info(f"Configuration: {config}")
    
    # 2. Load patient data
    logger.info("Loading patient data...")
    data_loader = DataLoader(config)
    patient_configs = data_loader.load_patient_configs(max_patients=2)  # Small subset for example
    
    logger.info(f"Loaded {len(patient_configs)} patient configurations")
    
    # 3. Generate scenarios
    logger.info("Generating scenarios...")
    scenario_generator = ScenarioGenerator(config)
    
    # Get scenario summary
    summary = scenario_generator.get_scenario_summary(patient_configs)
    logger.info(f"Scenario summary: {summary}")
    
    # Generate all scenarios
    scenarios = scenario_generator.generate_all_scenarios(patient_configs)
    logger.info(f"Will generate scenarios using iterator (estimated: {summary['estimated_total_scenarios']})")
    
    # 4. Run simulations
    logger.info("Running simulations...")
    simulation_runner = SimulationRunner(config)
    
    simulations = {}
    sim_counter = 0
    full_results = {}
    
    for scenario in scenarios:
        simulation = simulation_runner.create_simulation_from_scenario(scenario)
        simulations[simulation.sim_id] = simulation
        sim_counter += 1
        
        if sim_counter % 14 == 0:
            # Run batch simulations
            results, _ = simulation_runner.run_batch_simulations(simulations)
            full_results = full_results | results  # Merge results
            simulations = {}  # Reset for next batch
            sim_counter = 0
        
    if simulations:
        results, _ = simulation_runner.run_batch_simulations(simulations)
        full_results = full_results | results  # Merge results
    
    logger.info(f"Completed {len(full_results)} simulations")
    
    # 5. Calculate metrics
    logger.info("Calculating metrics...")
    metrics_calculator = MetricsCalculator(config)
    
    # Calculate metrics for all results
    metrics_dict = metrics_calculator.calculate_metrics_batch(full_results)
    
    # Create metrics DataFrame
    metrics_df = metrics_calculator.create_metrics_dataframe(metrics_dict)
    
    logger.info(f"Calculated metrics for {len(metrics_dict)} simulations")
    logger.info(f"Metrics columns: {list(metrics_df.columns)}")
    
    # 6. Statistical analysis
    logger.info("Performing statistical analysis...")
    statistical_analyzer = StatisticalAnalyzer(config)
    
    # Perform paired comparison
    comparison_results = statistical_analyzer.compare_algorithms(
        metrics_df, 
        reference_algorithm='tempbasal',
        comparison_algorithms=['autobolus']
    )
    
    logger.info("Statistical analysis completed")
    
    # Print key results
    print("\n" + "="*60)
    print("BASIC COMPARISON RESULTS")
    print("="*60)
    
    # Summary statistics
    print("\nSummary Statistics by Algorithm:")
    summary_stats = metrics_df.groupby('alg')[['time_in_range_70_180', 'time_below_70', 'mean_glucose']].agg(['mean', 'std'])
    print(summary_stats)
    
    # Statistical test results
    alpha = config.get_analysis_config().alpha
    if 'statistical_tests' in comparison_results:
        print("\nStatistical Test Results:")
        for metric, test_results in comparison_results['statistical_tests'].items():
            print(f"\n{metric}:")
            for test_name, test_result in test_results.items():
                p_value = test_result["p_value"]
                significant = statistical_analyzer._is_significant(p_value, alpha)
                print(f"  {test_name}: p={p_value:.4f}, significant={significant}")
    
    # 7. Visualization
    logger.info("Creating visualizations...")
    try:
        plotter = ComparisonPlotter(config)
        
        # Create comparison plots
        plotter.plot_algorithm_comparison(metrics_df, save_path='basic_comparison_metrics.png')
        plotter.plot_glucose_traces_sample(full_results, n_samples=6, save_path='basic_comparison_traces.png')
        
        logger.info("Visualizations saved")
    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
    
    # 8. Save results
    logger.info("Saving results...")
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Save metrics
    metrics_df.to_csv(output_dir / 'basic_comparison_metrics.csv', index=False)
    
    # Save statistical results
    if comparison_results:
        import json
        with open(output_dir / 'basic_comparison_statistics.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy(obj):
                if hasattr(obj, 'item'):
                    return obj.item()
                elif hasattr(obj, 'tolist'):
                    return obj.tolist()
                return obj
            
            json_results = {}
            for key, value in comparison_results.items():
                if isinstance(value, dict):
                    json_results[key] = {k: convert_numpy(v) for k, v in value.items()}
                else:
                    json_results[key] = convert_numpy(value)
            
            json.dump(json_results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    print(f"\nExample completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"- Metrics: basic_comparison_metrics.csv")
    print(f"- Statistics: basic_comparison_statistics.json")
    print(f"- Plots: basic_comparison_*.png")


if __name__ == "__main__":
    main()
