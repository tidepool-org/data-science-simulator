#!/usr/bin/env python3
"""
Basic comparison example for insulin algorithm testing framework.

This simplified example demonstrates how to use the run_experiment function
to easily compare insulin delivery algorithms with minimal setup.

Usage:
    python basic_comparison.py
"""

import logging
import json
from pathlib import Path
import os

from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.analysis.statistical_analyzer import StatisticalAnalyzer
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import ExperimentConfig
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import (
    calculate_metrics_batch,
    create_metrics_dataframe
)
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.scenario_runner import run_experiment
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.visualization.comparison_plots import ComparisonPlotter

from tidepool_data_science_simulator.utils import DATA_DIR

CONFIG_FILE = 'tidepool_data_science_simulator/projects/insulin_algorithm_testing_framework/config/510k_short_run_config.yaml'  # Default config file path
OUTPUT_DIR = os.path.join(DATA_DIR, 'processed_data', 'insulin_algorithm_testing_framework', '510k_short_run_example')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Run basic comparison example using the simplified run_experiment function."""
    
    logger.info("Starting basic insulin algorithm comparison")
    
    # 1. Load and configure experiment
    logger.info("Loading configuration...")
    config = ExperimentConfig(CONFIG_FILE)  # Uses default config
    
    config.set('experiment.output_dir', OUTPUT_DIR)

    # 2. Run complete experiment
    logger.info("Running experiment...")
    try:
        full_results = run_experiment(config, max_patients=5)
        
        logger.info("Experiment completed successfully!")
        
        #  # 4. Calculate metrics
        # logger.info("Calculating metrics...")
        # metrics_calculator = MetricsCalculator(config)
        
        # # Calculate metrics for all results
        # metrics_dict = metrics_calculator.calculate_metrics_batch(full_results)
        
        # # Create metrics DataFrame
        # metrics_df = metrics_calculator.create_metrics_dataframe(metrics_dict)
        
        # if metrics_df.empty:
        #     raise ValueError("No metrics calculated")
        
        # logger.info(f"Calculated metrics for {len(metrics_dict)} simulations")
        # logger.info(f"Metrics columns: {list(metrics_df.columns)}")
        
        # # 5. Statistical analysis
        # logger.info("Performing statistical analysis...")
        # statistical_analyzer = StatisticalAnalyzer(config)
        
        # # Get enabled algorithms for comparison
        # enabled_algorithms = config.get_enabled_algorithms()
        
        # if len(enabled_algorithms) < 2:
        #     logger.warning("Statistical comparison requires at least 2 algorithms")
        #     comparison_results = {}
        # else:
        #     # Use first algorithm as reference, others as comparison
        #     reference_algorithm = enabled_algorithms[0]
        #     comparison_algorithms = enabled_algorithms[1:]
            
        #     # Perform paired comparison
        #     comparison_results = statistical_analyzer.compare_algorithms(
        #         metrics_df, 
        #         reference_algorithm=reference_algorithm,
        #         comparison_algorithms=comparison_algorithms
        #     )
        
        # logger.info("Statistical analysis completed")
        
        # # 6. Return results
        # logger.info(f"Experiment completed successfully!")
        # logger.info(f"Total simulations: {len(full_results)}")
        # logger.info(f"Metrics calculated: {len(metrics_df)}")

        # # 3. Display key results
        # print("\n" + "="*60)
        # print("BASIC COMPARISON RESULTS")
        # print("="*60)
        
        # # Summary statistics
        # print(f"\nCompleted {len(metrics_df)} simulations")
        # print(f"Algorithms tested: {metrics_df['alg'].unique()}")
        
        # print("\nSummary Statistics by Algorithm:")
        # summary_stats = metrics_df.groupby('alg')[['time_in_range_70_180', 'time_below_70', 'mean_glucose']].agg(['mean', 'std'])
        # print(summary_stats)
        
        # # Statistical test results
        # if comparison_results and 'statistical_tests' in comparison_results:
        #     print("\nStatistical Test Results:")
        #     alpha = config.get_analysis_config().alpha
        #     for metric, test_results in comparison_results['statistical_tests'].items():
        #         print(f"\n{metric}:")
        #         for test_name, test_result in test_results.items():
        #             p_value = test_result["p_value"]
        #             significant = p_value < alpha
        #             print(f"  {test_name}: p={p_value:.4f}, significant={significant}")
        
        # # 4. Create visualizations (optional)
        # logger.info("Creating visualizations...")
        # try:
        #     plotter = ComparisonPlotter(config)
        #     plotter.plot_algorithm_comparison(metrics_df, save_path='basic_comparison_metrics.png')
        #     logger.info("Visualizations saved")
        # except Exception as e:
        #     logger.warning(f"Visualization failed: {e}")
        
        # # 5. Save results
        # logger.info("Saving results...")
        # output_dir = Path(config.output_dir)
        # output_dir.mkdir(exist_ok=True)
        
        # # Save metrics
        # metrics_df.to_csv(output_dir / 'basic_comparison_metrics.csv', index=False)
        
        # # Save statistical results
        # if comparison_results:
        #     with open(output_dir / 'basic_comparison_statistics.json', 'w') as f:
        #         # Convert numpy types to native Python types for JSON serialization
        #         def convert_numpy(obj):
        #             if hasattr(obj, 'item'):
        #                 return obj.item()
        #             elif hasattr(obj, 'tolist'):
        #                 return obj.tolist()
        #             return obj
                
        #         json_results = {}
        #         for key, value in comparison_results.items():
        #             if isinstance(value, dict):
        #                 json_results[key] = {k: convert_numpy(v) for k, v in value.items()}
        #             else:
        #                 json_results[key] = convert_numpy(value)
                
        #         json.dump(json_results, f, indent=2)
        
        # logger.info(f"Results saved to {output_dir}")
        
        # print(f"\nExample completed successfully!")
        # print(f"Results saved to: {output_dir}")
        # print(f"- Metrics: basic_comparison_metrics.csv")
        # print(f"- Statistics: basic_comparison_statistics.json")
        # print(f"- Plots: basic_comparison_*.png")
        
    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
