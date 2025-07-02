#!/usr/bin/env python3
"""
Parameter sweep example for insulin algorithm testing framework.

This example demonstrates how to perform a comprehensive parameter sweep
across different partial application factors for autobolus algorithm.

Usage:
    python parameter_sweep.py
"""

import logging
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import ExperimentConfig
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.data_loader import DataLoader
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.scenario_generator import ScenarioGenerator
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.simulation_runner import SimulationRunner
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import MetricsCalculator
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.analysis.statistical_analyzer import StatisticalAnalyzer

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    """Run parameter sweep example."""
    
    logger.info("Starting parameter sweep analysis")
    
    # 1. Load configuration
    logger.info("Loading configuration...")
    config = ExperimentConfig()
    
    # Configure for parameter sweep
    config.set('experiment.name', 'parameter_sweep_example')
    config.set('scenarios.initial_bg.range', [120, 120])  # Fixed initial BG
    config.set('scenarios.meal_scenarios.unannounced_meals', [40])  # Fixed meal size
    config.set('scenarios.settings_mismatches.multipliers', [1.0])  # No settings mismatch
    
    # Define parameter sweep range
    paf_values = np.arange(0.2, 0.7, 0.05)  # 0.2 to 0.65 in steps of 0.05
    config.set('algorithms.autobolus.partial_application_factors', paf_values.tolist())
    
    logger.info(f"Parameter sweep range: {paf_values}")
    
    # 2. Load patient data
    logger.info("Loading patient data...")
    data_loader = DataLoader(config)
    patient_configs = data_loader.load_patient_configs(max_patients=5)  # Small subset
    
    logger.info(f"Loaded {len(patient_configs)} patient configurations")
    
    # 3. Generate scenarios
    logger.info("Generating scenarios...")
    scenario_generator = ScenarioGenerator(config)
    
    # Generate scenarios for autobolus only
    scenarios = list(scenario_generator.generate_scenarios_for_algorithm(
        patient_configs, 'autobolus'
    ))
    
    logger.info(f"Generated {len(scenarios)} autobolus scenarios")
    
    # 4. Run simulations
    logger.info("Running simulations...")
    simulation_runner = SimulationRunner(config)
    
    # Create simulation objects
    simulations = {}
    for scenario in scenarios:
        simulation = simulation_runner.create_simulation_from_scenario(scenario)
        simulations[simulation.sim_id] = simulation
    
    # Run batch simulations
    full_results, summary_results = simulation_runner.run_batch_simulations(simulations)
    
    logger.info(f"Completed {len(full_results)} simulations")
    
    # 5. Calculate metrics
    logger.info("Calculating metrics...")
    metrics_calculator = MetricsCalculator(config)
    
    # Calculate metrics for all results
    metrics_dict = metrics_calculator.calculate_metrics_batch(full_results)
    
    # Create metrics DataFrame
    metrics_df = metrics_calculator.create_metrics_dataframe(metrics_dict)
    
    logger.info(f"Calculated metrics for {len(metrics_dict)} simulations")
    
    # 6. Analyze parameter sweep results
    logger.info("Analyzing parameter sweep results...")
    
    # Group by partial application factor
    sweep_results = metrics_df.groupby('paf').agg({
        'time_in_range_70_180': ['mean', 'std'],
        'time_below_70': ['mean', 'std'],
        'time_below_54': ['mean', 'std'],
        'mean_glucose': ['mean', 'std'],
        'cv_glucose': ['mean', 'std'],
        'cumulative_insulin': ['mean', 'std']
    }).round(2)
    
    print("\n" + "="*80)
    print("PARAMETER SWEEP RESULTS")
    print("="*80)
    print("\nResults by Partial Application Factor:")
    print(sweep_results)
    
    # Find optimal PAF values
    optimal_tir = sweep_results[('time_in_range_70_180', 'mean')].idxmax()
    optimal_safety = sweep_results[('time_below_70', 'mean')].idxmin()
    optimal_glucose = sweep_results[('mean_glucose', 'mean')].apply(lambda x: abs(x - 140)).idxmin()
    
    print(f"\nOptimal PAF values:")
    print(f"- Best Time in Range (70-180): {optimal_tir:.2f}")
    print(f"- Best Safety (lowest time <70): {optimal_safety:.2f}")
    print(f"- Best Mean Glucose (closest to 140): {optimal_glucose:.2f}")
    
    # 7. Create visualizations
    logger.info("Creating parameter sweep visualizations...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Parameter Sweep: Partial Application Factor Analysis', fontsize=16)
    
    # Plot 1: Time in Range
    ax = axes[0, 0]
    paf_means = sweep_results[('time_in_range_70_180', 'mean')]
    paf_stds = sweep_results[('time_in_range_70_180', 'std')]
    ax.errorbar(paf_means.index, paf_means.values, yerr=paf_stds.values, 
                marker='o', capsize=5, capthick=2)
    ax.set_xlabel('Partial Application Factor')
    ax.set_ylabel('Time in Range 70-180 (%)')
    ax.set_title('Time in Range vs PAF')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Time Below 70
    ax = axes[0, 1]
    safety_means = sweep_results[('time_below_70', 'mean')]
    safety_stds = sweep_results[('time_below_70', 'std')]
    ax.errorbar(safety_means.index, safety_means.values, yerr=safety_stds.values, 
                marker='s', color='red', capsize=5, capthick=2)
    ax.set_xlabel('Partial Application Factor')
    ax.set_ylabel('Time Below 70 (%)')
    ax.set_title('Hypoglycemia Risk vs PAF')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean Glucose
    ax = axes[0, 2]
    glucose_means = sweep_results[('mean_glucose', 'mean')]
    glucose_stds = sweep_results[('mean_glucose', 'std')]
    ax.errorbar(glucose_means.index, glucose_means.values, yerr=glucose_stds.values, 
                marker='^', color='green', capsize=5, capthick=2)
    ax.set_xlabel('Partial Application Factor')
    ax.set_ylabel('Mean Glucose (mg/dL)')
    ax.set_title('Mean Glucose vs PAF')
    ax.axhline(y=140, color='gray', linestyle='--', alpha=0.7, label='Target (140)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Glucose Variability
    ax = axes[1, 0]
    cv_means = sweep_results[('cv_glucose', 'mean')]
    cv_stds = sweep_results[('cv_glucose', 'std')]
    ax.errorbar(cv_means.index, cv_means.values, yerr=cv_stds.values, 
                marker='d', color='purple', capsize=5, capthick=2)
    ax.set_xlabel('Partial Application Factor')
    ax.set_ylabel('Coefficient of Variation (%)')
    ax.set_title('Glucose Variability vs PAF')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Insulin Delivery
    ax = axes[1, 1]
    insulin_means = sweep_results[('cumulative_insulin', 'mean')]
    insulin_stds = sweep_results[('cumulative_insulin', 'std')]
    ax.errorbar(insulin_means.index, insulin_means.values, yerr=insulin_stds.values, 
                marker='h', color='orange', capsize=5, capthick=2)
    ax.set_xlabel('Partial Application Factor')
    ax.set_ylabel('Total Insulin (U)')
    ax.set_title('Insulin Delivery vs PAF')
    ax.grid(True, alpha=0.3)
    
    # Plot 6: Multi-objective view
    ax = axes[1, 2]
    # Normalize metrics for comparison (0-1 scale)
    tir_norm = (paf_means - paf_means.min()) / (paf_means.max() - paf_means.min())
    safety_norm = 1 - (safety_means - safety_means.min()) / (safety_means.max() - safety_means.min())  # Invert for safety
    
    ax.plot(tir_norm.index, tir_norm.values, 'o-', label='TIR (normalized)', linewidth=2)
    ax.plot(safety_norm.index, safety_norm.values, 's-', label='Safety (normalized)', linewidth=2)
    ax.set_xlabel('Partial Application Factor')
    ax.set_ylabel('Normalized Score (0-1)')
    ax.set_title('Multi-objective Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path(config.output_dir)
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'parameter_sweep_analysis.png', dpi=300, bbox_inches='tight')
    logger.info("Parameter sweep plot saved")
    
    # 8. Statistical analysis of trends
    logger.info("Performing trend analysis...")
    
    # Correlation analysis
    correlations = {}
    for metric in ['time_in_range_70_180', 'time_below_70', 'mean_glucose', 'cv_glucose']:
        metric_values = sweep_results[(metric, 'mean')].values
        correlation = np.corrcoef(paf_values, metric_values)[0, 1]
        correlations[metric] = correlation
    
    print(f"\nCorrelations with PAF:")
    for metric, corr in correlations.items():
        print(f"- {metric}: {corr:.3f}")
    
    # 9. Save results
    logger.info("Saving results...")
    
    # Save detailed metrics
    metrics_df.to_csv(output_dir / 'parameter_sweep_detailed.csv', index=False)
    
    # Save summary results
    sweep_results.to_csv(output_dir / 'parameter_sweep_summary.csv')
    
    # Save optimal values
    optimal_results = {
        'optimal_tir_paf': float(optimal_tir),
        'optimal_safety_paf': float(optimal_safety),
        'optimal_glucose_paf': float(optimal_glucose),
        'correlations': correlations
    }
    
    import json
    with open(output_dir / 'parameter_sweep_optimal.json', 'w') as f:
        json.dump(optimal_results, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    
    print(f"\nParameter sweep completed successfully!")
    print(f"Results saved to: {output_dir}")
    print(f"- Detailed metrics: parameter_sweep_detailed.csv")
    print(f"- Summary results: parameter_sweep_summary.csv")
    print(f"- Optimal values: parameter_sweep_optimal.json")
    print(f"- Visualization: parameter_sweep_analysis.png")


if __name__ == "__main__":
    main()
