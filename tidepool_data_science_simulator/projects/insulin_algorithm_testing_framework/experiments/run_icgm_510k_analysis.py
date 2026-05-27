#!/usr/bin/env python3
"""
Turnkey iCGM Sensitivity Analysis for FDA 510k Submissions

This script runs a complete end-to-end iCGM sensitivity analysis pipeline for
regulatory submissions, including scenario generation, simulation execution,
risk scoring, and visualization.

Pipeline Steps:
    1. Load virtual patient configurations
    2. Generate iCGM grid scenarios (true BG × sensor BG)
    3. Run batch simulations with parallel processing
    4. Calculate LBGI-based risk scores across severity bands
    5. Generate regulatory-compliant visualizations (heatmaps)
    6. Export submission package with all required artifacts

Command-Line Arguments:
    --config PATH
        Path to YAML configuration file containing experiment parameters.
        Default: 'config/510k_configs/icgm_sensitivity_analysis.yaml'
        
        The config file specifies:
        - True BG range and step size (e.g., 40-405 mg/dL, step 5)
        - Sensor BG range and step size
        - Virtual patient selection (all or specific IDs)
        - Algorithm parameters (PAF, gradual transition thresholds)
        - Risk scoring parameters (severity bands, safety thresholds)
        - Processing options (parallel processes, batch size)
        
    --quick-test
        Run in quick test mode with reduced parameters for rapid validation.
        Overrides config to use:
        - Only 1 virtual patient
        - Small BG grid (70-150 mg/dL, step 20)
        - Approximately 16 scenarios instead of 93,555
        
        Use this to validate the pipeline before running full analysis.
        
    --output-dir PATH
        Override the output directory specified in the config file.
        All results, visualizations, and submission packages will be
        saved to this directory. Creates the directory if it doesn't exist.
        
    --no-visualizations
        Skip generation of visualizations (heatmaps, plots).
        Use this to speed up processing when only metrics are needed,
        or when running on headless systems without display capabilities.

Usage Examples:
    # Full production run with default config
    python run_icgm_510k_analysis.py
    
    # Quick test to validate pipeline
    python run_icgm_510k_analysis.py --quick-test
    
    # Use custom config and output directory
    python run_icgm_510k_analysis.py \\
        --config my_custom_config.yaml \\
        --output-dir results/icgm_sensitivity_2025_12_02
    
    # Skip visualizations (faster, good for metrics-only runs)
    python run_icgm_510k_analysis.py \\
        --config config/510k_configs/icgm_sensitivity_analysis.yaml \\
        --no-visualizations
    
    # Quick test with custom output location
    python run_icgm_510k_analysis.py \\
        --quick-test \\
        --output-dir results/quick_test

Output Structure:
    output_dir/
    ├── scenario_summary.json           # Scenario generation metadata
    ├── simulation_results/             # Individual simulation TSV files
    ├── risk_severity_analysis.csv      # Risk scores by severity band
    ├── risk_analysis_report.txt        # Human-readable risk report
    ├── visualizations/                 # Regulatory figures
    │   ├── risk_heatmap_grid.png
    │   └── risk_heatmap_grid.pdf
    └── submission_package/             # FDA submission-ready files
        ├── risk_severity_analysis.csv
        ├── risk_analysis_report.txt
        ├── scenario_summary.json
        ├── risk_heatmap_grid.png
        └── risk_heatmap_grid.pdf

Performance Notes:
    - Full analysis (~93,555 scenarios) typically takes 6-12 hours
    - Quick test (~16 scenarios) takes 5-10 minutes
    - Memory usage scales with batch size (configured in YAML)
    - Parallel processing uses CPU count specified in config
    - Results are saved incrementally to prevent data loss

Exit Codes:
    0: Success - all steps completed without errors
    1: Configuration error - failed to load or parse config file
    2: Data loading error - failed to load virtual patients
    3: Simulation error - simulation execution failed
    4: Analysis error - risk scoring or visualization failed

Requirements:
    - Python 3.8+
    - Virtual patient configuration files (JSON format)
    - Valid YAML configuration file
    - Sufficient disk space for results (estimate 10-50 GB)
    - Sufficient memory for batch processing (8-16 GB recommended)

For detailed configuration options, see:
    config/510k_configs/icgm_sensitivity_analysis.yaml

For framework documentation, see:
    tidepool_data_science_simulator/projects/insulin_algorithm_testing_framework/README.md
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import json

import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import (
    ExperimentConfig
)
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.data_loader import (
    DataLoader
)
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.simulation_builder import (
    generate_simulations
)
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import (
    calculate_point_metrics, metrics_to_dataframe, calculate_metrics_from_parquet
)
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.risk_scoring import (
    analyze_icgm_risk, generate_risk_report
)
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.visualization.regulatory_plots import (
    plot_risk_heatmap_grid, save_regulatory_figure
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run iCGM sensitivity analysis for FDA 510k submission'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='tidepool_data_science_simulator/projects/insulin_algorithm_testing_framework/config/510k_configs/icgm_test_config.yaml',
        # default='tidepool_data_science_simulator/projects/insulin_algorithm_testing_framework/config/510k_configs/icgm_sensitivity_analysis.yaml',
        help='Path to configuration file'
    )
    
    parser.add_argument(
        '--quick-test',
        action='store_true',
        help='Run quick test with reduced scenarios (1 patient, small grid)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Override output directory from config'
    )
    
    parser.add_argument(
        '--no-visualizations',
        action='store_true',
        help='Skip visualization generation'
    )
    
    return parser.parse_args()


def run_quick_test(config: ExperimentConfig) -> dict:
    """
    Run quick test with reduced parameters.
    
    Args:
        config: Experiment configuration
        
    Returns:
        Dictionary with test results
    """
    logger.info("=" * 80)
    logger.info("QUICK TEST MODE")
    logger.info("=" * 80)
    
    # Override config for testing
    config.set('scenarios.spurious_sensor_errors.true_bg_values.start', 110)
    config.set('scenarios.spurious_sensor_errors.true_bg_values.end', 150)
    config.set('scenarios.spurious_sensor_errors.true_bg_values.step', 20)
    config.set('scenarios.spurious_sensor_errors.sensor_bg_values.start', 110)
    config.set('scenarios.spurious_sensor_errors.sensor_bg_values.end', 150)
    config.set('scenarios.spurious_sensor_errors.sensor_bg_values.step', 20)
    config.set('scenarios.patient_parameters.specific_vp_ids', [0])  # Just 1 patient
    
    logger.info("Quick test parameters:")
    logger.info("  - True BG range: 70-150 mg/dL (step: 20)")
    logger.info("  - Sensor BG range: 70-150 mg/dL (step: 20)")
    logger.info("  - Virtual patients: 1")
    
    return {}


def main():
    """Main execution function."""
    args = parse_args()
    
    start_time = datetime.now()
    logger.info("=" * 80)
    logger.info("iCGM SENSITIVITY ANALYSIS FOR FDA 510k SUBMISSION")
    logger.info("=" * 80)
    logger.info(f"Start time: {start_time}")
    
    # Load configuration
    try:
        config = ExperimentConfig(args.config)
        logger.info(f"Loaded configuration from: {args.config}")
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Override output directory if specified
    if args.output_dir:
        config.set('experiment.output_dir', args.output_dir)
    
    output_dir = Path(config.get('experiment.output_dir'))
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Quick test mode
    if args.quick_test:
        run_quick_test(config)
    
    # ========================================================================
    # STEP 1: Load Patient Configurations
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 1: Loading Virtual Patient Configurations")
    logger.info("=" * 80)
    
    data_loader = DataLoader(config)
    
    vp_ids = config.get('scenarios.patient_parameters.specific_vp_ids')
    if vp_ids:
        patient_configs = data_loader.load_patient_configs(patient_ids=vp_ids)
        logger.info(f"Loaded {len(patient_configs)} specific virtual patients")
    else:
        num_patients = config.get('scenarios.patient_parameters.num_patients')
        patient_configs = data_loader.load_patient_configs(max_patients=num_patients)
        logger.info(f"Loaded {len(patient_configs)} virtual patients")
    
    # ========================================================================
    # STEP 2: Generate and Run Simulations
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 2: Generating and Running Simulations")
    logger.info("=" * 80)
    
    # Create BG range tuples from config (reusable)
    true_bg_cfg = config.get('scenarios.spurious_sensor_errors.true_bg_values')
    sensor_bg_cfg = config.get('scenarios.spurious_sensor_errors.sensor_bg_values')
    true_bg_range = (true_bg_cfg['start'], true_bg_cfg['end'], true_bg_cfg['step'])
    sensor_bg_range = (sensor_bg_cfg['start'], sensor_bg_cfg['end'], sensor_bg_cfg['step'])
    
    results_dir = output_dir / 'simulation_results'
    
    try:
        # Generate simulations - returns (generator, num_sims) tuple
        simulation_generator, num_sims = generate_simulations(
            config,
            patient_configs,
            true_bg_range=true_bg_range,
            sensor_bg_range=sensor_bg_range
        )
        
        # Save scenario summary before running
        summary = {
            'scenario_type': 'icgm_sensitivity',
            'num_patients': len(patient_configs),
            'true_bg_range': true_bg_range,
            'sensor_bg_range': sensor_bg_range,
            'total_simulations': num_sims
        }
        
        summary_path = output_dir / 'scenario_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Saved scenario summary: {summary_path}")
        logger.info(f"Expected simulations: {num_sims}")
        
        # Run simulations directly using run.py (bypassing SimulationRunner)
        processing_config = config.get_processing_config()
        _, _ = run_simulations(
            simulation_generator,
            save_dir=str(results_dir),
            save_results=processing_config.save_individual_results,
            compute_summary_metrics=False,
            num_procs=processing_config.parallel_processes,
            num_sims=num_sims,
            save_format=processing_config.save_format
        )
        
        logger.info("Simulation batch complete")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    
    # ========================================================================
    # STEP 3: Compute Metrics from Saved Results
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 3: Computing Metrics from Simulation Results")
    logger.info("=" * 80)
    
    try:
        # Load simulation results based on save format
        logger.info(f"Loading simulation results from: {results_dir}")
        
        save_format = processing_config.save_format
        
        # Check for parquet format first (preferred for performance with parallel processing)
        parquet_file = results_dir / "combined_results.parquet"
        if save_format in ('parquet', 'both') and parquet_file.exists():
            # Use parallel metrics calculation from parquet
            point_metrics_dict, metadata = calculate_metrics_from_parquet(
                str(parquet_file),
                n_processes=processing_config.parallel_processes,
                show_progress=True
            )
        else:
            # Fall back to TSV files (sequential processing)
            point_metrics_dict = {}
            tsv_files = list(results_dir.glob("*.tsv"))
            logger.info(f"Found {len(tsv_files)} TSV result files")
            
            for tsv_file in tsv_files:
                sim_id = tsv_file.stem  # filename without extension
                try:
                    results_df = pd.read_csv(tsv_file, sep='\t')
                    point_metrics = calculate_point_metrics(results_df)
                    point_metrics_dict[sim_id] = point_metrics
                except Exception as e:
                    logger.warning(f"Failed to process {tsv_file.name}: {e}")
            
            logger.info(f"Computed metrics for {len(point_metrics_dict)} simulations from TSV files")
        
        # Create summary DataFrame
        summary_df = metrics_to_dataframe(point_metrics_dict, parse_sim_ids=True)
        
        # Save summary CSV
        summary_csv_path = output_dir / 'simulation_summary.csv'
        summary_df.to_csv(summary_csv_path, index=False)
        logger.info(f"Saved simulation summary: {summary_csv_path}")
        
    except Exception as e:
        logger.error(f"Metrics calculation failed: {e}")
        raise
    
    # ========================================================================
    # STEP 4: Calculate Risk Scores
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 4: Calculating Risk Scores")
    logger.info("=" * 80)
    
    try:
        # Perform risk analysis
        severity_df, analysis_arrays, report = analyze_icgm_risk(
            summary_df,
            population_type=config.get('risk_scoring.population_type', 'adult')
        )
        
        # Save risk analysis results
        severity_df.to_csv(output_dir / 'risk_severity_analysis.csv', index=False)
        logger.info(f"Saved risk severity analysis")
        
        # Save risk report
        report_path = output_dir / 'risk_analysis_report.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        logger.info(f"Saved risk analysis report: {report_path}")
        
        # Print report to console
        print("")
        print(report)
        
    except Exception as e:
        logger.error(f"Risk analysis failed: {e}")
        raise
    
    # ========================================================================
    # STEP 5: Generate Regulatory Visualizations
    # ========================================================================
    if not args.no_visualizations:
        logger.info("")
        logger.info("=" * 80)
        logger.info("STEP 5: Generating Regulatory Visualizations")
        logger.info("=" * 80)
        
        viz_dir = output_dir / 'visualizations'
        viz_dir.mkdir(exist_ok=True)
        
        try:
            # Prepare data for heatmap grid
            severity_bands = severity_df['severity_band'].tolist()
            
            # Reshape analysis arrays for plotting
            risk_data_dict = {}
            for idx, band in enumerate(severity_bands):
                # Extract severity probability data for this band
                risk_data_dict[band] = analysis_arrays['severity_probs'][:, idx] * analysis_arrays['joint_probs']
            
            # Create risk heatmap grid
            fig, axes = plot_risk_heatmap_grid(
                analysis_arrays['true_bg'],
                analysis_arrays['sensor_bg'],
                risk_data_dict,
                severity_bands,
                safety_thresholds=severity_df['safety_threshold'].tolist(),
                shared_z_scale=True
            )
            
            # Save in multiple formats
            save_regulatory_figure(
                fig,
                viz_dir / 'risk_heatmap_grid',
                dpi=300,
                formats=['png', 'pdf']
            )
            
            logger.info(f"Saved risk heatmap grid visualizations")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            logger.warning("Continuing without visualizations...")
    
    # ========================================================================
    # STEP 6: Export Results for Submission
    # ========================================================================
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 6: Exporting Results for Submission")
    logger.info("=" * 80)
    
    submission_dir = output_dir / 'submission_package'
    submission_dir.mkdir(exist_ok=True)
    
    # Copy key files to submission package
    import shutil
    
    files_to_copy = [
        (output_dir / 'risk_severity_analysis.csv', submission_dir / 'risk_severity_analysis.csv'),
        (output_dir / 'risk_analysis_report.txt', submission_dir / 'risk_analysis_report.txt'),
        (output_dir / 'scenario_summary.json', submission_dir / 'scenario_summary.json'),
    ]
    
    if not args.no_visualizations:
        files_to_copy.extend([
            (viz_dir / 'risk_heatmap_grid.png', submission_dir / 'risk_heatmap_grid.png'),
            (viz_dir / 'risk_heatmap_grid.pdf', submission_dir / 'risk_heatmap_grid.pdf'),
        ])
    
    for src, dst in files_to_copy:
        if src.exists():
            shutil.copy(src, dst)
            logger.info(f"Copied: {src.name}")
    
    logger.info(f"Submission package ready: {submission_dir}")
    
    # ========================================================================
    # COMPLETION
    # ========================================================================
    end_time = datetime.now()
    duration = end_time - start_time
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("ANALYSIS COMPLETE")
    logger.info("=" * 80)
    logger.info(f"End time: {end_time}")
    logger.info(f"Total duration: {duration}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Submission package: {submission_dir}")
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
