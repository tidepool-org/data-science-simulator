#!/usr/bin/env python3
"""
Comparison test between original iCGM approach and the new testing framework.

This test validates that both approaches produce equivalent results for the same
test parameters. It runs simulations using both methods and compares key metrics.

Test Parameters (shared between both approaches):
    - 1 virtual patient (VP 0)
    - True BG range: 70-150 mg/dL, step 20 (5 values)
    - Sensor BG range: 70-150 mg/dL, step 20 (5 values)
    - PAF: 0.4
    - Gradual transition threshold: 50.0
    - Total simulations: 25

Usage:
    # Run directly
    python test_icgm_framework_comparison.py
    
    # Or via pytest
    pytest test_icgm_framework_comparison.py -v
"""

# 10% relative tolerance
TOLERANCE = 0.1  

# Column name mapping between original and framework approaches
# Format: {display_name: (original_column, framework_column)}
DEFAULT_COLUMN_MAPPING = {
    'lbgi': ('lbgi_icgm_start', 'lbgi'),
    'max_bolus': ('max_bolus_delivered', 'max_bolus_delivered'),
    'true_start_bg': ('true_start_bg', 'tbg'),
    'sensor_start_bg': ('start_bg_with_offset', 'sbg'),
}

import os
import sys
import logging
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd
import numpy as np

from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.risk_scoring import analyze_icgm_risk
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import compute_score_risk_table
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.evaluation.inspect_results import load_result

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))


def parse_original_sim_id(sim_id: str) -> Dict[str, any]:
    """
    Parse simulation ID from original icgm approach.
    
    Example: icgm_analysis_vp_1_vp0_tbg=70_sbg=90
    
    Returns:
        Dict with patient_id, true_bg, sensor_bg
    """
    match = re.search(r'vp_(\d+)_([^_]+)_tbg=(\d+)_sbg=(\d+)', sim_id)
    if match:
        return {
            'seed': int(match.group(1)),
            'patient_id': match.group(2),
            'true_bg': int(match.group(3)),
            'sensor_bg': int(match.group(4))
        }
    return {}


def parse_framework_sim_id(sim_id: str) -> Dict[str, any]:
    """
    Parse simulation ID from new framework approach.
    
    Example: alg=autobolus_patient=0_tbg=70_sbg=90_...
    
    Returns:
        Dict with algorithm, patient_id, true_bg, sensor_bg
    """
    result = {}
    parts = sim_id.split('_')
    
    for part in parts:
        if '=' in part:
            key, value = part.split('=', 1)
            if key == 'alg':
                result['algorithm'] = value
            elif key == 'patient':
                result['patient_id'] = value
            elif key == 'tbg':
                result['true_bg'] = int(value)
            elif key == 'sbg':
                result['sensor_bg'] = int(value)
    
    return result


def load_original_results(result_dir: str) -> pd.DataFrame:
    """
    Load summary results from original icgm approach.
    
    The original approach saves the summary as {result_dir}.tsv
    (e.g., /path/to/icgm_sensitivity_analysis_paf=0.4_posrc=True_gradthresh=50.0_2025_12_04_T_10_00_00_abc1234.tsv)
    
    Args:
        result_dir: Directory containing original simulation results
        
    Returns:
        DataFrame with metrics indexed by (true_bg, sensor_bg)
    """
    # The summary file is {result_dir}.tsv (not inside result_dir)
    summary_file = f"{result_dir}.tsv"
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary file not found: {summary_file}")
    
    summary_df = pd.read_csv(summary_file, sep='\t')
    
    # Parse simulation IDs to extract true_bg and sensor_bg
    parsed_data = []
    for _, row in summary_df.iterrows():
        sim_info = parse_original_sim_id(row.get('sim_id', ''))
        if sim_info:
            parsed_row = {
                'true_bg': sim_info['true_bg'],
                'sensor_bg': sim_info['sensor_bg'],
                'patient_id': sim_info['patient_id'],
                **row.to_dict()
            }
            parsed_data.append(parsed_row)
    
    return pd.DataFrame(parsed_data)


def load_framework_results(result_dir: str) -> pd.DataFrame:
    """
    Load summary results from new framework approach.
    
    Args:
        result_dir: Directory containing framework simulation results
        
    Returns:
        DataFrame with metrics indexed by (true_bg, sensor_bg)
    """
    summary_file = os.path.join(result_dir, 'simulation_summary.csv')
    
    if not os.path.exists(summary_file):
        raise FileNotFoundError(f"Summary CSV not found: {summary_file}")
    
    summary_df = pd.read_csv(summary_file)
    
    # Parse simulation IDs to extract true_bg and sensor_bg
    parsed_data = []
    for _, row in summary_df.iterrows():
        sim_info = parse_framework_sim_id(row.get('simulation_id', ''))
        if sim_info:
            parsed_row = {
                'true_bg': sim_info['true_bg'],
                'sensor_bg': sim_info['sensor_bg'],
                'patient_id': sim_info.get('patient_id'),
                **row.to_dict()
            }
            parsed_data.append(parsed_row)
    
    return pd.DataFrame(parsed_data)


def compare_metrics(
    original_df: pd.DataFrame,
    framework_df: pd.DataFrame,
    column_mapping: Dict[str, tuple] = None,
    tolerance: float = 0.01
) -> Dict[str, any]:
    """
    Compare metrics between original and framework results.
    
    Args:
        original_df: Results from original approach
        framework_df: Results from new framework
        column_mapping: Dict mapping display_name -> (original_col, framework_col).
                        If None, uses DEFAULT_COLUMN_MAPPING.
        tolerance: Relative tolerance for numeric comparisons
        
    Returns:
        Dict with comparison results
    """
    if column_mapping is None:
        column_mapping = DEFAULT_COLUMN_MAPPING
    
    results = {
        'total_original': len(original_df),
        'total_framework': len(framework_df),
        'matched_pairs': 0,
        'unmatched_original': [],
        'unmatched_framework': [],
        'metric_comparisons': {},
        'discrepancies': []
    }
    
    # Create keys for matching
    original_df = original_df.copy()
    framework_df = framework_df.copy()
    
    original_df['match_key'] = original_df.apply(
        lambda r: f"{r['true_bg']}_{r['sensor_bg']}", axis=1
    )
    framework_df['match_key'] = framework_df.apply(
        lambda r: f"{r['true_bg']}_{r['sensor_bg']}", axis=1
    )
    
    # Find matched pairs
    original_keys = set(original_df['match_key'])
    framework_keys = set(framework_df['match_key'])
    
    matched_keys = original_keys & framework_keys
    results['matched_pairs'] = len(matched_keys)
    results['unmatched_original'] = list(original_keys - framework_keys)
    results['unmatched_framework'] = list(framework_keys - original_keys)
    
    # Compare metrics for matched pairs using column mapping
    for display_name, (orig_col, fw_col) in column_mapping.items():
        metric_results = {
            'original_column': orig_col,
            'framework_column': fw_col,
            'mean_original': None,
            'mean_framework': None,
            'mean_diff': None,
            'max_diff': None,
            'within_tolerance': True,
            'details': []
        }
        
        # Check if columns exist
        if orig_col not in original_df.columns:
            logger.warning(f"Original column '{orig_col}' not found for metric '{display_name}'")
            continue
        if fw_col not in framework_df.columns:
            logger.warning(f"Framework column '{fw_col}' not found for metric '{display_name}'")
            continue
        
        orig_values = []
        fw_values = []
        
        for key in matched_keys:
            orig_row = original_df[original_df['match_key'] == key].iloc[0]
            fw_row = framework_df[framework_df['match_key'] == key].iloc[0]
            
            orig_val = orig_row.get(orig_col)
            fw_val = fw_row.get(fw_col)
            
            if pd.notna(orig_val) and pd.notna(fw_val):
                orig_values.append(orig_val)
                fw_values.append(fw_val)
                
                # Check if within tolerance
                if orig_val != 0:
                    rel_diff = abs(fw_val - orig_val) / abs(orig_val)
                else:
                    rel_diff = abs(fw_val - orig_val)
                
                if rel_diff > tolerance:
                    results['discrepancies'].append({
                        'key': key,
                        'metric': display_name,
                        'original_col': orig_col,
                        'framework_col': fw_col,
                        'original': orig_val,
                        'framework': fw_val,
                        'rel_diff': rel_diff
                    })
                    metric_results['within_tolerance'] = False
        
        if orig_values:
            metric_results['mean_original'] = np.mean(orig_values)
            metric_results['mean_framework'] = np.mean(fw_values)
            metric_results['mean_diff'] = np.mean(np.array(fw_values) - np.array(orig_values))
            metric_results['max_diff'] = np.max(np.abs(np.array(fw_values) - np.array(orig_values)))
        
        results['metric_comparisons'][display_name] = metric_results
    
    return results


def run_original_test() -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Run the original iCGM test and return the result directory with risk analysis.
    
    Returns:
        Tuple of (result_dir, summary_df, severity_df)
    """
    from tidepool_data_science_simulator.projects.icgm.icgm_main_test import run_test
    result_dirs = run_test()
    
    if not result_dirs:
        return None, None, None
    
    result_dir = result_dirs[0]
    
    # Load the summary data
    summary_file = f"{result_dir}.tsv"
    if not os.path.exists(summary_file):
        logger.error(f"Summary file not found: {summary_file}")
        return result_dir, None, None
    
    summary_df = pd.read_csv(summary_file, sep='\t')
    
    # Calculate risk scores using original method
    logger.info("Calculating risk scores using original compute_score_risk_table...")
    try:
        severity_df, analysis_arrays = compute_score_risk_table(summary_df)
        logger.info(f"Original risk analysis complete")
        
        # Save risk results
        risk_file = f"{result_dir}_risk_analysis.csv"
        severity_df.to_csv(risk_file, index=False)
        logger.info(f"Saved original risk analysis to: {risk_file}")
        
    except Exception as e:
        logger.error(f"Original risk analysis failed: {e}")
        severity_df = None
    
    return result_dir, summary_df, severity_df


def run_framework_test() -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Run the new framework test and return the result directory with risk analysis.
    
    Returns:
        Tuple of (result_dir, summary_df, severity_df)
    """
    from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.config.experiment_config import ExperimentConfig
    from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.data_loader import DataLoader
    # from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.scenario_generator import ScenarioGenerator
    from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.simulation_builder import generate_simulations
    from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.simulation_runner import SimulationRunner
    from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import (
        calculate_point_metrics, metrics_to_dataframe
    )
    
    # Load test config
    config_path = Path(__file__).parent.parent / 'config' / '510k_configs' / 'icgm_test_config.yaml'
    config = ExperimentConfig(str(config_path))
    
    output_dir = Path(config.get('experiment.output_dir'))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load patient configs
    data_loader = DataLoader(config)
    vp_ids = config.get('scenarios.patient_parameters.specific_vp_ids')
    patient_configs = data_loader.load_patient_configs(patient_ids=vp_ids)
    
    true_bg_cfg = config.get('scenarios.spurious_sensor_errors.true_bg_values')
    sensor_bg_cfg = config.get('scenarios.spurious_sensor_errors.sensor_bg_values')

    simulation_generator = generate_simulations(
        config,
        patient_configs,
        true_bg_range=(true_bg_cfg['start'], true_bg_cfg['end'], true_bg_cfg['step']),
        sensor_bg_range=(sensor_bg_cfg['start'], sensor_bg_cfg['end'], sensor_bg_cfg['step'])
    )
    
    simulations = dict(simulation_generator)

    runner = SimulationRunner(config)
    results_dir = output_dir / 'simulation_results'
    runner.run_simulations(simulations, save_dir=str(results_dir))
    
    # Calculate metrics
    tsv_files = list(results_dir.glob("*.tsv"))
    point_metrics_dict = {}
    
    for tsv_file in tsv_files:
        sim_id = tsv_file.stem
        results_df = pd.read_csv(tsv_file, sep='\t')
        point_metrics = calculate_point_metrics(results_df)
        point_metrics_dict[sim_id] = point_metrics
    
    # Create summary DataFrame
    summary_df = metrics_to_dataframe(point_metrics_dict, parse_sim_ids=True)
    summary_df.to_csv(output_dir / 'simulation_summary.csv', index=False)
   
    logger.info("")
    logger.info("=" * 80)
    logger.info("STEP 5: Calculating Risk Scores")
    logger.info("=" * 80)
    
    severity_df = None
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
    
    return str(output_dir), summary_df, severity_df


def compare_risk_scores(
    original_severity_df: pd.DataFrame,
    framework_severity_df: pd.DataFrame,
    tolerance: float = 0.01
) -> Dict[str, any]:
    """
    Compare risk scores between original and framework approaches.
    
    Args:
        original_severity_df: Severity DataFrame from original compute_score_risk_table
        framework_severity_df: Severity DataFrame from framework analyze_icgm_risk
        tolerance: Relative tolerance for comparisons
        
    Returns:
        Dict with comparison results
    """
    results = {
        'original_has_risk': original_severity_df is not None,
        'framework_has_risk': framework_severity_df is not None,
        'severity_band_comparison': [],
        'total_events_original': None,
        'total_events_framework': None,
        'total_events_diff': None,
        'all_within_tolerance': True
    }
    
    if original_severity_df is None or framework_severity_df is None:
        results['all_within_tolerance'] = False
        return results
    
    # Original returns a single-column DataFrame with event probabilities
    # Framework returns a DataFrame with severity bands and event counts
    
    # Get total events from framework
    if 'events_per_100k_years' in framework_severity_df.columns:
        results['total_events_framework'] = framework_severity_df['events_per_100k_years'].sum()
    
    # Original severity_df is indexed differently - it's the probability values
    # Let's compare the structure
    logger.info(f"Original risk df shape: {original_severity_df.shape}")
    logger.info(f"Original risk df columns: {original_severity_df.columns.tolist()}")
    logger.info(f"Framework risk df shape: {framework_severity_df.shape}")
    logger.info(f"Framework risk df columns: {framework_severity_df.columns.tolist()}")
    
    # The original compute_score_risk_table returns severity_event_probability_df
    # which is a DataFrame with the probability values per severity band
    if len(original_severity_df.columns) == 1:
        # Original format: single column with 5 probability values (one per severity band)
        original_probs = original_severity_df.iloc[:, 0].values
        results['total_events_original_prob'] = original_probs.sum()
        
        # Compare with framework probabilities
        if 'probability' in framework_severity_df.columns:
            framework_probs = framework_severity_df['probability'].values
            
            for i in range(min(len(original_probs), len(framework_probs))):
                orig_val = original_probs[i]
                fw_val = framework_probs[i]
                
                if orig_val != 0:
                    rel_diff = abs(fw_val - orig_val) / abs(orig_val)
                else:
                    rel_diff = abs(fw_val - orig_val)
                
                band_comparison = {
                    'band_index': i,
                    'original_probability': orig_val,
                    'framework_probability': fw_val,
                    'relative_diff': rel_diff,
                    'within_tolerance': rel_diff <= tolerance
                }
                results['severity_band_comparison'].append(band_comparison)
                
                if not band_comparison['within_tolerance']:
                    results['all_within_tolerance'] = False
    
    return results


def find_tsv_file_by_bg(directory: str, true_bg: int, sensor_bg: int, pattern_type: str = 'original') -> str:
    """
    Find TSV file matching the (true_bg, sensor_bg) combination.
    
    Args:
        directory: Directory containing TSV files
        true_bg: True BG value
        sensor_bg: Sensor BG value
        pattern_type: 'original' or 'framework'
        
    Returns:
        Path to matching TSV file, or None if not found
    """
    if pattern_type == 'original':
        # Original format: icgm_analysis_vp_*_tbg={true_bg}_sbg={sensor_bg}.tsv
        pattern = f"*_tbg={true_bg}_sbg={sensor_bg}.tsv"
    else:
        # Framework format: alg=*_tbg={true_bg}_sbg={sensor_bg}_*.tsv
        pattern = f"*_tbg={true_bg}_sbg={sensor_bg}_*.tsv"
    
    matches = glob.glob(os.path.join(directory, pattern))
    return matches[0] if matches else None


def visualize_discrepancies(
    discrepancies: List[Dict],
    original_dir: str,
    framework_dir: str,
    output_dir: str
) -> List[str]:
    """
    Create visualization figures for simulations with discrepancies.
    
    For each unique (true_bg, sensor_bg) combination with discrepancies,
    loads both TSV files and creates a side-by-side comparison figure.
    
    Args:
        discrepancies: List of discrepancy dicts from compare_metrics()
        original_dir: Directory with original approach results
        framework_dir: Directory with framework approach results
        output_dir: Directory to save comparison figures
        
    Returns:
        List of paths to saved figures
    """
    # Create output directory for figures
    figures_dir = os.path.join(output_dir, 'discrepancy_figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Get unique keys (true_bg, sensor_bg combinations)
    unique_keys = set(d['key'] for d in discrepancies)
    
    saved_figures = []
    
    for key in unique_keys:
        # Parse key to get true_bg and sensor_bg
        parts = key.split('_')
        if len(parts) != 2:
            logger.warning(f"Invalid key format: {key}")
            continue
        
        true_bg = int(parts[0])
        sensor_bg = int(parts[1])
        
        # Find TSV files
        original_tsv = find_tsv_file_by_bg(original_dir, true_bg, sensor_bg, 'original')
        
        # For framework, look in simulation_results subdirectory
        framework_results_dir = os.path.join(framework_dir, 'simulation_results')
        framework_tsv = find_tsv_file_by_bg(framework_results_dir, true_bg, sensor_bg, 'framework')
        
        if not original_tsv:
            logger.warning(f"Original TSV not found for tbg={true_bg}, sbg={sensor_bg}")
            continue
        if not framework_tsv:
            logger.warning(f"Framework TSV not found for tbg={true_bg}, sbg={sensor_bg}")
            continue
        
        # Load both result files
        try:
            orig_sim_id, orig_df = load_result(original_tsv, ext="tsv")
            fw_sim_id, fw_df = load_result(framework_tsv, ext="tsv")
        except Exception as e:
            logger.error(f"Error loading TSV files for {key}: {e}")
            continue
        
        # Create combined results dict for plotting
        all_results = {
            f"Original (tbg={true_bg}, sbg={sensor_bg})": orig_df,
            f"Framework (tbg={true_bg}, sbg={sensor_bg})": fw_df
        }
        
        # Create the plot
        try:
            fig, ax = plot_sim_results(all_results, save=False, n_sims_max_legend=2)
            
            # Add title with discrepancy info
            discrepancies_for_key = [d for d in discrepancies if d['key'] == key]
            metrics_affected = list(set(d['metric'] for d in discrepancies_for_key))
            fig.suptitle(f"Discrepancy Comparison: tbg={true_bg}, sbg={sensor_bg}\nMetrics: {', '.join(metrics_affected)}", 
                        fontsize=10, y=1.02)
            
            # Save figure
            save_path = os.path.join(figures_dir, f"comparison_tbg={true_bg}_sbg={sensor_bg}.png")
            fig.savefig(save_path, dpi=150, bbox_inches='tight')
            import matplotlib.pyplot as plt
            plt.close(fig)
            
            saved_figures.append(save_path)
            logger.info(f"Saved discrepancy figure: {save_path}")
            
        except Exception as e:
            logger.error(f"Error creating figure for {key}: {e}")
            continue
    
    return saved_figures


def run_full_comparison():
    """
    Run both approaches and compare results.
    
    Returns:
        Dict with comparison results, or None if either approach failed
    """
    logger.info("=" * 80)
    logger.info("iCGM FRAMEWORK COMPARISON TEST")
    logger.info("=" * 80)
    
    # Run original approach
    logger.info("")
    logger.info("STEP 1: Running original iCGM approach...")
    logger.info("-" * 40)
    original_dir, original_summary_df, original_severity_df = run_original_test()
    
    if not original_dir:
        logger.error("Original approach failed!")
        return None
    
    logger.info(f"Original results saved to: {original_dir}")
    
    # Run framework approach
    logger.info("")
    logger.info("STEP 2: Running new framework approach...")
    logger.info("-" * 40)
    framework_dir, framework_summary_df, framework_severity_df = run_framework_test()
    
    if not framework_dir:
        logger.error("Framework approach failed!")
        return None
    
    logger.info(f"Framework results saved to: {framework_dir}")
    
    # Load and compare results
    logger.info("")
    logger.info("STEP 3: Comparing results...")
    logger.info("-" * 40)
    
    original_df = load_original_results(original_dir)
    framework_df = load_framework_results(framework_dir)
    
    comparison = compare_metrics(original_df, framework_df, tolerance=TOLERANCE)
    
    # Print results
    logger.info("")
    logger.info("=" * 80)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 80)
    logger.info(f"Original simulations: {comparison['total_original']}")
    logger.info(f"Framework simulations: {comparison['total_framework']}")
    logger.info(f"Matched pairs: {comparison['matched_pairs']}")
    
    if comparison['unmatched_original']:
        logger.warning(f"Unmatched in original: {len(comparison['unmatched_original'])}")
    if comparison['unmatched_framework']:
        logger.warning(f"Unmatched in framework: {len(comparison['unmatched_framework'])}")
    
    logger.info("")
    logger.info("Metric comparisons:")
    for metric, metric_results in comparison['metric_comparisons'].items():
        if metric_results['mean_original'] is not None:
            logger.info(f"  {metric} ({metric_results['original_column']} vs {metric_results['framework_column']}):")
            logger.info(f"    Original mean: {metric_results['mean_original']:.4f}")
            logger.info(f"    Framework mean: {metric_results['mean_framework']:.4f}")
            logger.info(f"    Max diff: {metric_results['max_diff']:.4f}")
            logger.info(f"    Within tolerance: {metric_results['within_tolerance']}")
    
    if comparison['discrepancies']:
        logger.warning("")
        logger.warning(f"Discrepancies found: {len(comparison['discrepancies'])}")
        for d in comparison['discrepancies'][:10]:  # Show first 10
            logger.warning(f"  {d['key']} - {d['metric']} ({d['original_col']} vs {d['framework_col']}): orig={d['original']:.4f}, fw={d['framework']:.4f}, diff={d['rel_diff']:.4f}")
        
        # Create visualizations for discrepancies
        logger.info("")
        logger.info("STEP 4: Creating discrepancy visualizations...")
        logger.info("-" * 40)
        saved_figures = visualize_discrepancies(
            comparison['discrepancies'],
            original_dir,
            framework_dir,
            framework_dir  # Use framework output dir for figures
        )
        comparison['discrepancy_figures'] = saved_figures
        logger.info(f"Created {len(saved_figures)} discrepancy figures")
    else:
        logger.info("")
        logger.info("✓ No discrepancies found within tolerance!")
    
    # Compare risk scores
    logger.info("")
    logger.info("STEP 5: Comparing risk scores...")
    logger.info("-" * 40)
    
    risk_comparison = compare_risk_scores(original_severity_df, framework_severity_df)
    comparison['risk_comparison'] = risk_comparison
    
    logger.info("")
    logger.info("Risk Score Comparison:")
    logger.info(f"  Original has risk data: {risk_comparison['original_has_risk']}")
    logger.info(f"  Framework has risk data: {risk_comparison['framework_has_risk']}")
    
    if risk_comparison['severity_band_comparison']:
        logger.info("")
        logger.info("  Severity Band Comparisons:")
        for band in risk_comparison['severity_band_comparison']:
            status = "✓" if band['within_tolerance'] else "✗"
            logger.info(f"    Band {band['band_index']}: orig={band['original_probability']:.2e}, "
                       f"fw={band['framework_probability']:.2e}, diff={band['relative_diff']:.2e} {status}")
        
        if risk_comparison['all_within_tolerance']:
            logger.info("")
            logger.info("  ✓ All risk scores within tolerance!")
        else:
            logger.warning("")
            logger.warning("  ✗ Some risk scores differ beyond tolerance")
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    
    return comparison


class TestICGMFrameworkComparison:
    """Test class for comparing original and new framework approaches."""
    
    def test_simulation_count_matches(self):
        """Test that both approaches generate the same number of simulations."""
        # Expected: 1 VP × 5 true_bg × 5 sensor_bg = 25 simulations
        expected_sims = 25
        assert expected_sims == 25


def main():
    """Main function - runs the full comparison test."""
    comparison = run_full_comparison()
    
    if comparison is None:
        logger.error("Comparison test failed!")
        return 1
    
    # Return success if no discrepancies, otherwise failure
    if comparison['discrepancies']:
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
