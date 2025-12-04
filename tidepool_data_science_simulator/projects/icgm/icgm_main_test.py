#!/usr/bin/env python3
"""
Test version of icgm_main.py with reduced parameters for quick validation.

This script runs a minimal iCGM sensitivity analysis that can be used to:
1. Validate the simulation pipeline is working correctly
2. Compare results with the new insulin_algorithm_testing_framework

Test Parameters:
    - 1 virtual patient (VP 0)
    - True BG range: 70-150 mg/dL, step 20 (5 values)
    - Sensor BG range: 70-150 mg/dL, step 20 (5 values)  
    - Total simulations: 1 × 5 × 5 = 25 simulations
    - Expected runtime: ~2-5 minutes

Usage:
    python icgm_main_test.py
    
Output:
    Results saved to ~/data/processed/icgm_test_results/
"""

import os
import logging
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import process_simulation_data
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_simulation import run_icgm_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Test parameters - reduced for quick validation
TEST_PARAMS = {
    'paf_values': [0.4],
    'positive_rc_values': [True],
    'gradual_transitions_threshold_values': [50.0],
    'vp_ids': [0],  # Single virtual patient for quick test
    'true_bg_values': range(70, 151, 20),  # 5 values: 70, 90, 110, 130, 150
    'sensor_bg_values': range(70, 151, 20),  # 5 values: 70, 90, 110, 130, 150
}


def run_test():
    """Run the test iCGM simulation with reduced parameters."""
    
    logger.info("=" * 80)
    logger.info("iCGM SENSITIVITY ANALYSIS - TEST MODE")
    logger.info("=" * 80)
    
    # Calculate expected number of simulations
    num_vps = len(TEST_PARAMS['vp_ids'])
    num_true_bg = len(list(TEST_PARAMS['true_bg_values']))
    num_sensor_bg = len(list(TEST_PARAMS['sensor_bg_values']))
    expected_sims = num_vps * num_true_bg * num_sensor_bg
    
    logger.info(f"Test parameters:")
    logger.info(f"  - Virtual patients: {num_vps} (IDs: {TEST_PARAMS['vp_ids']})")
    logger.info(f"  - True BG values: {num_true_bg} ({list(TEST_PARAMS['true_bg_values'])})")
    logger.info(f"  - Sensor BG values: {num_sensor_bg} ({list(TEST_PARAMS['sensor_bg_values'])})")
    logger.info(f"  - PAF values: {TEST_PARAMS['paf_values']}")
    logger.info(f"  - Gradual threshold: {TEST_PARAMS['gradual_transitions_threshold_values']}")
    logger.info(f"  - Expected simulations: {expected_sims}")
    
    # Run simulations
    result_dirs = run_icgm_simulations(
        base_result_dir=os.path.join(DATA_DIR, "processed", "icgm_test_results"),
        **TEST_PARAMS
    )
    
    logger.info(f"Simulations complete. Results in: {result_dirs}")
    
    # Process results
    for result_dir in result_dirs:
        logger.info(f"Processing results in: {result_dir}")
        summary_csv = process_simulation_data(result_dir)
        logger.info(f"Summary saved to: {summary_csv}")
    
    logger.info("=" * 80)
    logger.info("TEST COMPLETE")
    logger.info("=" * 80)
    
    return result_dirs


if __name__ == '__main__':
    run_test()
