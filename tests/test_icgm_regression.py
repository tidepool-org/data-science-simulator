"""
Regression tests for iCGM analysis pipeline.

These tests ensure that changes to the codebase don't unexpectedly alter
the simulation outputs. Baseline data is stored in tests/test_data/regression/icgm/
and compared against current runs with exact matching on key metrics.

Configuration:
- 5x5 BG ranges (25 combinations per VP)
- 3 VPs (75 total simulations)
- Fixed random seeds for determinism
- Exact comparison on all key metrics
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from tidepool_data_science_simulator.projects.icgm.icgm_analysis_simulation import run_icgm_simulations
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import (
    process_simulation_data,
    compute_score_risk_table
)

# Constants
BASELINE_DIR = Path("tests/test_data/regression/icgm")
TOLERANCE = 1e-10  # Strict tolerance for exact matching

# Fixed test parameters for reproducibility
TEST_PARAMS = {
    'paf': 0.4,
    'positive_rc': False,
    'num_vps': 3,
    'true_bg_values': [450, 60, 70, 80],   # 4 values
    'sensor_bg_values': [55, 65, 75, 85], # 4 values
}

# Key columns to compare (all critical metrics)
KEY_COLUMNS_SUMMARY = [
    'lbgi_icgm_start',
    'lbgi_icgm_valid',
    'true_start_bg',
    'start_bg_with_offset',
    'max_bolus_delivered',
    'traditional_bolus_delivered',
    'bolus_diff',
    'sbr',
    'isf',
    'cir'
]


@pytest.fixture
def baseline_data():
    """Load baseline regression data"""
    if not BASELINE_DIR.exists():
        pytest.skip("Baseline data not found. Run test_generate_baseline first.")
    
    return {
        'summary': pd.read_csv(BASELINE_DIR / "baseline_summary.csv", sep='\t', index_col=0),
        'risk_table': pd.read_csv(BASELINE_DIR / "baseline_risk_table.csv", header=None),
        'aux_data': np.load(BASELINE_DIR / "baseline_aux_data.npz")
    }


def test_regression_summary_df(baseline_data):
    """
    Regression test: summary DataFrame matches baseline exactly.
    
    Tests all key metrics from simulation results:
    - LBGI values (start and valid)
    - Blood glucose values (true and sensor)
    - Bolus delivery (max, traditional, diff)
    - Patient parameters (sbr, isf, cir)
    """
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run pipeline with FIXED parameters
        result_dir = run_icgm_simulations(
            result_dir=tmp_dir,
            **TEST_PARAMS
        )
        
        summary_csv = process_simulation_data(result_dir)
        summary_df = pd.read_csv(summary_csv, sep='\t', index_col=0)
        
        baseline_summary = baseline_data['summary']
        
        # Check row count (should be 4x4x3 = 48 simulations)
        assert len(summary_df) == len(baseline_summary), \
            f"Row count mismatch: {len(summary_df)} vs {len(baseline_summary)} (expected 75)"
        
        # Sort both for comparison (sim_id might vary in order)
        summary_df_sorted = summary_df.sort_values('sim_id').reset_index(drop=True)
        baseline_sorted = baseline_summary.sort_values('sim_id').reset_index(drop=True)
        
        # Compare key columns with exact matching
        failures = []
        for col in KEY_COLUMNS_SUMMARY:
            if col in summary_df_sorted.columns:
                try:
                    pd.testing.assert_series_equal(
                        summary_df_sorted[col],
                        baseline_sorted[col],
                        check_exact=False,  # Use atol instead
                        atol=TOLERANCE,
                        rtol=0,
                        check_names=True,
                        obj=f"Column: {col}"
                    )
                except AssertionError as e:
                    failures.append(f"Column '{col}': {str(e)}")
            else:
                failures.append(f"Column '{col}' not found in current results")
        
        if failures:
            pytest.fail(f"Regression test failed:\n" + "\n".join(failures))


def test_regression_risk_table(baseline_data):
    """
    Regression test: risk table matches baseline exactly.
    
    Tests the 5 severity bands risk table computed from
    the simulation results.
    """
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_dir = run_icgm_simulations(result_dir=tmp_dir, **TEST_PARAMS)
        summary_csv = process_simulation_data(result_dir)
        summary_df = pd.read_csv(summary_csv, sep='\t', index_col=0)
        
        severity_df, aux_data = compute_score_risk_table(summary_df)
        
        baseline_risk = baseline_data['risk_table']
        
        # Compare risk table (5 severity bands)
        pd.testing.assert_frame_equal(
            severity_df,
            baseline_risk,
            check_exact=False,
            atol=TOLERANCE,
            rtol=0,
            check_dtype=True,
            obj="Risk table DataFrame"
        )


def test_regression_auxiliary_data(baseline_data):
    """
    Regression test: auxiliary data matches baseline exactly.
    
    Tests the auxiliary arrays returned by compute_score_risk_table:
    - low_icgm_axis: iCGM BG axis values
    - low_true_axis: True BG axis values  
    - mean_lbgi_start: Mean LBGI probabilities per severity band
    - joint_prob: Joint probability distribution
    """
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        result_dir = run_icgm_simulations(result_dir=tmp_dir, **TEST_PARAMS)
        summary_csv = process_simulation_data(result_dir)
        summary_df = pd.read_csv(summary_csv, sep='\t', index_col=0)
        
        severity_df, aux_data = compute_score_risk_table(summary_df)
        low_icgm_axis, low_true_axis, mean_lbgi_start, joint_prob = aux_data
        
        baseline_aux = baseline_data['aux_data']
        
        # Compare all auxiliary arrays with detailed error messages
        failures = []
        
        try:
            np.testing.assert_allclose(
                low_icgm_axis,
                baseline_aux['low_icgm_axis'],
                atol=TOLERANCE, rtol=0
            )
        except AssertionError as e:
            failures.append(f"low_icgm_axis: {str(e)}")
        
        try:
            np.testing.assert_allclose(
                low_true_axis,
                baseline_aux['low_true_axis'],
                atol=TOLERANCE, rtol=0
            )
        except AssertionError as e:
            failures.append(f"low_true_axis: {str(e)}")
        
        try:
            np.testing.assert_allclose(
                mean_lbgi_start,
                baseline_aux['mean_lbgi_start'],
                atol=TOLERANCE, rtol=0
            )
        except AssertionError as e:
            failures.append(f"mean_lbgi_start: {str(e)}")
        
        try:
            np.testing.assert_allclose(
                joint_prob,
                baseline_aux['joint_prob'],
                atol=TOLERANCE, rtol=0
            )
        except AssertionError as e:
            failures.append(f"joint_prob: {str(e)}")
        
        if failures:
            pytest.fail(f"Auxiliary data regression test failed:\n" + "\n".join(failures))


# Baseline generation (run manually when needed)
@pytest.mark.skip(reason="Manual baseline generation - remove skip to regenerate")
def test_generate_baseline():
    """
    Generate baseline regression data.
    
    To regenerate baseline:
    1. Remove @pytest.mark.skip decorator above
    2. Run: pytest tests/test_icgm_regression.py::test_generate_baseline -v -s
    3. Re-add @pytest.mark.skip decorator
    4. Commit baseline files with clear commit message
    
    The baseline will be saved to: tests/test_data/regression/icgm/
    """
    
    BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"\nGenerating baseline data with parameters:")
    print(f"  PAF: {TEST_PARAMS['paf']}")
    print(f"  Positive RC: {TEST_PARAMS['positive_rc']}")
    print(f"  VPs: {TEST_PARAMS['num_vps']}")
    print(f"  True BG values: {TEST_PARAMS['true_bg_values']}")
    print(f"  Sensor BG values: {TEST_PARAMS['sensor_bg_values']}")
    print(f"  Expected simulations: {len(TEST_PARAMS['true_bg_values']) * len(TEST_PARAMS['sensor_bg_values']) * TEST_PARAMS['num_vps']}")
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Run pipeline with fixed parameters
        print(f"\nRunning simulations...")
        result_dir = run_icgm_simulations(result_dir=tmp_dir, **TEST_PARAMS)
        
        # Process and save summary
        print(f"Processing simulation results...")
        summary_csv = process_simulation_data(result_dir)
        summary_df = pd.read_csv(summary_csv, sep='\t', index_col=0)
        summary_df.to_csv(BASELINE_DIR / "baseline_summary.csv", sep='\t')
        
        # Compute and save risk table
        print(f"Computing risk table...")
        severity_df, aux_data = compute_score_risk_table(summary_df)
        severity_df.to_csv(BASELINE_DIR / "baseline_risk_table.csv", header=False, index=False)
        
        # Save auxiliary data
        print(f"Saving auxiliary data...")
        low_icgm_axis, low_true_axis, mean_lbgi_start, joint_prob = aux_data
        np.savez(
            BASELINE_DIR / "baseline_aux_data.npz",
            low_icgm_axis=low_icgm_axis,
            low_true_axis=low_true_axis,
            mean_lbgi_start=mean_lbgi_start,
            joint_prob=joint_prob
        )
        
        # Save test parameters for reference
        with open(BASELINE_DIR / "test_params.json", 'w') as f:
            json.dump(TEST_PARAMS, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Baseline data saved to {BASELINE_DIR}")
        print(f"{'='*60}")
        print(f"Summary rows: {len(summary_df)}")
        print(f"Risk table shape: {severity_df.shape}")
        print(f"Auxiliary data shapes:")
        print(f"  low_icgm_axis: {len(low_icgm_axis)}")
        print(f"  low_true_axis: {len(low_true_axis)}")
        print(f"  mean_lbgi_start: {mean_lbgi_start.shape}")
        print(f"  joint_prob: {len(joint_prob)}")
        print(f"\nNext steps:")
        print(f"1. Re-add @pytest.mark.skip decorator to test_generate_baseline")
        print(f"2. Run regression tests: pytest tests/test_icgm_regression.py -v")
        print(f"3. Commit baseline files: git add {BASELINE_DIR}")


if __name__ == '__main__':
    pytest.main([__file__, '-v', '-s'])
