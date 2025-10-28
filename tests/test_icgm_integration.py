"""
Simple integration test for iCGM analysis pipeline
"""
import pytest
import pandas as pd
import tempfile
import os

from tidepool_data_science_simulator.projects.icgm.icgm_analysis_simulation import run_icgm_simulations
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import (
    process_simulation_data,
    compute_score_risk_table
)


def test_icgm_pipeline_small():
    """
    Test the complete iCGM analysis pipeline with minimal data for speed.
    
    This test:
    1. Runs simulations with small BG ranges (3x3 = 9 simulations)
    2. Processes the simulation results
    3. Computes risk table
    4. Verifies outputs are reasonable
    """
    # Small ranges for testing (just 3 values each)
    true_bgs = [45, 50, 55]
    sensor_bgs = [50, 55, 60]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Step 1: Run simulations
        result_dir = run_icgm_simulations(
            paf=0.4, 
            positive_rc=False,
            result_dir=tmp_dir,
            num_vps=1,  # Just one virtual patient for speed
            true_bg_values=true_bgs,
            sensor_bg_values=sensor_bgs
        )
        
        # Verify result directory exists
        assert os.path.exists(result_dir)
        
        # Step 2: Process simulation results
        summary_csv = process_simulation_data(result_dir)
        
        # Verify summary file was created
        assert os.path.exists(summary_csv)
        assert summary_csv.endswith('.csv')
        
        # Step 3: Load and verify summary data
        summary_df = pd.read_csv(summary_csv, sep='\t')
        
        # Verify we have the expected number of simulations (3x3 = 9)
        assert len(summary_df) == 9, f"Expected 9 simulations, got {len(summary_df)}"
        
        # Verify expected columns exist
        expected_columns = [
            'sim_id', 'lbgi_icgm_start', 'lbgi_icgm_valid',
            'true_start_bg', 'start_bg_with_offset'
        ]
        for col in expected_columns:
            assert col in summary_df.columns, f"Missing column: {col}"
        
        # Verify BG values are in expected ranges
        assert summary_df['true_start_bg'].isin(true_bgs).all(), "Unexpected true BG values"
        assert summary_df['start_bg_with_offset'].isin(sensor_bgs).all(), "Unexpected sensor BG values"
        
        # Verify LBGI values are non-negative
        assert (summary_df['lbgi_icgm_start'] >= 0).all(), "LBGI start values should be non-negative"
        assert (summary_df['lbgi_icgm_valid'] >= 0).all(), "LBGI valid values should be non-negative"
        
        # Step 4: Compute risk table
        severity_df, aux_data = compute_score_risk_table(summary_df)
        
        # Verify risk table structure
        assert len(severity_df) == 5, f"Expected 5 severity bands, got {len(severity_df)}"
        assert (severity_df >= 0).all().all(), "All risk probabilities should be non-negative"
        
        # Verify auxiliary data structure
        low_icgm_axis, low_true_axis, mean_lbgi, joint_prob = aux_data
        assert len(low_icgm_axis) == len(low_true_axis), "Axis lengths should match"
        assert mean_lbgi.shape[1] == 5, "Should have 5 severity probabilities per BG combination"


def test_icgm_pipeline_with_different_parameters():
    """
    Test pipeline with different parameter combinations.
    """
    true_bgs = [100, 110]
    sensor_bgs = [105, 115]
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Test with different PAF value
        result_dir = run_icgm_simulations(
            paf=0.6,  # Different from default 0.4
            positive_rc=True,  # Enable positive RC
            result_dir=tmp_dir,
            num_vps=1,
            true_bg_values=true_bgs,
            sensor_bg_values=sensor_bgs
        )
        
        summary_csv = process_simulation_data(result_dir)
        summary_df = pd.read_csv(summary_csv, sep='\t')
        
        # Should have 2x2 = 4 simulations
        assert len(summary_df) == 4


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
