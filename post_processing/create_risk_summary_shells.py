#!/usr/bin/env python3
"""
Create RTF shell documents for Tidepool Risk Severity Evaluation results.

This script generates placeholder RTF documents for each TLR-* subdirectory
in a simulation results directory, ready to be populated with actual analysis results.
"""

import os
import json
import glob
import pandas as pd
from datetime import datetime
from pathlib import Path


def extract_metric_data(tlr_dir, column_name):
    """
    Extract metric data from all CSV files in a TLR directory.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        column_name: Name of the column to extract (e.g., 'percent_values_ge_70_le_180')
        
    Returns:
        Dictionary with keys 'pre', 'no_loop', 'post', each containing a list of metric values
    """
    metric_data = {
        'pre': [],
        'no_loop': [],
        'post': []
    }
    
    # Find all summary results CSV files
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    
    if not csv_files:
        print(f"  Warning: No CSV files found in {tlr_dir}")
        return metric_data
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            
            # Check if required columns exist
            if 'sim_id' not in df.columns or column_name not in df.columns:
                print(f"  CSV file malformed; check data configuration: {csv_file}")
                continue
            
            # Extract data for each stage
            for _, row in df.iterrows():
                sim_id = row['sim_id']
                metric_value = row[column_name]
                
                # Pre-mitigation
                if sim_id.startswith('pre-Loop_NoMitigations_') or sim_id.startswith('pre-Loop-NoMitigations_'):
                    metric_data['pre'].append(metric_value)
                
                # No Loop
                elif sim_id.startswith('pre-noLoop_') or sim_id.startswith('pre-NoLoop_'):
                    metric_data['no_loop'].append(metric_value)
                
                # Post-mitigation
                elif sim_id.startswith('post-Loop-WithMitigations_') or sim_id.startswith('post-LoopWithMitigations_'):
                    metric_data['post'].append(metric_value)
        
        except Exception as e:
            print(f"  CSV file malformed; check data configuration: {csv_file}")
            print(f"  Error details: {e}")
            continue
    
    return metric_data


def calculate_stage_averages(metric_data):
    """
    Calculate average metric values for each evaluation stage.
    
    Args:
        metric_data: Dictionary with metric values for each stage
        
    Returns:
        Dictionary with formatted average strings for each stage
    """
    averages = {}
    
    for stage in ['pre', 'no_loop', 'post']:
        values = metric_data[stage]
        if values:
            avg = sum(values) / len(values)
            averages[stage] = f"{avg:.1f}"
        else:
            averages[stage] = "NA"
    
    return averages


def count_profiles(tlr_dir):
    """
    Count the number of profile CSV files in a TLR directory.
    
    Args:
        tlr_dir: Path to TLR subdirectory
        
    Returns:
        Integer count of CSV files
    """
    csv_files = glob.glob(os.path.join(tlr_dir, 'summary_results_*.csv'))
    return len(csv_files)


def create_rtf_shell(subdirectory_name, timestamp, tir_averages, tar_averages, profile_count, output_path):
    """
    Create an RTF shell document with metric data populated.
    
    Args:
        subdirectory_name: The full subdirectory name (e.g., "TLR-1119_bike" or "TLR-1119")
        timestamp: Simulation run timestamp
        tir_averages: Dictionary with TIR average values for each stage
        tar_averages: Dictionary with TAR average values for each stage
        profile_count: Number of virtual patient profiles
        output_path: Full path where the RTF file should be saved
    """
    
    # Format timestamp: YYYY-MM-DD HH:MM:SS
    formatted_timestamp = timestamp.replace('T', ' ').split('.')[0]
    
    rtf_content = r"""{\rtf1\ansi\deff0
{\fonttbl{\f0 Arial;}}
\f0\fs24

{\b\fs28 Risk severity summary for """ + subdirectory_name + r"""}
\par\par

{\b Date and time of simulation run:} """ + formatted_timestamp + r"""
\par\par

Auto-generated output from Tidepool Risk Severity Evaluation Simulator Tool
\par\par

{\b Table of results}
\par\par

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl {\b Evaluation stage}\cell
\pard\intbl {\b Harm}\cell
\pard\intbl {\b Severity}\cell
\pard\intbl {\b TIR % (70 - 180 mg/dL)}\cell
\pard\intbl {\b TBR % (<54 mg/dL}\cell
\pard\intbl {\b TAR % (>180 mg/dL)}\cell
\row

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl Pre-mitigation\cell
\pard\intbl TBD\cell
\pard\intbl TBD\cell
\pard\intbl """ + tir_averages['pre'] + r"""\cell
\pard\intbl TBD\cell
\pard\intbl """ + tar_averages['pre'] + r"""\cell
\row

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl No Loop\cell
\pard\intbl TBD\cell
\pard\intbl TBD\cell
\pard\intbl """ + tir_averages['no_loop'] + r"""\cell
\pard\intbl TBD\cell
\pard\intbl """ + tar_averages['no_loop'] + r"""\cell
\row

\trowd
\cellx1700\cellx3400\cellx5100\cellx6800\cellx8500\cellx10200
\pard\intbl Post-mitigation\cell
\pard\intbl TBD\cell
\pard\intbl TBD\cell
\pard\intbl """ + tir_averages['post'] + r"""\cell
\pard\intbl TBD\cell
\pard\intbl """ + tar_averages['post'] + r"""\cell
\row

\pard
\par\par

""" + str(profile_count) + r""" virtual patient profiles aggregated for this summary.
\par\par

{\b Critical/Catastrophic Identifier}
\par\par

TBD
\par\par

{\b Outlier Results}
\par\par

TBD

\pard
}"""
    
    with open(output_path, 'w') as f:
        f.write(rtf_content)
    
    print(f"Created shell document: {output_path}")


def extract_simulation_id(summary_file_path):
    """
    Extract simulation ID from summary results filename.
    
    Args:
        summary_file_path: Path to summary results file
        
    Returns:
        Simulation ID (e.g., "TLR-1119")
    """
    filename = os.path.basename(summary_file_path)
    # Extract from pattern: summary_results_Simulation-Configuration-TLR-XXXX
    parts = filename.split('-')
    # Find index of 'TLR' and take TLR-XXXX
    for i, part in enumerate(parts):
        if part == 'TLR':
            # Get TLR and the number after it
            sim_id = f"TLR-{parts[i+1].split('.')[0].split('_')[0]}"
            return sim_id
    return None


def process_results_directory(results_dir):
    """
    Process a results directory and create shell RTF documents for each TLR subdirectory.
    
    Args:
        results_dir: Path to the results directory
    """
    
    # Read metadata.json for timestamp
    metadata_path = os.path.join(results_dir, 'metadata.json')
    if not os.path.exists(metadata_path):
        print(f"Error: metadata.json not found in {results_dir}")
        return
    
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    # Extract timestamp - adjust key name as needed based on actual metadata structure
    timestamp = metadata.get('timestamp', metadata.get('run_timestamp', 'Unknown'))
    
    # Find all TLR-* subdirectories
    tlr_dirs = glob.glob(os.path.join(results_dir, 'TLR-*'))
    
    if not tlr_dirs:
        print(f"No TLR-* subdirectories found in {results_dir}")
        return
    
    print(f"Found {len(tlr_dirs)} TLR subdirectories")
    
    for tlr_dir in tlr_dirs:
        print(f"\nProcessing: {tlr_dir}")
        
        # Find summary results files
        summary_files = glob.glob(
            os.path.join(tlr_dir, 'summary_results_Simulation-Configuration-TLR*.csv')
        )
        
        if not summary_files:
            print(f"  Warning: No summary results files found in {tlr_dir}")
            continue
        
        # Extract simulation ID from first summary file
        simulation_id = extract_simulation_id(summary_files[0])
        
        if not simulation_id:
            print(f"  Error: Could not extract simulation ID from {summary_files[0]}")
            continue
        
        print(f"  Simulation ID: {simulation_id}")
        
        # Extract TIR data from CSV files
        tir_data = extract_metric_data(tlr_dir, 'percent_values_ge_70_le_180')
        tir_averages = calculate_stage_averages(tir_data)
        
        # Extract TAR data from CSV files
        tar_data = extract_metric_data(tlr_dir, 'percent_cgm_gt_180')
        tar_averages = calculate_stage_averages(tar_data)
        
        # Count profiles
        profile_count = count_profiles(tlr_dir)
        
        print(f"  Profile count: {profile_count}")
        print(f"  TIR averages: Pre={tir_averages['pre']}, No Loop={tir_averages['no_loop']}, Post={tir_averages['post']}")
        print(f"  TAR averages: Pre={tar_averages['pre']}, No Loop={tar_averages['no_loop']}, Post={tar_averages['post']}")
        
        # Get subdirectory name for the header
        subdirectory_name = os.path.basename(tlr_dir)
        
        # Create RTF shell document with metric data
        output_path = os.path.join(tlr_dir, f"risk_summary_{simulation_id}.rtf")
        create_rtf_shell(subdirectory_name, timestamp, tir_averages, tar_averages, profile_count, output_path)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Create RTF shell documents for risk severity evaluation results'
    )
    parser.add_argument(
        'results_dir',
        help='Path to results directory (e.g., Risk_Run_2025-10-10T14:30:55.391233)'
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.results_dir):
        print(f"Error: Directory not found: {args.results_dir}")
        return 1
    
    process_results_directory(args.results_dir)
    print("\nDone!")
    return 0


if __name__ == '__main__':
    exit(main())
