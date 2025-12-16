__author__ = "Shawn Foster"
"""
Check severity level based on glucose values in simulation output.

This script recursively analyzes TSV files in a directory containing simulation results 
and determines the severity level based on:
1. Presence of any glucose values ≤ 0 mg/dL
2. Presence of ≥ 48 consecutive glucose values ≤ 40 mg/dL

If either condition is met, the simulation is classified as Catastrophic.
Otherwise, it is classified as Critical, if the LBGI risk score is a 4.

Results are written to a CSV file with columns: filename, result. The CSV file is stored in the results directory.
"""

import argparse
import sys
import os
from pathlib import Path
from collections import defaultdict
import pandas as pd


def check_consecutive_low_values(bg_series, threshold=40, min_consecutive=48):
    """
    Check if there are at least min_consecutive values ≤ threshold in a row.
    
    Args:
        bg_series: pandas Series containing glucose values
        threshold: glucose threshold value (default: 40)
        min_consecutive: minimum number of consecutive values needed (default: 48)
    
    Returns:
        bool: True if condition is met, False otherwise
    """
    consecutive_count = 0
    max_consecutive = 0
    
    for value in bg_series:
        if pd.notna(value) and value <= threshold:
            consecutive_count += 1
            max_consecutive = max(max_consecutive, consecutive_count)
            if consecutive_count >= min_consecutive:
                return True
        else:
            consecutive_count = 0
    
    return False


def check_severity_level(tsv_path):
    """
    Analyze TSV file and determine severity level.
    
    Args:
        tsv_path: path to the TSV file to analyze
    
    Returns:
        str: severity level message
    """
    # Read the TSV file
    try:
        df = pd.read_csv(tsv_path, sep='\t')
    except Exception as e:
        return f"Error reading file: {e}"
    
    # Check if 'bg' column exists
    if 'bg' not in df.columns:
        return "Error: 'bg' column not found in TSV file"
    
    bg_series = df['bg']
    
    # Condition 1: Check for any values ≤ 0
    has_zero_or_negative = (bg_series <= 0).any()
    
    # Condition 2: Check for ≥ 48 consecutive values ≤ 40
    has_extended_low = check_consecutive_low_values(bg_series, threshold=40, min_consecutive=48)
    
    # Determine severity level
    if has_zero_or_negative or has_extended_low:
        return "Simulation meets Catastrophic severity criteria."
    else:
        return "Simulation meets Critical severity criteria if LBGI risk score is 4."


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Check severity level of simulations based on glucose values'
    )
    parser.add_argument(
        'directory',
        help='Path to the directory containing TSV files to analyze'
    )
    
    args = parser.parse_args()
    
    # Get directory path
    directory = Path(args.directory)
    
    # Check if directory exists
    if not directory.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)
    
    if not directory.is_dir():
        print(f"Error: '{directory}' is not a directory")
        sys.exit(1)
    
    # Find all TSV files recursively in the directory
    tsv_files = list(directory.rglob('*.tsv'))
    
    # Check if any TSV files were found
    if not tsv_files:
        print("No TSV files in selected directory")
        return
    
    # Group TSV files by their parent directory
    files_by_directory = defaultdict(list)
    for tsv_file in tsv_files:
        files_by_directory[tsv_file.parent].append(tsv_file)
    
    # Process each directory
    csv_count = 0
    for dir_path in sorted(files_by_directory.keys()):
        tsv_files_in_dir = files_by_directory[dir_path]
        
        # Process each TSV file in this directory and collect results
        results = []
        for tsv_file in tsv_files_in_dir:
            result = check_severity_level(tsv_file)
            results.append({
                'filename': tsv_file.name,
                'result': result
            })
        
        # Create DataFrame and sort alphabetically by filename
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('filename').reset_index(drop=True)
        
        # Write to CSV in the same directory as the TSV files
        output_path = dir_path / 'severity_results.csv'
        results_df.to_csv(output_path, index=False)
        
        csv_count += 1
        print(f"Results written to {output_path}")
    
    print(f"\nTotal: {csv_count} CSV file(s) created")


if __name__ == '__main__':
    main()
