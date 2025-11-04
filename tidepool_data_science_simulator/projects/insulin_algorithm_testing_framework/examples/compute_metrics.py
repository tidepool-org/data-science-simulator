import numpy as np
import os
import logging
import pandas as pd
import glob
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import (
    calculate_metrics_batch, 
    create_point_metrics_dataframe,
    save_timeseries_metrics
)
from tidepool_data_science_simulator.utils import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - Set these to control which metrics to calculate
CALCULATE_POINT_METRICS = True      # Calculate scalar metrics (CSV output)
CALCULATE_TIMESERIES_METRICS = False # Calculate timeseries metrics (.npy output)

DEBUG_BREAK = False  # Set to True to break after first batch for debugging

# Other configuration
OUTPUT_DIR = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/unannounced_meals/510k_short_run'
OUTPUT_DIR = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/510k_short_run_example_mitigation/'
OUTPUT_DIR = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/510k_short_run_example_mitigation_new'

BATCH_SIZE = 500  # Process files in batches to avoid memory issues

def load_simulation_results(tsv_file):
    """Load simulation results from TSV file."""
    try:
        # Extract sim_id from filename (remove .tsv extension)
        sim_id = os.path.basename(tsv_file)[:-4]
        
        # Load the TSV file
        df = pd.read_csv(tsv_file, sep='\t')
        
        # Convert time column to datetime if needed
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
        
        return sim_id, df
    except Exception as e:
        logger.error(f"Error loading {tsv_file}: {e}")
        return None, None

def process_batch(tsv_files):
    """Process a batch of TSV files and calculate metrics."""
    batch_results = {}
    
    for tsv_file in tsv_files:
        sim_id, df = load_simulation_results(tsv_file)
        if sim_id is not None and df is not None:
            batch_results[sim_id] = df
            # logger.info(f"Loaded {sim_id}")
    
    # Calculate metrics for this batch using functional interface
    if batch_results:
        point_metrics, timeseries_metrics = calculate_metrics_batch(batch_results, 
            calculate_point_metrics_bool=CALCULATE_POINT_METRICS,
            calculate_timeseries_metrics_bool=CALCULATE_TIMESERIES_METRICS
        )
        
        # Return only the requested metric types
        if not CALCULATE_POINT_METRICS:
            point_metrics = {}
        if not CALCULATE_TIMESERIES_METRICS:
            timeseries_metrics = {}
            
        return point_metrics, timeseries_metrics
    else:
        return {}, {}

def main():
    """Main function to compute metrics for all simulation results."""
    
    # Validate configuration
    if not CALCULATE_POINT_METRICS and not CALCULATE_TIMESERIES_METRICS:
        logger.error("At least one of CALCULATE_POINT_METRICS or CALCULATE_TIMESERIES_METRICS must be True")
        return
    
    logger.info("Starting metrics computation...")
    logger.info(f"Configuration: Point metrics: {CALCULATE_POINT_METRICS}, Timeseries metrics: {CALCULATE_TIMESERIES_METRICS}")
    
    # Find all TSV files in the output directory
    tsv_pattern = os.path.join(OUTPUT_DIR, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    logger.info(f"Found {len(tsv_files)} TSV files to process")
    
    if not tsv_files:
        logger.error(f"No TSV files found in {OUTPUT_DIR}")
        return
    
    # Process files in batches
    all_point_metrics = {}
    all_timeseries_metrics = {}
    total_processed = 0
    
    for i in range(0, len(tsv_files), BATCH_SIZE):
        batch_files = tsv_files[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(tsv_files)-1)//BATCH_SIZE + 1} ({len(batch_files)} files)")
        
        # Process this batch using functional interface
        batch_point_metrics, batch_timeseries_metrics = process_batch(batch_files)
        
        # Add to overall results
        all_point_metrics.update(batch_point_metrics)
        all_timeseries_metrics.update(batch_timeseries_metrics)
        total_processed += len(batch_point_metrics)
        
        logger.info(f"Processed {len(batch_point_metrics)} simulations in this batch. Total: {total_processed}")
        if DEBUG_BREAK:
            break
    
    # Save results using new separated approach
    parent_dir = os.path.dirname(OUTPUT_DIR)
    point_metrics_df = None
    
    # Save point metrics if enabled and available
    if CALCULATE_POINT_METRICS and all_point_metrics:
        logger.info("Creating point metrics DataFrame...")
        point_metrics_df = create_point_metrics_dataframe(all_point_metrics)
        
        if point_metrics_df.empty:
            raise ValueError("No point metrics calculated")
        
        # Save point metrics as CSV
        point_metrics_file = os.path.join(parent_dir, "point_metrics.csv")
        logger.info(f"Saving point metrics to {point_metrics_file}")
        point_metrics_df.to_csv(point_metrics_file, index=False)
        
        logger.info(f"Point metrics columns: {list(point_metrics_df.columns)}")
        logger.info(f"Point metrics saved to: {point_metrics_file}")
        
        # Print summary statistics
        logger.info("\n=== POINT METRICS SUMMARY ===")
        if 'time_in_range_70_180' in point_metrics_df.columns:
            logger.info(f"Mean Time in Range (70-180): {point_metrics_df['time_in_range_70_180'].mean():.2f}%")
        if 'time_below_70' in point_metrics_df.columns:
            logger.info(f"Mean Time Below 70: {point_metrics_df['time_below_70'].mean():.2f}%")
        if 'mean_glucose' in point_metrics_df.columns:
            logger.info(f"Mean Glucose: {point_metrics_df['mean_glucose'].mean():.2f} mg/dL")
    
    # Save timeseries metrics if enabled and available
    if CALCULATE_TIMESERIES_METRICS and all_timeseries_metrics:
        logger.info("Saving timeseries metrics...")
        
        # Get simulation IDs - use point metrics order if available, otherwise use timeseries keys
        if point_metrics_df is not None:
            sim_ids = point_metrics_df['simulation_id'].tolist()
        else:
            sim_ids = list(all_timeseries_metrics.keys())
            
        timeseries_file = save_timeseries_metrics(
            all_timeseries_metrics, 
            sim_ids, 
            parent_dir, 
            "cumulative_sum_insulin"
        )
        logger.info(f"Timeseries metrics saved to: {timeseries_file}")
    
    # Final summary
    total_processed = len(all_point_metrics) if CALCULATE_POINT_METRICS else len(all_timeseries_metrics)
    if total_processed > 0:
        logger.info(f"Successfully processed {total_processed} simulations")
        metrics_calculated = []
        if CALCULATE_POINT_METRICS and all_point_metrics:
            metrics_calculated.append("point metrics")
        if CALCULATE_TIMESERIES_METRICS and all_timeseries_metrics:
            metrics_calculated.append("timeseries metrics")
        logger.info(f"Calculated: {', '.join(metrics_calculated)}")
    else:
        logger.error("No metrics calculated - no valid simulation results found")

if __name__ == "__main__":
    main()
