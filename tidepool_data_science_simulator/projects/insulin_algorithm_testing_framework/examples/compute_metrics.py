import os
import logging
import pandas as pd
import glob
from tidepool_data_science_simulator.projects.insulin_algorithm_testing_framework.core.metrics_calculator import (
    calculate_metrics_batch, 
    create_metrics_dataframe
)
from tidepool_data_science_simulator.utils import DATA_DIR

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
OUTPUT_DIR = '/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/510k_short_run'
BATCH_SIZE = 50  # Process files in batches to avoid memory issues

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
        batch_metrics = calculate_metrics_batch(batch_results)
        return batch_metrics
    else:
        return {}

def main():
    """Main function to compute metrics for all simulation results."""
    
    logger.info("Starting metrics computation...")
    
    # Find all TSV files in the output directory
    tsv_pattern = os.path.join(OUTPUT_DIR, "*.tsv")
    tsv_files = glob.glob(tsv_pattern)
    
    logger.info(f"Found {len(tsv_files)} TSV files to process")
    
    if not tsv_files:
        logger.error(f"No TSV files found in {OUTPUT_DIR}")
        return
    
    # Process files in batches
    all_metrics = {}
    total_processed = 0
    
    for i in range(0, len(tsv_files), BATCH_SIZE):
        batch_files = tsv_files[i:i + BATCH_SIZE]
        logger.info(f"Processing batch {i//BATCH_SIZE + 1}/{(len(tsv_files)-1)//BATCH_SIZE + 1} ({len(batch_files)} files)")
        
        # Process this batch using functional interface
        batch_metrics = process_batch(batch_files)
        
        # Add to overall results
        all_metrics.update(batch_metrics)
        total_processed += len(batch_metrics)
        
        logger.info(f"Processed {len(batch_metrics)} simulations in this batch. Total: {total_processed}")
    
    # Create metrics DataFrame using functional interface
    if all_metrics:
        logger.info("Creating metrics DataFrame...")
        metrics_df = create_metrics_dataframe(all_metrics)
        
        if metrics_df.empty:
            raise ValueError("No metrics calculated")
        
        # Save results to parent directory
        parent_dir = os.path.dirname(OUTPUT_DIR)
        output_file = os.path.join(parent_dir, "metrics_results.csv")
        
        logger.info(f"Saving metrics to {output_file}")
        metrics_df.to_csv(output_file, index=False)
        
        logger.info(f"Successfully calculated metrics for {len(all_metrics)} simulations")
        logger.info(f"Metrics columns: {list(metrics_df.columns)}")
        logger.info(f"Results saved to: {output_file}")
        
        # Print summary statistics
        logger.info("\n=== SUMMARY STATISTICS ===")
        if 'time_in_range_70_180' in metrics_df.columns:
            logger.info(f"Mean Time in Range (70-180): {metrics_df['time_in_range_70_180'].mean():.2f}%")
        if 'time_below_70' in metrics_df.columns:
            logger.info(f"Mean Time Below 70: {metrics_df['time_below_70'].mean():.2f}%")
        if 'mean_glucose' in metrics_df.columns:
            logger.info(f"Mean Glucose: {metrics_df['mean_glucose'].mean():.2f} mg/dL")
        
    else:
        logger.error("No metrics calculated - no valid simulation results found")

if __name__ == "__main__":
    main()
