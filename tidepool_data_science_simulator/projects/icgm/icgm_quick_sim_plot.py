import os
import logging
import glob
from multiprocessing import freeze_support
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import process_simulation_data
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_simulation import run_icgm_simulations
from tidepool_data_science_simulator.utils import DATA_DIR
from tidepool_data_science_simulator.visualization.sim_viz import load_and_plot_tsv
import shutil

logger = logging.getLogger(__name__)


if __name__ == '__main__':
    # freeze_support()
    
    gradual_transitions_threshold_values = [20.0]
    
    for gradual_transitions_threshold in gradual_transitions_threshold_values:
        TEST_PARAMS = {
            'paf_values': [0.4],
            'positive_rc_values': [True],
            'gradual_transitions_threshold_values': [gradual_transitions_threshold],
            'vp_ids': [34],  # Specific virtual patient IDs to run
            'true_bg_values': range(40, 45, 5),  # 73 values
            'sensor_bg_values': range(100, 105, 5), # 73 values
        }

        result_dir = run_icgm_simulations(
            base_result_dir=os.path.join(DATA_DIR, "processed"),
            **TEST_PARAMS
        )
        result_dir = result_dir[0]
        
        # Load and plot TSV files from result directory
        tsv_files = glob.glob(os.path.join(result_dir, "*.tsv"))
        logger.info(f"Found {len(tsv_files)} TSV files in {result_dir}")
        
        # Plot the first TSV file (or you can iterate through all)
        if tsv_files:
            logger.info(f"Plotting first TSV file: {tsv_files[0]}")
            load_and_plot_tsv(tsv_files[0])
        
       
