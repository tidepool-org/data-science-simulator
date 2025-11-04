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
    
    gradual_transitions_threshold_values = [20.0, 30.0, 40.0]
    
    for gradual_transitions_threshold in gradual_transitions_threshold_values:
        TEST_PARAMS = {
            'paf_values': [0.4],
            'positive_rc_values': [True],
            'gradual_transitions_threshold_values': [gradual_transitions_threshold],
            'vp_ids': None,  # Specific virtual patient IDs to run
            'true_bg_values': range(40, 405, 5),  # 73 values
            'sensor_bg_values': range(40, 405, 5), # 73 values
        }

        result_dir = run_icgm_simulations(
            base_result_dir=os.path.join(DATA_DIR, "processed"),
            **TEST_PARAMS
        )
        result_dir = result_dir[0]

        # Process each result directory
        summary_csv = process_simulation_data(result_dir)

        parent_dir = os.path.dirname(result_dir)
        dir_name = os.path.basename(result_dir.rstrip(os.sep))
        archive_base = os.path.join(parent_dir, dir_name)

        try:
            logger.info(f"Archiving result directory: {result_dir} -> {archive_base}.zip")
            shutil.make_archive(archive_base, 'zip', root_dir=parent_dir, base_dir=dir_name)
            logger.info(f"Archive created successfully: {archive_base}.zip")
            
            logger.info(f"Deleting result directory: {result_dir}")
            shutil.rmtree(result_dir)
            logger.info(f"Result directory deleted successfully: {result_dir}")
        except Exception as e:
            logger.error(f"Failed to archive/delete {result_dir}: {e}")
