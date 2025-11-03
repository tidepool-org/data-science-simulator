import os
from multiprocessing import freeze_support
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import process_simulation_data
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_simulation import run_icgm_simulations
from tidepool_data_science_simulator.utils import DATA_DIR
import shutil


if __name__ == '__main__':
    # freeze_support()
    
    TEST_PARAMS = {
        'paf_values': [0.4],
        'positive_rc_values': [True],
        'gradual_transitions_threshold_values': [10.0, 30.0, 50.0, 500.0],
        'num_vps': None,  # Use all available virtual patients
        'true_bg_values': range(40, 405, 5),  # 73 values
        'sensor_bg_values': range(40, 405, 5), # 73 values
    }

    result_dirs = run_icgm_simulations(
        base_result_dir=os.path.join(DATA_DIR, "processed"),
        **TEST_PARAMS
    )

    # Process each result directory
    for result_dir in result_dirs:
        summary_csv = process_simulation_data(result_dir)

        parent_dir = os.path.dirname(result_dir)
        dir_name = os.path.basename(result_dir.rstrip(os.sep))
        archive_base = os.path.join(parent_dir, dir_name)

        try:
            shutil.make_archive(archive_base, 'zip', root_dir=parent_dir, base_dir=dir_name)
            shutil.rmtree(result_dir)
        except Exception as e:
            print(f"Failed to archive/delete {result_dir}: {e}")
