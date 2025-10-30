import os
from multiprocessing import freeze_support
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_evaluation import process_simulation_data
from tidepool_data_science_simulator.projects.icgm.icgm_analysis_simulation import run_icgm_simulations
from tidepool_data_science_simulator.utils import DATA_DIR


if __name__ == '__main__':
    # freeze_support()
    
    TEST_PARAMS = {
        'paf_values': [0.4],
        'positive_rc_values': [True],
        'num_vps': 2 ,  # Use all available virtual patients
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
