__author__ = "Mark Connolly"

import logging
logger = logging.getLogger(__name__)

import time
import os
import datetime

from tidepool_data_science_simulator.projects.icgm.icgm_sensitivity_analysis_ai_letter_June2021 import build_icgm_sim_generator
from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR

today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
result_dir = os.path.join(DATA_DIR, "processed/icgm-sensitivity-analysis-results-" + today_timestamp)

if not os.path.exists(result_dir):
    os.makedirs(result_dir)
    logger.info("Made director for results: {}".format(result_dir))

if 1:
    sim_batch_size = 10

    json_base_configs = transform_icgm_json_to_v2_parser()
    sim_batch_generator = build_icgm_sim_generator(json_base_configs, sim_batch_size=sim_batch_size)

    start_time = time.time()
    for i, sim_batch in enumerate(sim_batch_generator):

        batch_start_time = time.time()

        full_results, summary_results_df = run_simulations(
            sim_batch,
            save_dir=result_dir,
            save_results=True,
            num_procs=sim_batch_size
        )
        batch_total_time = (time.time() - batch_start_time) / 60
        run_total_time = (time.time() - start_time) / 60
        logger.info("Batch {}".format(i))
        logger.info("Minutes to build sim batch {} of {} sensors. Total minutes {}".format(batch_total_time, len(sim_batch), run_total_time))

