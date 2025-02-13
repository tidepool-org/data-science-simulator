__author__ = "Mark Connolly"

from itertools import product
import logging
import pandas as pd
import numexpr
logger = logging.getLogger(__name__)

import time
import os
import datetime
import copy
from collections import defaultdict

import numpy as np
from numpy.random import RandomState


from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor_icgm import (
    CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, 
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import Bolus, Carb, TargetRange

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.makedata.make_icgm_patients import transform_icgm_json_to_v2_parser
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.utils import DATA_DIR


def generate_autobolus_simulations(json_sim_base_config, simulation_configurations, base_sim_seed):
    """
    Generator simulations from a base configuration that have different true bg
    starting points and different t0 sensor error values.
    """
    num_history_values = len(json_sim_base_config["patient"]["sensor"]["glucose_history"]["value"])


    random_state = RandomState(base_sim_seed)

    for index, simulation_configuration in simulation_configurations.iterrows():

        initial_blood_glucose = simulation_configuration['initial_blood_glucose']
        partial_application_factor = simulation_configuration['partial_application_factor']
        
        new_sim_base_config = copy.deepcopy(json_sim_base_config)    

        date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"
        t0 = datetime.datetime.strptime(new_sim_base_config["time_to_calculate_at"], date_str_format)

        glucose_history_values = {i: initial_blood_glucose for i in range(num_history_values)}

        new_sim_base_config["patient"]["sensor"]["glucose_history"]["value"] = glucose_history_values
        new_sim_base_config["patient"]["patient_model"]["glucose_history"]["value"] = glucose_history_values        

        new_sim_base_config["controller"]["id"] = 'swift'
        new_sim_base_config["controller"]["settings"]["include_positive_velocity_and_RC"] = True
        new_sim_base_config["controller"]["settings"]["use_mid_absorption_isf"] = True

        true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])
        reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])            
        
        sim_parser = ScenarioParserV2()
        sim_start_time, duration_hrs, virtual_patient, controller = sim_parser.build_components_from_config(new_sim_base_config)
        
        virtual_patient.carb_event_timeline = true_carb_timeline
        # virtual_patient.pump.carb_event_timeline = reported_carb_timeline
        
        controller.controller_config.controller_settings['minimum_autobolus'] = 0.1
        controller.controller_config.controller_settings['maximum_autobolus'] = 100
        controller.controller_config.controller_settings['partial_application_factor'] = partial_application_factor

        patient_id = new_sim_base_config['patient_id']
        sim_id="vp={}_patient_id={}_withRC_paf={}_ibg={}".format(base_sim_seed, patient_id, partial_application_factor, initial_blood_glucose)
        
        sim = Simulation(sim_start_time,
                            duration_hrs=duration_hrs,
                            virtual_patient=virtual_patient,
                            controller=controller,
                            multiprocess=True,
                            sim_id=sim_id
                            )

        sim.random_state = random_state

        yield sim


def build_autobolus_sim_generator(json_base_configs, sim_batch_size=30):
    """
    Build simulations for the FDA AI Letter iCGM sensitivity analysis.
    """
    for i, json_config in enumerate(json_base_configs, 1):
        logger.info("VP: {}. {} of {}".format(json_config["patient_id"], i, len(json_base_configs)))

        sim_ctr = 0
        sims = {}

        partial_application_factors = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
        partial_application_factors = [0.4]
        true_start_glucose_list = range(140, 141, 10)

        simulation_configurations = list(product(partial_application_factors, true_start_glucose_list))
        simulation_configurations = pd.DataFrame(simulation_configurations, columns=['partial_application_factor', 'initial_blood_glucose'])
        
        for sim in generate_autobolus_simulations(json_config, simulation_configurations=simulation_configurations, base_sim_seed=i):

            sims[sim.sim_id] = sim
            sim_ctr += 1

            if sim_ctr == sim_batch_size:
                yield sims
                sims = {}
                sim_ctr = 0

        if sims:
            yield sims


if __name__ == "__main__":

    today_timestamp = datetime.datetime.now().strftime(r"%Y_%m_%d_T_%H_%M_%S")
    result_dir = os.path.join(DATA_DIR, "processed/no_meal_announcements_" + today_timestamp)
    os.environ['NUMEXPR_MAX_THREADS'] = str(14)
    numexpr.set_num_threads(14)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
        logger.info("Made directory for results: {}".format(result_dir))


    sim_batch_size = 1

    json_base_configs = transform_icgm_json_to_v2_parser()
    sim_batch_generator = build_autobolus_sim_generator(json_base_configs, sim_batch_size=sim_batch_size)

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

        plot_sim_results(full_results)

        logger.info("Batch {}".format(i))
        logger.info("Minutes to build sim batch {} of {} sensors. Total minutes {}".format(batch_total_time, len(sim_batch), run_total_time))
