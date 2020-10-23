__author__ = "Jason Meno"

import time
import pdb
import os
import datetime
import json
import copy
import re
from collections import defaultdict

from tidepool_data_science_simulator.models.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.run import run_simulations


def load_vp_training_data(scenarios_dir):

    file_names = os.listdir(scenarios_dir)
    all_scenario_files = [filename for filename in file_names if filename.endswith('.csv')]
    print("Num scenario files: {}".format(len(all_scenario_files)))

    vp_scenario_dict = defaultdict(dict)
    for filename in all_scenario_files:
        vp_id = re.search("train_(.*)\.csv.+", filename).groups()[0]
        bg_condition = re.search("condition(\d)", filename).groups()[0]
        vp_scenario_dict[vp_id][bg_condition] = filename

    return vp_scenario_dict


def build_icgm_sim_generator(vp_scenario_dict, sim_batch_size=30):
    """
    Build simulations for the FDA 510k Loop iCGM sensitivity analysis.

    Scenario files are on Compute-2 in Cameron Summers' copy of this code base.
    """

    n_sensors = 30
    analysis_type_list = ["temp_basal_only", "correction_bolus", "meal_bolus"]

    sim_ctr = 0
    sims = {}
    for vp_id, bg_scenario_dict in vp_scenario_dict.items():
        for bg_cond_id, scenario_filename in bg_scenario_dict.items():

            scenario_path = os.path.join(scenarios_dir, scenario_filename)
            sim_parser = ScenarioParserCSV(scenario_path)

            # Save patient properties for analysis
            vp_filename = "vp_{}.json".format(vp_id)
            vp_properties = {
                "age": sim_parser.age,
                "ylw": sim_parser.ylw,
                "patient_scenario_filename": scenario_filename
            }
            with open(os.path.join(save_dir, vp_filename), "w") as file_to_write:
                json.dump(vp_properties, file_to_write, indent=4)

            t0 = sim_parser.get_simulation_start_time()

            controller = LoopController(time=t0, controller_config=sim_parser.get_controller_config())
            controller.num_hours_history = 8  # Force 8 hours to look for boluses at start of simulation

            pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

            true_bg_trace = sim_parser.patient_glucose_history.bg_values
            sensor_generator = iCGMSensorGenerator(batch_training_size=30)
            sensor_generator.fit(true_bg_trace)
            train_percent_pass, train_loss = sensor_generator.score(true_bg_trace)
            print("Train percent pass {}. Train loss {}".format(train_percent_pass, train_loss))

            sensors = sensor_generator.generate_sensors(n_sensors, sensor_start_datetime=t0)

            for sensor_num in range(len(sensors)):

                sensor = copy.deepcopy(sensors[sensor_num])
                sensor.prefill_sensor_history(true_bg_trace[-289:])  # Load sensor with the previous 24hrs of bg data

                # Save sensor properties for analysis
                sensor_json_filename = "vp{}.bg{}.s{}.json".format(vp_id, bg_cond_id, sensor_num)
                sensor_save_path = os.path.join(save_dir, sensor_json_filename)
                sensor.serialize_properties_to_json(sensor_save_path)

                for analysis_type in analysis_type_list:
                    sim_id = "vp{}.bg{}.s{}.{}".format(
                        vp_id, bg_cond_id, sensor_num, analysis_type
                    )

                    vp = VirtualPatientISA(
                        time=t0,
                        pump=copy.deepcopy(pump),
                        sensor=copy.deepcopy(sensor),
                        metabolism_model=SimpleMetabolismModel,
                        patient_config=copy.deepcopy(sim_parser.get_patient_config()),
                        t0=t0,
                        analysis_type=analysis_type,
                    )

                    sim = Simulation(
                        time=t0,
                        duration_hrs=8.0,
                        virtual_patient=vp,
                        controller=copy.deepcopy(controller),
                        multiprocess=True,
                        sim_id=sim_id
                    )

                    sim.seed = 0
                    sims[sim_id] = sim

                    sim_ctr += 1

                    if sim_ctr == sim_batch_size:
                        yield sims
                        sims = {}
                        sim_ctr = 0

    return sims


# %%
if __name__ == "__main__":

    scenarios_dir = "data/raw/icgm-sensitivity-analysis-scenarios-2020-07-10/"

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    save_dir = "data/processed/icgm-sensitivity-analysis-results-" + today_timestamp

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Made director for results: {}".format(save_dir))

    vp_scenario_dict = load_vp_training_data(scenarios_dir)
    sim_batch_generator = build_icgm_sim_generator(vp_scenario_dict, sim_batch_size=30)

    start_time = time.time()
    for i, sim_batch in enumerate(sim_batch_generator):
        batch_start_time = time.time()

        # all_results = run_simulations(
        #     sim_batch,
        #     save_dir=save_dir,
        #     save_results=True,
        #     num_procs=30
        # )
        batch_total_time = (time.time() - batch_start_time) / 60
        run_total_time = (time.time() - start_time) / 60
        print("Batch {}".format(i))
        print("Minutes to build sim batch {} of {} sensors. Total minutes {}".format(batch_total_time, len(sim_batch), run_total_time))
