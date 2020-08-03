__author__ = "Jason Meno"

import copy
from tidepool_data_science_simulator.models.patient.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator
import time
import os
import datetime
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
# %% Tests

def run_icgm_sensitivity_analysis():
    """Tests the an entire icgm sensitivity analysis on one scenario file"""
    sim_start = time.time()
    sim_seed = 0
    save_results = True
    use_multiprocess = True
    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    all_scenario_path = "data/raw/icgm-sensitivity-analysis-scenarios-2020-07-10/"
    file_names = os.listdir(all_scenario_path)
    all_scenario_files = [filename for filename in file_names if filename.endswith('.csv')]
    virtual_patient_list = list(set([file[:-5] for file in all_scenario_files]))
    save_dir = "data/processed/icgm-sensitivity-analysis-results-" + today_timestamp

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    num_virtual_patients = 1
    bg_test_conditions = 1
    n_sensors = 30
    analysis_type_list = ["temp_basal_only", "correction_bolus", "meal_bolus"]

    # all_results = {}
    for virtual_patient_num in range(num_virtual_patients):
        for bg_test_condition in range(1, bg_test_conditions+1):
            vp_name = virtual_patient_list[virtual_patient_num]
            scenario_path = os.path.join(all_scenario_path, vp_name + "{}.csv".format(bg_test_condition))

            if os.path.exists(scenario_path):
                sim_parser = ScenarioParserCSV(scenario_path)
                t0 = sim_parser.get_simulation_start_time()

                controller = LoopController(time=t0, controller_config=sim_parser.get_controller_config())
                controller.num_hours_history = 8  # Force 8 hours to look for boluses at start of simulation

                pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

                true_bg_trace = sim_parser.patient_glucose_history.bg_values
                sensor_generator = iCGMSensorGenerator(batch_training_size=30)
                sensor_generator.fit(true_bg_trace)
                sensors = sensor_generator.generate_sensors(n_sensors, sensor_start_datetime=t0)

                sims = {}

                for sensor_num in range(len(sensors)):
                    sensor = sensors[sensor_num]
                    sensor.prefill_sensor_history(true_bg_trace[-289:])  # Load sensor with the previous 24hrs of bg data

                    for analysis_type in analysis_type_list:
                        simulation_name = "vp{}.bg{}.s{}.{}".format(
                            virtual_patient_num, bg_test_condition, sensor_num, analysis_type
                        )
                        print("Starting: " + simulation_name)

                        vp = VirtualPatientISA(
                            time=t0,
                            pump=copy.deepcopy(pump),
                            sensor=copy.deepcopy(sensor),
                            metabolism_model=SimpleMetabolismModel,
                            patient_config=sim_parser.get_patient_config(),
                            t0=t0,
                            analysis_type=analysis_type,
                        )

                        simulation = Simulation(
                            time=t0, duration_hrs=8.0, virtual_patient=vp, controller=copy.deepcopy(controller), multiprocess=use_multiprocess,
                        )

                        simulation.seed = sim_seed  # set here for multiprocessing

                        sims[simulation_name] = simulation
                        if use_multiprocess:
                            simulation.start()
                        else:
                            simulation.run()

                if use_multiprocess:
                    all_results = {id: sim.queue.get() for id, sim in sims.items()}
                    [sim.join() for id, sim in sims.items()]
                else:
                    all_results = {id: sim.get_results_df() for id, sim in sims.items()}

                # Save all results
                for sim_id, results_df in all_results.items():
                    if save_results:
                        results_df.to_csv(os.path.join(save_dir, sim_id + ".csv"))

    sim_end = time.time()
    sim_time = round(sim_end - sim_start, 4)
    expected_result_size = num_virtual_patients * bg_test_conditions * n_sensors * len(analysis_type_list)
    print(str(expected_result_size) + " Simulations Completed in " + str(sim_time) + " Seconds")

# %%
if __name__ == "__main__":
    run_icgm_sensitivity_analysis()
