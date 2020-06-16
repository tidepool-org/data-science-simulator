__author__ = "Jason Meno"

import copy
from tidepool_data_science_simulator.models.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator

# %% Tests


def test_icgm_sensitivity_analysis():
    """Tests the an entire icgm sensitivity analysis on one scenario file"""

    num_virtual_patients = 1
    bg_test_conditions = [3]
    n_sensors = 3
    analysis_type_list = ["temp_basal_only", "correction_bolus", "meal_bolus"]

    all_results = {}
    for virtual_patient_num in range(num_virtual_patients):
        for bg_test_condition in bg_test_conditions:
            scenario_path = "tests/data/scenario-icgm-sensitivity-analysis_condition{}.csv".format(bg_test_condition)
            sim_parser = ScenarioParserCSV(scenario_path)
            t0 = sim_parser.get_simulation_start_time()

            controller = LoopController(time=t0, controller_config=sim_parser.get_controller_config())
            controller.num_hours_history = 8  # Force 8 hours to look for boluses at start of simulation

            pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

            true_bg_trace = sim_parser.patient_glucose_history.bg_values
            sensor_generator = iCGMSensorGenerator(batch_training_size=30)
            sensor_generator.fit(true_bg_trace)
            sensors = sensor_generator.generate_sensors(n_sensors)

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
                        time=t0, duration_hrs=8.0, virtual_patient=vp, controller=copy.deepcopy(controller),
                    )

                    simulation.run()

                    results_df = simulation.get_results_df()
                    all_results[simulation_name] = results_df

    expected_result_size = num_virtual_patients * len(bg_test_conditions) * n_sensors * len(analysis_type_list)
    assert len(all_results) == expected_result_size
