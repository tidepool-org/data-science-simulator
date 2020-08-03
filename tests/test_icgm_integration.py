__author__ = "Jason Meno"

import os
import datetime

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.patient.virtual_patient import VirtualPatient
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensorGenerator, sf
from tidepool_data_science_simulator.models.pump import Omnipod, ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.simulation import Simulation


TEST_DATETIME = datetime.datetime(year=2020, month=1, day=1)
# %% Tests


def test_basic_sensor_generator():
    """Verify that the main sensor generator works within the simulator environment"""
    true_bg_trace = sf.generate_test_bg_trace()
    sensor_generator = iCGMSensorGenerator(batch_training_size=3)
    sensor_generator.fit(true_bg_trace)

    # Create 100 sensors
    sensors = sensor_generator.generate_sensors(100, sensor_start_datetime=TEST_DATETIME)

    # Backfill all sensors with the first day true_bg_traces
    for sensor in sensors:
        sensor.prefill_sensor_history(true_bg_trace[-289:])
        assert sensor.time_index == 289


def test_virtual_patient_icgm_integration():
    scenario_csv_filepath = "tests/data/Scenario-0-simulation-template - inputs.tsv"
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

    sensor_generator = iCGMSensorGenerator(batch_training_size=3)
    sensor_generator.fit(true_bg_trace=sim_parser.patient_glucose_history.bg_values)
    sensor = sensor_generator.generate_sensors(1, sensor_start_datetime=t0)[0]
    sensor_prefill_data = sim_parser.patient_glucose_history.bg_values[:-1]
    sensor.prefill_sensor_history(
        true_bg_history=sensor_prefill_data,
    )

    vp = VirtualPatient(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=sim_parser.get_patient_config(),
    )

    # TODO: Instantiate patient with function instead of from file
    # vp = get_canonical_risk_patient(
    #     t0=None, patient_class=VirtualPatient, pump_class=pump, sensor_class=sensor
    # )

    assert vp.sensor.current_datetime == t0
    assert vp.sensor.time_index == len(sensor_prefill_data)


def test_all_controllers_icgm_integration():
    scenario_csv_filepath = "tests/data/Scenario-0-simulation-template - inputs.tsv"
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    controllers = [
        DoNothingController(time=t0, controller_config=sim_parser.get_controller_config()),
        LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config(),
        ),
    ]

    sensor_generator = iCGMSensorGenerator(batch_training_size=3)
    sensor_generator.fit(true_bg_trace=sim_parser.patient_glucose_history.bg_values)


    all_results = {}
    for controller in controllers:
        sim_id = controller.name
        print("Running: {}".format(sim_id))

        pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())

        sensor = sensor_generator.generate_sensors(1, sensor_start_datetime=t0)[0]
        sensor_prefill_data = sim_parser.patient_glucose_history.bg_values[:-1]
        sensor.prefill_sensor_history(
            true_bg_history=sensor_prefill_data,
        )

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
        )

        simulation = Simulation(
            time=t0,
            duration_hrs=8.0,
            virtual_patient=vp,
            controller=controller,
        )

        simulation.run()

        results_df = simulation.get_results_df()
        all_results[sim_id] = results_df
