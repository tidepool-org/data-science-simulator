import os
import datetime
import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import Omnipod
from tidepool_data_science_simulator.models.sensor import NoisySensor, iCGMSensorGenerator
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV


def do_icgm_risk_analysis(scenario_csv_filepath):

    sim_parser = ScenarioParserCSV(scenario_csv_filepath)

    t0 = sim_parser.get_simulation_start_time()
    controller = LoopController(
        time=t0,
        loop_config=sim_parser.get_controller_config(),
        simulation_config=sim_parser.get_simulation_config(),
    )

    pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())

    num_sensors = 30
    sensor_generator = iCGMSensorGenerator(sensor_config=sim_parser.get_sensor_config())

    virtual_patients = [
        VirtualPatient(
            time=t0,
            pump=pump,
            sensor=None,  # NOTE 1: not the best, but architecture conflicts
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
        )
        for patient_config in sim_parser.get_patient_config()
    ]

    all_results = {}
    for vp in virtual_patients:
        for bg_trace in vp.patient_config.glucose_history.bg_traces:

            sensor_generator.fit(bg_trace)
            sensors = [sensor_generator.get_sensor() for _ in range(num_sensors)]

            for sensor in sensors:
                sim_id = "{} {} {}".format(vp.name, bg_trace.name, sensor.name)
                vp.sensor = sensor  # NOTE 1: not the best, but architecture conflicts

                simulation = Simulation(
                    time=t0,
                    duration_hrs=sim_parser.get_simulation_duration_hours(),
                    simulation_config=sim_parser.get_simulation_config(),
                    virtual_patient=vp,
                    controller=controller,
                )

                simulation.run()
                all_results[sim_id] = simulation.get_results_df()

    # Compute risk metrics and viz on results


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/fda_risk_scenarios/"
    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs.tsv"
    )
    do_icgm_risk_analysis(scenario_csv_filepath)
