__author__ = "Cameron Summers"


__author__ = "Cameron Summers"

import datetime
import numpy as np
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.models.measures import TempBasal

import os

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


def analyze_omnipod_missing_pulses():

    current_time, patient = get_canonical_risk_patient(pump_class=OmnipodMissingPulses)

    delivered_insulin = []
    undelivered_insulin = []

    temp_basal_scenario = [0.55, 0.45] * 12  # This scenario gives no insulin

    for temp_basal_value in temp_basal_scenario:
        next_time = current_time + datetime.timedelta(minutes=5)

        patient.pump.update(next_time)
        temp_basal = TempBasal(next_time, temp_basal_value, duration_minutes=30, units="U/hr")
        patient.pump.set_temp_basal(temp_basal)

        delivered_insulin.append(patient.pump.insulin_delivered_last_update)
        undelivered_insulin.append(patient.pump.undelivered_insulin)

    total_delivered_insulin = np.sum(delivered_insulin)
    total_undelivered_insulin = np.sum(undelivered_insulin)
    total_expected_insulin = total_delivered_insulin + total_undelivered_insulin
    print("Total Delivered Insulin  {:>4} ({:.0f}%)".format(total_delivered_insulin,
                                                           total_delivered_insulin / total_expected_insulin * 100.0)),
    print("Total Undelivered Insulin {:>4} ({:.0f}%)".format(total_undelivered_insulin,
                                                            total_undelivered_insulin / total_expected_insulin * 100.0))

    plt.title("Omnipod Missing Insulin Pulses")
    plt.plot(delivered_insulin, label="delivered")
    plt.plot(undelivered_insulin, label="undelivered")
    plt.plot(temp_basal_scenario, label="Temp Basal Values")
    plt.legend()
    plt.show()


@timing
def analyze_omnipod_missing_pulses_wLoop(scenario_csv_filepath):
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    controllers = [
        DoNothingController(
            time=t0, controller_config=sim_parser.get_controller_config()
        ),
        LoopController(
            time=t0,
            loop_config=sim_parser.get_controller_config(),
            simulation_config=sim_parser.get_simulation_config(),
        ),
    ]

    all_results = {}
    for controller in controllers:
        sim_id = controller.name
        print("Running: {}".format(sim_id))

        pump = OmnipodMissingPulses(time=t0, pump_config=sim_parser.get_pump_config())
        # pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())
        # pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

        # sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
        sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

        patient_config = sim_parser.get_patient_config()
        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file
        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
        )

        simulation = Simulation(
            time=t0,
            duration_hrs=24.0,
            simulation_config=sim_parser.get_simulation_config(),
            virtual_patient=vp,
            controller=controller,
        )

        simulation.run()

        results_df = simulation.get_results_df()
        all_results[sim_id] = results_df

    plot_sim_results(all_results)


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/"
    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs - OmnipodMissingPulses.tsv"
    )
    analyze_omnipod_missing_pulses_wLoop(scenario_csv_filepath)

    # analyze_omnipod_missing_pulses()


