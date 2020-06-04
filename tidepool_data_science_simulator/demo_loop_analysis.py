__author__ = "Cameron Summers"

import os
import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import (
    DoNothingController,
    LoopController,
    LoopControllerDisconnector,
)
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def analyze_controllers(scenario_csv_filepath):
    """
    Compare two controllers for a given scenario file:
        1. No controller, that is modulation comes from pump only
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
        # LoopController(time=t0, loop_config=sim_parser.get_controller_config(), simulation_config=sim_parser.get_simulation_config()),
        # LoopControllerDisconnector(time=t0, loop_config=sim_parser.get_controller_config(), simulation_config=sim_parser.get_simulation_config(), connect_prob=0.25),
    ]

    virtual_patients = []
    for i in range(1):
        t0, vp = get_canonical_risk_patient(t0, patient_class=VirtualPatientModel)
        virtual_patients.append(vp)

    all_results = {}
    for controller in controllers:
        for vp in virtual_patients:
            sim_id = "{} {}".format(vp.name, controller.name)
            print("Running: {}".format(sim_id))

            simulation = Simulation(
                time=t0,
                duration_hrs=8.0,
                simulation_config=sim_parser.get_simulation_config(),
                virtual_patient=vp,
                controller=controller,
            )
            simulation.run()

            results_df = simulation.get_results_df()
            all_results[sim_id] = results_df

    plot_sim_results(all_results)

    # Then...
    # Do statistical significance test on outcome variables
    # e.g. Measure amount of "work" for each controller, human vs loop, for a given outcome


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/fda_risk_scenarios/"
    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs.tsv"
    )

    analyze_controllers(scenario_csv_filepath)
