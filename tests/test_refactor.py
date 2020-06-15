__author__ = "Cameron Summers"

import os
import subprocess

import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation

REGRESSION_COMMIT = "d3a1e74"


def SUNSETTED_TEST_simulator_refactor():
    """
    Check that the output of the refactored simulation matches the original code for MVP.

    Returns
    -------

    """
    sim_parser = ScenarioParserCSV("tests/data/Scenario-0-simulation-template - inputs.tsv")
    t0 = sim_parser.get_simulation_start_time()

    controllers = [
        DoNothingController(time=t0, controller_config=sim_parser.get_controller_config()),
        LoopController(time=t0, controller_config=sim_parser.get_controller_config())
    ]

    all_controller_results = []

    for controller in controllers:
        print("Running w/controller: {}".format(controller.name))
        pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())
        sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
        )
        vp.patient_config.recommendation_accept_prob = 0.0

        simulation = Simulation(
            time=t0,
            duration_hrs=sim_parser.get_simulation_duration_hours(),
            virtual_patient=vp,
            controller=controller
        )

        simulation.run()
        results_df = simulation.get_results_df()
        all_controller_results.append(results_df)

    do_nothing_results_df = all_controller_results[0]
    loop_results_df = all_controller_results[1]

    # Pump only
    old_code_pump_bg = np.load("tests/data/scenario_template_v0.5_bg_pump.npy")
    assert np.sum(np.abs(do_nothing_results_df['bg'].to_numpy()[:96] - old_code_pump_bg[:96])) < 1e-6

    # Loop
    old_code_bg_actual = np.load("tests/data/scenario_template_v0.5_bg_actual.npy")
    old_code_iob = np.load("tests/data/scenario_template_v0.5_iob.npy")
    old_code_temp_basal = np.load("tests/data/scenario_template_v0.5_temp_basal.npy")

    assert np.sum(np.abs(loop_results_df['temp_basal'].to_numpy() - old_code_temp_basal)) < 1e-6
    assert np.sum(np.abs(loop_results_df['bg'].to_numpy() - old_code_bg_actual)) < 1e-6

    # assert np.sum(np.abs(loop_results_df['iob'].to_numpy()[:96] - old_code_iob[:96])) < 1e-6


def test_regression():
    """
    Test the output of the canonical simulation to saved values from a standard commit.
    """

    controllers = [
        DoNothingController,
        LoopController
    ]

    for controller in controllers:
        load_path = "tests/data/regression/commit-{}/{}/".format(REGRESSION_COMMIT, controller.get_classname())

        t0, sim = get_canonical_simulation(
            controller_class=controller,
            duration_hrs=8,
            include_initial_events=True
        )
        sim.run()
        results_df = sim.get_results_df()

        regr_bg = np.load(os.path.join(load_path, "bg.npy"))
        assert np.sum(np.abs(results_df['bg'].to_numpy() - regr_bg)) < 1e-6

        regr_iob = np.load(os.path.join(load_path, "iob.npy"))
        assert np.sum(np.abs(results_df['iob'].to_numpy() - regr_iob)) < 1e-6


def make_regression():
    """
    Make a regression test with the current version of the code.
    """

    current_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")

    controllers = [
        DoNothingController,
        LoopController
    ]

    this_dir = os.path.dirname(os.path.abspath(__file__))

    for controller in controllers:
        save_path = os.path.join(this_dir,
                                 "data/regression/commit-{}/{}/".format(current_commit, controller.get_classname()))
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        else:
            raise Exception("Save path exists: {}".format(save_path))

        t0, sim = get_canonical_simulation(
            controller_class=controller,
            duration_hrs=8,
            include_initial_events=True
        )
        sim.run()
        results_df = sim.get_results_df()

        np.save(os.path.join(save_path, "bg.npy"), results_df['bg'].to_numpy())
        np.save(os.path.join(save_path, "iob.npy"), results_df['iob'].to_numpy())
        np.save(os.path.join(save_path, "temp_basal.npy"), results_df['temp_basal'].to_numpy())
        pass

        # plot_sim_results({0: results_df})


if __name__ == "__main__":

    make_regression()