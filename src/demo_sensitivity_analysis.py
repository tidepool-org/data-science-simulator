__author__ = "Cameron Summers"


__author__ = "Cameron Summers"

import os
import numpy as np
from collections import defaultdict
import time

from tdsm.models.simple_metabolism_model import SimpleMetabolismModel

from src.models.simulation import Simulation
from src.models.controller import (
    DoNothingController,
    LoopController,
    LoopControllerDisconnector,
)
from src.models.patient import VirtualPatient, VirtualPatientModel
from src.models.pump import Omnipod
from src.models.sensor import IdealSensor, NoisySensor
from src.makedata.scenario_parser import ScenarioParserCSV
from src.visualization.sim_viz import plot_sim_results
from src.evaluation.variance_analysis import get_first_order_indices
from src.utils import timing


@timing
def analyze_variance(scenario_csv_filepath, param_grid, plot):
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

    pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())

    # sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
    sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

    print("Length of param grid: {}".format(len(param_grid)))

    all_results = {}
    sim_id_params = {}
    for i, pgrid in enumerate(param_grid):

        vp = VirtualPatientModel(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
            remember_meal_bolus_prob=1.0,
            correct_bolus_delay_minutes=30,
            correct_carb_delay_minutes=10,
            id=i,
            **pgrid
        )

        for controller in controllers:
            sim_id = "{} {}".format(vp.name, controller.name)
            print("Running: {}".format(sim_id))

            simulation = Simulation(
                time=t0,
                duration_hrs=24.0 * 10,
                simulation_config=sim_parser.get_simulation_config(),
                virtual_patient=vp,
                controller=controller,
            )
            simulation.run()

            results_df = simulation.get_results_df()
            all_results[sim_id] = results_df
            sim_id_params[sim_id] = pgrid

    if plot:
        plot_sim_results(all_results)

    param_names = param_grid[0].keys()
    print(get_first_order_indices(param_names, sim_id_params, all_results))


@timing
def analyze_variance_multiprocess(scenario_csv_filepath, param_grid, plot):
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

    pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())

    # sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
    sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

    print("Length of param grid: {}".format(len(param_grid)))

    sims = {}
    sim_id_params = {}
    for i, pgrid in enumerate(param_grid):

        vp = VirtualPatientModel(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
            remember_meal_bolus_prob=1.0,
            correct_bolus_delay_minutes=30,
            correct_carb_delay_minutes=10,
            id=i,
            **pgrid
        )

        for controller in controllers:
            sim_id = "{} {}".format(vp.name, controller.name)
            print("Running: {}".format(sim_id))

            simulation = Simulation(
                time=t0,
                duration_hrs=24.0 * 10,
                simulation_config=sim_parser.get_simulation_config(),
                virtual_patient=vp,
                controller=controller,
                multiprocess=True,
            )

            sims[sim_id] = simulation
            simulation.start()
            sim_id_params[sim_id] = pgrid

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

    if plot:
        plot_sim_results(all_results)

    param_names = param_grid[0].keys()
    print(get_first_order_indices(param_names, sim_id_params, all_results))


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/fda_risk_scenarios/"

    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs.tsv"
    )

    param_grid = [
        {
            "correct_bolus_bg_threshold": a,
            "correct_carb_bg_threshold": b,
            "carb_count_noise_percentage": c,
        }
        for a in np.arange(140, 200, 10)
        for b in np.arange(50, 90, 10)
        for c in np.arange(0.05, 0.2, 0.02)
    ]

    # analyze_variance(scenario_csv_filepath, param_grid, False)
    analyze_variance_multiprocess(scenario_csv_filepath, param_grid, False)


