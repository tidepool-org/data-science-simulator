__author__ = "Cameron Summers"

import os
import logging

import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_virtual_patient_model_config
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.sensor_icgm import SensoriCGM, get_base_icgm_sensor_config
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing, DATA_DIR


logger = logging.getLogger(__name__)

@timing
def build_sims(scenario_csv_filepath):
    """
    Demo Tidepool Loop v0.2
    """
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    controllers = [
        DoNothingController(time=t0, controller_config=sim_parser.get_controller_config()),
        LoopController(time=t0, controller_config=sim_parser.get_controller_config()),
        LoopControllerDisconnector(time=t0, controller_config=sim_parser.get_controller_config(), connect_prob=0.8)
    ]

    sims = {}
    random_state = np.random.RandomState(0)
    for controller in controllers:

        sim_id = controller.name

        pump_config = sim_parser.get_pump_config()
        pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

        sensor_config = sim_parser.get_sensor_config()
        # sensor_config = get_base_icgm_sensor_config(t0=t0)
        sensor = IdealSensor(time=t0, sensor_config=sensor_config)#, random_state=random_state)

        # patient_config = sim_parser.get_patient_config()
        t0, patient_config = get_canonical_virtual_patient_model_config(random_state=random_state)
        patient_config.action_timeline = ActionTimeline()

        vp = VirtualPatientModel(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
        )

        simulation = Simulation(
            time=t0,
            duration_hrs=24.0,
            virtual_patient=vp,
            controller=controller,
            sim_id=sim_id,
            multiprocess=True
        )

        sims[sim_id] = simulation

    return sims


if __name__ == "__main__":

    scenarios_folder_path = os.path.join(DATA_DIR, "raw/fda_risk_scenarios/")

    for scenario_file_names in os.listdir(scenarios_folder_path)[:1]:
        scenario_csv_filepath = os.path.join(scenarios_folder_path, scenario_file_names)
        sims = build_sims(scenario_csv_filepath)

        logger.info("Starting to run {} sims.".format(len(sims)))

        sim_results_df = run_simulations(sims, save_dir=".", save_results=False)

        plot_sim_results(sim_results_df, save=False)

