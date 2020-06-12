import os
import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


@timing
def eval_settings(scenario_csv_filepath):
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    np.random.seed(1234)
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    controller = LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config(),
        )

    sim_id = controller.name
    print("Running: {}".format(sim_id))

    pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

    # sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
    sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

    vp = VirtualPatientModel(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=sim_parser.get_patient_config(),
        remember_meal_bolus_prob=1.0,
        correct_bolus_bg_threshold=180,
        correct_bolus_delay_minutes=30,
        correct_carb_bg_threshold=80,
        correct_carb_delay_minutes=10,
        carb_count_noise_percentage=0.1,
        id=0,
    )

    simulation = Simulation(
        time=t0,
        duration_hrs=8.0,
        virtual_patient=vp,
        controller=controller,
    )

    simulation.run()

    results_df = simulation.get_results_df()

    print("BG median", results_df['bg'].median())
    print("BG variance", results_df['bg'].var())

    true_isf = vp.patient_config.insulin_sensitivity_schedule.get_state().value
    pump_isf = vp.pump.pump_config.insulin_sensitivity_schedule.get_state().value
    plot_sim_results({"ISF {}, Set ISF {}".format(true_isf, pump_isf): results_df})


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/"
    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs - SettingsEvalDemo.tsv"
    )
    eval_settings(scenario_csv_filepath)
