import os
import datetime
from datetime import timedelta

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation, ActionTimeline, VirtualPatientDeleteLoopData
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


@timing
def compare_two_loop_scenarios(scenario_csv_filepath):
    """
    Compare loop running with an action and without that action.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """

    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    comparison_controllers = [
        DoNothingController(
            time=t0,
            controller_config=sim_parser.get_controller_config()
        ),
        LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config()
        )
    ]

    comparison_patient_action_configs = [
        [],
        [
            {
                "t0_delay_minutes": 30,
                "action": VirtualPatientDeleteLoopData("Deleted Insulin History")
            }
        ]
    ]

    configurations = [
        {
            "controller": controller,
            "actions": actions,
        }
        for controller in comparison_controllers
        for actions in comparison_patient_action_configs
    ]

    all_results = {}
    for i, config in enumerate(configurations):

        sim_id = i
        actions = config["actions"]
        controller = config["controller"]

        print("Running: {}".format(sim_id))

        pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())
        sensor = IdealSensor(time=t0, sensor_config=sim_parser.get_sensor_config())

        patient_config = sim_parser.get_patient_config()
        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file
        patient_config.action_timeline = ActionTimeline()

        for action_config in actions:
            delay_minutes = action_config["t0_delay_minutes"]
            action = action_config["action"]
            patient_config.action_timeline.add_event(t0 + timedelta(minutes=delay_minutes), action)

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
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

    plot_sim_results(all_results, save=False)


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/fda_risk_scenarios/"
    scenario_file_names = os.listdir(scenarios_folder_path)

    for file_name in scenario_file_names[:1]:
        scenario_csv_filepath = os.path.join(
            scenarios_folder_path, file_name
        )
        compare_two_loop_scenarios(scenario_csv_filepath)
