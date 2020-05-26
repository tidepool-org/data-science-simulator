import os

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import Omnipod
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


@timing
def compare_loop_to_pump_only(scenario_csv_filepath):
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

        pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())

        sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
        # sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

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
            simulation_config=sim_parser.get_simulation_config(),
            virtual_patient=vp,
            controller=controller,
        )

        simulation.run()

        results_df = simulation.get_results_df()
        all_results[sim_id] = results_df

    plot_sim_results(all_results)


if __name__ == "__main__":

    scenarios_folder_path = "data/raw/fda_risk_scenarios/"
    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs.tsv"
    )
    compare_loop_to_pump_only(scenario_csv_filepath)
