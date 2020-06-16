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

# TEMP: Testing iCGM Sensor
from tidepool_data_science_models.models.icgm_sensor_generator import iCGMSensor, iCGMSensorGenerator, sf
import pandas as pd


# %% @timing
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
        DoNothingController(time=t0, controller_config=sim_parser.get_controller_config()),
        LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config(),
        ),
    ]

    all_results = {}
    for controller in controllers:
        sim_id = controller.name
        print("Running: {}".format(sim_id))

        pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())

        # sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
        # sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

        # sample_sensor_properties = pd.DataFrame(index=[0])
        # sample_sensor_properties["initial_bias"] = 2.867306
        # sample_sensor_properties["phi_drift"] = 1.862317
        # sample_sensor_properties["bias_drift_range_start"] = 0.968757
        # sample_sensor_properties["bias_drift_range_end"] = 0.956423
        # sample_sensor_properties["bias_drift_oscillations"] = 0.918726
        # sample_sensor_properties["bias_norm_factor"] = 55.000000
        # sample_sensor_properties["noise_coefficient"] = 2.292680
        # sample_sensor_properties["delay"] = 10
        # sample_sensor_properties["random_seed"] = 0
        # sample_sensor_properties["bias_drift_type"] = "random"


        sensor_generator = iCGMSensorGenerator(batch_training_size=3)
        sensor_generator.fit(true_bg_trace=sim_parser.patient_glucose_history.bg_values)
        sensor = sensor_generator.generate_sensors(1, sensor_start_time_index=290, sensor_start_datetime=t0)[0]
        # sample_sensor = iCGMSensor(sensor_properties=sample_sensor_properties, time_index=136, sensor_datetime=t0)
        # sensor = sample_sensor

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
        )

        # iCGM Backfill
        vp.sensor.backfill_and_calculate_sensor_data(vp.bg_history.bg_values[-289:])

        simulation = Simulation(
            time=t0,
            duration_hrs=8.0,
            virtual_patient=vp,
            controller=controller,
        )

        simulation.run()

        results_df = simulation.get_results_df()
        all_results[sim_id] = results_df

    plot_sim_results(all_results)


# %%
if __name__ == "__main__":

    scenarios_folder_path = "data/raw/fda_risk_scenarios/"
    scenario_csv_filepath = os.path.join(scenarios_folder_path, "Scenario-0-simulation-template - inputs.tsv")

    scenario_csv_filepath = "/Users/jameno/Desktop/github/icgm-sensitivity-analysis/data/interim/sample-snapshot-export/anonymized-sample.csv_condition8.csv"
    compare_loop_to_pump_only(scenario_csv_filepath)
