import os

from tidepool_data_science_simulator.makedata.jaeb_data_sim_parser import JaebDataSimParser
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.patient.virtual_patient import VirtualPatient
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


def run_replay(path_to_settings, path_to_time_series_data, t0=None):
    jaeb_parser = JaebDataSimParser(
        path_to_settings=path_to_settings,
        path_to_time_series_data=path_to_time_series_data,
        t0=t0
    )

    t0 = jaeb_parser.get_simulation_start_time()

    pump = ContinuousInsulinPump(
        time=t0,
        pump_config=jaeb_parser.get_pump_config()
    )
    sensor = IdealSensor(
        time=t0,
        sensor_config=jaeb_parser.get_sensor_config()
    )

    patient = VirtualPatient(
        time=t0, pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=jaeb_parser.get_patient_config()
    )

    controller = LoopController(
        time=t0,
        controller_config=jaeb_parser.get_controller_config()
    )

    sim = Simulation(
        time=t0,
        duration_hrs=24,
        virtual_patient=patient,
        controller=controller
    )

    sim.run()
    results_df = sim.get_results_df()
    plot_sim_results(results_df)


if __name__ == "__main__":
    parsed_data_folder = "../PHI-Jaeb-Data/"
    parsed_data_files = os.listdir(parsed_data_folder)

    time_series_files = ""
    for filename in parsed_data_files:
        if "data-summary" in filename:
            issue_report_settings_path = os.path.join(parsed_data_folder, filename)
        else:
            time_series_files = os.listdir(parsed_data_folder + filename)
            

    run_replay(path_to_settings=issue_report_settings_path, path_to_time_series_data=time_series_path)