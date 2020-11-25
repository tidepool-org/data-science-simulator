import os
import pandas as pd

from datetime import timedelta

from tidepool_data_science_simulator.makedata.jaeb_data_sim_parser import JaebReplayParser
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.patient.real_patient import RealPatientReplay
from tidepool_data_science_simulator.models.sensor import RealSensor
from tidepool_data_science_simulator.models.controller import LoopReplay
from tidepool_data_science_simulator.models.simulation import ReplaySimulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def run_replay(path_to_settings, path_to_time_series_data, t0=None):
    jaeb_parser = JaebReplayParser(
        path_to_settings=path_to_settings,
        path_to_time_series_data=path_to_time_series_data,
        t0=t0,
        days_in=7
    )

    t0 = jaeb_parser.get_simulation_start_time()
    all_results = {}

    pump = ContinuousInsulinPump(
        time=t0,
        pump_config=jaeb_parser.get_pump_config()
    )

    sensor = RealSensor(
        time=t0,
        sensor_config=jaeb_parser.get_sensor_config()
    )

    patient_config = jaeb_parser.get_patient_config()
    patient = RealPatientReplay(
        time=t0,
        pump=pump,
        sensor=sensor,
        patient_config=patient_config
    )

    controller = LoopReplay(
        time=t0,
        controller_config=jaeb_parser.get_controller_config()
    )

    sim = ReplaySimulation(
        time=t0,
        duration_hrs=24,
        virtual_patient=patient,
        controller=controller
    )
    print("Running simulation...")
    sim.run(early_stop_datetime=(t0 + timedelta(minutes=235)))
    sim.run()
    print("Simulation done.")
    results_df = sim.get_results_df()
    sim_id = jaeb_parser.patient_id + "-" + str(jaeb_parser.report_num)
    results_df.to_csv('../dataframe_for_patient {}.csv'.format(sim_id))
    all_results[sim_id] = results_df

    plot_sim_results(all_results)


if __name__ == "__main__":
    parsed_data_folder = "../PHI-Jaeb-Data/"
    parsed_data_files = os.listdir(parsed_data_folder)

    time_series_files = []
    time_series_folder = ""
    for filename in parsed_data_files:
        if "data-summary" in filename:
            issue_report_settings_path = os.path.join(parsed_data_folder, filename)
        elif "time-series" in filename:
            time_series_files = os.listdir(parsed_data_folder + filename)
            time_series_folder = filename

    issue_report_settings = pd.read_csv(issue_report_settings_path, sep=",")
    all_loops = issue_report_settings.loc[issue_report_settings['loop_version'] == 'Loop v1.9.6']

    for filename in time_series_files[:10]:
        try:
            time_series_path = parsed_data_folder + time_series_folder + "/" + filename
            run_replay(path_to_settings=issue_report_settings_path, path_to_time_series_data=time_series_path)
        except:
            pass