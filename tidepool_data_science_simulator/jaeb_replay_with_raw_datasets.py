import os
import pandas as pd

from datetime import timedelta
import glob

from tidepool_data_science_simulator.makedata.jaeb_data_sim_parser import RawDatasetJaebParser
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.patient.real_patient import RealPatientReplay
from tidepool_data_science_simulator.models.sensor import RealSensor
from tidepool_data_science_simulator.models.controller import LoopReplay
from tidepool_data_science_simulator.models.simulation import ReplaySimulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def run_replay(path_to_data, settings_data):
    jaeb_parser = RawDatasetJaebParser(
        data_path=path_to_data,
        settings_data=settings_data
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
    # load in all settings from parsed settings file
    data_path = os.path.join("..", "data", "PHI")

    all_settings_schedule_file = os.path.join(
        data_path, "phi-all-setting-schedule_dataset-basal_minutes_30-outcome_hours_3-expanded_windows_False-days_around_issue_report_7.csv"
    )
    survey_file = os.path.join(
        data_path, "PHI Tidepool Survey Data 08-19-2020-cleaned-2020_09_15_13-v0_1_develop-cfb2713.csv"
    )
    all_jos_files = glob.glob(
        os.path.join("..", "data", "PHI", "compressed_and_zipped", "*LOOP*")
    )

    settings_schedule_df = pd.read_csv(all_settings_schedule_file, low_memory=False)
    survey_df = pd.read_csv(survey_file, low_memory=False)
    all_loopers = sorted(settings_schedule_df["loop_id"].unique())
    loop_versions = ["Loop v1.9.6", "Loop v1.9.5dev"]
    settings_filtered_by_version = settings_schedule_df[settings_schedule_df["loop_version"].isin(loop_versions)]
    unique_loopers = settings_filtered_by_version["loop_id"].unique()
    loop_user_df = pd.DataFrame(unique_loopers, columns=["unique_users"])
    for l, loop_id in enumerate(unique_loopers):
        try:
            loop_user_df.loc[l, "data_path"] = next(f for f in all_jos_files if loop_id in f)
        except:
            print("skipping {} bc we don't have data".format(loop_id))

    users_with_data_df = loop_user_df.dropna()
    for i in users_with_data_df.index:
        f = users_with_data_df.loc[i, "data_path"]
        loop_id = users_with_data_df.loc[i, "unique_users"]
        user_settings = settings_filtered_by_version[settings_filtered_by_version["loop_id"] == loop_id]
        run_replay(path_to_data=f, settings_data=user_settings)