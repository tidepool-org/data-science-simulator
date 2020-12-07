__author__ = "Cameron Summers"

import time
import pdb
import sys
import os
import datetime
import json
import copy
import re
from collections import defaultdict

import numpy as np
from numpy.random import RandomState
import pandas as pd
# import seaborn as sns
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results, load_result
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.models.icgm_sensor import (
    SensoriCGM, SensoriCGMModelOverlayNoiseBiasWorst, CLEAN_INITIAL_CONTROLS, iCGM_THRESHOLDS, SensoriCGMModelOverlayV1,
    DexcomG6RateModel, DexcomG6ValueModel
)
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace

from tidepool_data_science_simulator.run import run_simulations

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, percent_values_ge_70_le_180
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score


def load_vp_training_data(scenarios_dir):
    """
    Load scenarios in a dictionary for easier data management

    Parameters
    ----------
    scenarios_dir: str
        Path to directory with scenarios

    Returns
    -------
    dict
        Map of virtual patient id -> bg condition -> filename
    """

    file_names = os.listdir(scenarios_dir)
    all_scenario_files = [filename for filename in file_names if filename.endswith('.csv')]
    print("Num scenario files: {}".format(len(all_scenario_files)))

    vp_scenario_dict = defaultdict(dict)
    for filename in all_scenario_files:
        vp_id = re.search("train_(.*)\.csv.+", filename).groups()[0]
        bg_condition = re.search("condition(\d)", filename).groups()[0]
        vp_scenario_dict[vp_id][bg_condition] = filename

    return vp_scenario_dict


def get_dexcom_rate_sensor(t0, sim_parser, random_state):

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.history_window_hrs = 24 * 10

    sensor_config.behavior_models = [
        DexcomG6RateModel(),
    ]

    sensor_config.sensor_range = range(40, 401)
    sensor_config.special_controls = iCGM_THRESHOLDS
    sensor_config.initial_controls = CLEAN_INITIAL_CONTROLS
    sensor_config.do_look_ahead = True
    sensor_config.look_ahead_min_prob = 0.7

    num_history_values = 24 * 12
    sensor = SensoriCGM(t0 - datetime.timedelta(minutes=num_history_values * 5.0),
                        sensor_config=sensor_config,
                        random_state=random_state)
    sensor.name = "iCGM_DexcomRate"

    for dt, true_bg in zip(sim_parser.patient_glucose_history.datetimes[-num_history_values:],
                           sim_parser.patient_glucose_history.bg_values[-num_history_values:]):
        sensor.update(dt, patient_true_bg=true_bg)

    return sensor


def get_ideal_sensor(t0, sim_parser):

    ideal_sensor_config = SensorConfig(sensor_bg_history=sim_parser.patient_glucose_history)
    sensor = IdealSensor(time=t0, sensor_config=ideal_sensor_config)
    return sensor


def get_icgm_sensor(t0, sim_parser, max_bias_percentage, random_state):

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.history_window_hrs = 24 * 10

    sensor_config.behavior_models = [
        SensoriCGMModelOverlayNoiseBiasWorst(max_bias_percentage),
    ]

    sensor_config.sensor_range = range(40, 401)
    sensor_config.special_controls = iCGM_THRESHOLDS
    sensor_config.initial_controls = CLEAN_INITIAL_CONTROLS
    sensor_config.do_look_ahead = True
    sensor_config.look_ahead_min_prob = 0.7

    num_history_values = 24 * 12
    sensor = SensoriCGM(t0 - datetime.timedelta(minutes=num_history_values * 5.0),
                        sensor_config=sensor_config,
                        random_state=random_state)

    sensor.sim_start_time = t0

    for dt, true_bg in zip(sim_parser.patient_glucose_history.datetimes[-num_history_values:],
                           sim_parser.patient_glucose_history.bg_values[-num_history_values:]):
        sensor.update(dt, patient_true_bg=true_bg)

    return sensor


def get_initial_offset_sensor(t0, sim_parser, random_state, initial_error_value):

    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
    sensor_config.history_window_hrs = 24 * 10

    sensor_config.behavior_models = [
        SensoriCGMModelOverlayV1(bias=0, sigma=25, delay=0, spurious_value_prob=0.0, num_consecutive_spurious=1),
    ]

    sensor_config.sensor_range = range(40, 401)
    sensor_config.special_controls = iCGM_THRESHOLDS
    sensor_config.initial_controls = CLEAN_INITIAL_CONTROLS
    sensor_config.do_look_ahead = False
    sensor_config.look_ahead_min_prob = 0.7

    num_history_values = 24 * 12
    sensor = SensoriCGM(t0 - datetime.timedelta(minutes=num_history_values * 5.0),
                        sensor_config=sensor_config,
                        random_state=random_state)
    sensor.name = "iCGM_{}".format(initial_error_value)

    sensor.sim_start_time = t0

    sensor.t0_error_bg = initial_error_value

    for dt, true_bg in zip(sim_parser.patient_glucose_history.datetimes[-num_history_values:],
                           sim_parser.patient_glucose_history.bg_values[-num_history_values:]):
        sensor.update(dt, patient_true_bg=true_bg)

    return sensor


def sample_uniformly_positive_error_cgm_ranges(value, num_samples=10):

    if value < 40:
        icgm_low, icgm_high = (40, 80)
    elif 40 <= value <= 60:
        icgm_low, icgm_high = (40, 120)
    elif 60 < value <= 80:
        icgm_low, icgm_high = (61, 120)
    elif 80 < value <= 120:
        icgm_low, icgm_high = (81, 200)
    elif 120 < value <= 160:
        icgm_low, icgm_high = (121, 250)
    elif 160 < value <= 200:
        icgm_low, icgm_high = (161, 300)
    elif 200 < value <= 250:
        icgm_low, icgm_high = (201, 350)
    elif 250 < value <= 300:
        icgm_low, icgm_high = (251, 400)
    elif 300 < value <= 350:
        icgm_low, icgm_high = (301, 400)
    elif 350 < value <= 400:
        icgm_low, icgm_high = (351, 400)
    elif value > 400:
        icgm_low, icgm_high = (351, 401)

    step = (icgm_high - icgm_low) / (num_samples)

    sampled_errors = [round(v) for v in np.arange(icgm_low, icgm_high + step, step)]

    return sampled_errors


def sample_worst_negative_error_cgm_ranges(value):

    if value < 40:
        icgm_low = 40
    elif 40 <= value <= 60:
        icgm_low = 40
    elif 60 < value <= 80:
        icgm_low = 40
    elif 80 < value <= 120:
        icgm_low = 40
    elif 120 < value <= 160:
        icgm_low = 40
    elif 160 < value <= 200:
        icgm_low = 40
    elif 200 < value <= 250:
        icgm_low = 121
    elif 250 < value <= 300:
        icgm_low = 161
    elif 300 < value <= 350:
        icgm_low = 201
    elif 350 < value <= 400:
        icgm_low = 251
    elif value > 400:
        icgm_low = 351

    return [icgm_low]


def build_icgm_sim_generator(vp_scenario_dict, sim_batch_size=30):
    """
    Build simulations for the FDA 510k Loop iCGM sensitivity analysis.

    Scenario files are on Compute-2 in Cameron Summers' copy of this code base.
    """
    analysis_type_list = [
        "temp_basal_only",
        # "correction_bolus",
        # "meal_bolus"
    ]

    sim_ctr = 0
    sims = {}
    for vp_idx, (vp_id, bg_scenario_dict) in enumerate(vp_scenario_dict.items()):
        print("VP", vp_idx)
        for bg_cond_id, scenario_filename in bg_scenario_dict.items():

            scenario_path = os.path.join(scenarios_dir, scenario_filename)
            sim_parser = ScenarioParserCSV(scenario_path)

            # Save patient properties for analysis
            vp_filename = "vp{}.json".format(vp_id)
            vp_properties = {
                "age": sim_parser.age,
                "ylw": sim_parser.ylw,
                "patient_scenario_filename": scenario_filename
            }
            with open(os.path.join(save_dir, vp_filename), "w") as file_to_write:
                json.dump(vp_properties, file_to_write, indent=4)

            t0 = sim_parser.get_simulation_start_time()

            controller = LoopController(time=t0, controller_config=sim_parser.get_controller_config())
            controller.num_hours_history = 8  # Force 8 hours to look for boluses at start of simulation

            pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

            sensors = []
            sim_seed = np.random.randint(0, 1e7)

            sensors.append(get_ideal_sensor(t0, sim_parser))
            # sensors.append(get_icgm_sensor(t0, sim_parser, max_bias_percentage=0, random_state=RandomState(sim_seed)))
            # sensors.append(get_dexcom_rate_sensor(t0, sim_parser, random_state=RandomState(sim_seed)))

            t0_true_bg = sim_parser.patient_glucose_history.bg_values[-1]
            sampled_error_values = sample_uniformly_positive_error_cgm_ranges(t0_true_bg, num_samples=10)
            # sampled_error_values = sample_worst_negative_error_cgm_ranges(t0_true_bg)
            for initial_error_value in sampled_error_values:
                sensors.append(get_initial_offset_sensor(t0, sim_parser, random_state=RandomState(sim_seed),
                                                         initial_error_value=initial_error_value))

            print("Num sensors", len(sensors))
            for sensor in sensors:
                for analysis_type in analysis_type_list:

                    sim_id = "vp{}.bg{}.s{}.{}".format(
                        vp_id, bg_cond_id, sensor.name, analysis_type
                    )

                    # For restarting at same spot if things break midway
                    # if os.path.exists(os.path.join(save_dir, "{}.tsv".format(sim_id))):
                    #     continue

                    vp = VirtualPatientISA(
                        time=t0,
                        pump=copy.deepcopy(pump),
                        sensor=copy.deepcopy(sensor),
                        metabolism_model=SimpleMetabolismModel,
                        patient_config=copy.deepcopy(sim_parser.get_patient_config()),
                        t0=t0,
                        analysis_type=analysis_type,
                    )

                    sim = Simulation(
                        time=t0,
                        duration_hrs=8.0,
                        virtual_patient=vp,
                        controller=copy.deepcopy(controller),
                        multiprocess=True,
                        sim_id=sim_id
                    )

                    sim.seed = 0
                    sims[sim_id] = sim

                    sim_ctr += 1

                    if sim_ctr == sim_batch_size:
                        yield sims
                        sims = {}
                        sim_ctr = 0

    yield sims


def plot_icgm_results(result_dir):

    all_results = load_results(result_dir, ext="tsv", max_dfs=1000)

    sim_groups_to_plot = defaultdict(dict)
    for sim_id, result_df in all_results.items():
        non_sensor_id = re.sub("\.s.*\.", "", sim_id)
        if "Ideal" in sim_id:
            continue
        sim_groups_to_plot[non_sensor_id][sim_id] = result_df

    for sim_group_id, sim_group_results in sim_groups_to_plot.items():
        plot_sim_results(sim_group_results)


def plot_sensor_error_vs_risk(result_dir):

    sim_results = collect_sims_and_results(result_dir, sim_id_pattern="vp.*bg.*.json", max_sims=np.inf)

    summary_data = []
    for sim_id, sim_json_info in sim_results.items():
        if "Ideal" in sim_id:
            continue

        for sim_id_match, sim_json_info_match in sim_results.items():
            non_sensor_id = re.sub("\.s.*\.", "", sim_id)
            non_sensr_id_match = re.sub("\.sIdealSensor\.", "", sim_id_match)
            if non_sensor_id == non_sensr_id_match:
                sim_json_info_match = sim_json_info_match
                break

        df_results_dict = load_result(sim_json_info["result_path"])
        df_results = df_results_dict[sim_json_info["result_path"]]
        true_bg = df_results['bg']
        true_bg[true_bg < 1] = 1
        lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(true_bg)
        dkai_icgm = dka_index(df_results['iob'], df_results["sbr"])

        df_results_ideal_dict = load_result(sim_json_info_match["result_path"])
        df_results_ideal = df_results_ideal_dict[sim_json_info_match["result_path"]]
        true_bg = df_results_ideal["bg"]
        true_bg[true_bg < 1] = 1
        lbgi_ideal, hbgi_ideal, brgi_ideal = blood_glucose_risk_index(true_bg)
        dkai_ideal = dka_index(df_results_ideal['iob'], df_results_ideal["sbr"])

        bg_cond = int(re.search("bg(\d)", sim_id).groups()[0])

        row = {
            "sim_id": sim_id,
            "lbgi_icgm": lbgi_icgm,
            "lbgi_ideal": lbgi_ideal,
            "lbgi_diff": lbgi_icgm - lbgi_ideal,
            "dkai_icgm": dkai_icgm,
            "dkai_ideal": dkai_ideal,
            "dkai_diff": dkai_icgm - dkai_ideal,
            "bg_condition": bg_cond,
            "true_start_bg": sim_json_info["patient"]["sensor"]["true_start_bg"],
            "start_bg_with_offset": sim_json_info["patient"]["sensor"]["start_bg_with_offset"]
        }

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    compute_dka_risk_tp_icgm(summary_df)
    compute_lbgi_risk_tp_icgm_negative_bias(summary_df)
    compute_lbgi_risk_tp_icgm_positive_bias(summary_df)


def compute_dka_risk_tp_icgm(summary_df):

    initially_ok_mask = summary_df["dkai_ideal"] == 0.0
    print(summary_df[initially_ok_mask]["dkai_icgm"].describe())


def compute_lbgi_risk_tp_icgm_negative_bias(summary_df):

    initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
    print(summary_df[initially_ok_mask]["lbgi_icgm"].describe())


def compute_lbgi_risk_tp_icgm_positive_bias(summary_df):

    dexcome_value_model = DexcomG6ValueModel(concurrency_table="TP_iCGM")

    expected_value = 0.0
    total_sims = 0
    for (low_true, high_true), (low_icgm, high_icgm) in [
        ((40, 60), (40, 60)),
        ((40, 60), (61, 80)),
        ((40, 60), (81, 120)),

        ((61, 80), (61, 80)),
        ((61, 80), (81, 120)),

        ((81, 120), (81, 120)),
        ((81, 120), (121, 160)),
        ((81, 120), (161, 200)),

        ((121, 160), (121, 160)),
        ((121, 160), (161, 200)),
        ((121, 160), (201, 250)),

        ((161, 200), (161, 200)),
        ((161, 200), (201, 250)),
        ((161, 200), (251, 300)),

        ((201, 250), (201, 250)),
        ((201, 250), (251, 300)),
        ((201, 250), (301, 350)),

        ((251, 300), (251, 300)),
        ((251, 300), (301, 350)),
        ((251, 300), (351, 400)),

        ((301, 350), (301, 350)),
        ((301, 350), (351, 400)),
        ((351, 400), (351, 450)),
    ]:
        true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)
        icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (summary_df["start_bg_with_offset"] <= high_icgm)

        initially_ok_mask = summary_df["lbgi_ideal"] == 0.0

        # Metric - "expected severity"
        severity = summary_df[true_mask & icgm_mask & initially_ok_mask]["lbgi_icgm"].mean()
        if np.isnan(severity):
            severity = 0.0
        num_total = len(summary_df[true_mask & icgm_mask & initially_ok_mask])

        total_sims += num_total

        p_dexcom_square = dexcome_value_model.get_joint_probability(low_true, low_icgm)

        expected_value += severity * p_dexcom_square

        print(low_true, high_true, low_icgm, high_icgm, num_total)
        print("P(lbgi > thresh | CB, true range, icgm range)", severity, "P(true range, icgm range)", p_dexcom_square, "\n")

    print("E[LBGI iCGM]", expected_value)
    print("Total sims", total_sims)


# %%
if __name__ == "__main__":

    scenarios_dir = "data/raw/icgm-sensitivity-analysis-scenarios-2020-07-10/"

    today_timestamp = datetime.datetime.now().strftime("%Y-%m-%d")
    working_dir = os.getcwd()
    save_dir = os.path.join(working_dir, "data/processed/icgm-sensitivity-analysis-results-" + today_timestamp)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print("Made director for results: {}".format(save_dir))

    vp_scenario_dict = load_vp_training_data(scenarios_dir)

    if 0:
        sim_batch_size = 2
        sim_batch_generator = build_icgm_sim_generator(vp_scenario_dict, sim_batch_size=sim_batch_size)

        start_time = time.time()
        for i, sim_batch in enumerate(sim_batch_generator):

            batch_start_time = time.time()

            all_results = run_simulations(
                sim_batch,
                save_dir=save_dir,
                save_results=True,
                num_procs=sim_batch_size
            )
            batch_total_time = (time.time() - batch_start_time) / 60
            run_total_time = (time.time() - start_time) / 60
            print("Batch {}".format(i))
            print("Minutes to build sim batch {} of {} sensors. Total minutes {}".format(batch_total_time, len(sim_batch), run_total_time))

            for sim_id, result_df in all_results.items():
                if "Ideal" in sim_id:
                    continue

                sim_id_icgm = sim_id

                try:
                    lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(all_results[sim_id_icgm]['bg'])
                    print("LBGI iCGM: {}".format(lbgi_icgm))
                except:
                    print("Bgs below zero.")

    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-02-positive_bias_with_requirements/"
    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-04/"

    # On compute-1
    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-03/"  # worst case negative bias, 887 sims
    result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-04/"  # temp basal case

    # plot_icgm_results(result_dir)

    plot_sensor_error_vs_risk(result_dir)

    # To check before running
    # 1. Sensor behavior model & properties
    # 2. Future tbg prob
