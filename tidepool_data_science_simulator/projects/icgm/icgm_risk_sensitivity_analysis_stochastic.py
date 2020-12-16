__author__ = "Cameron Summers"

import pdb
import sys
import time
import os
import datetime
import json
import copy
import re
from collections import defaultdict

import numpy as np
from numpy.random import RandomState
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.patient_for_icgm_sensitivity_analysis import VirtualPatientISA
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.evaluation.inspect_results import load_results, collect_sims_and_results, load_result
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results, plot_sim_icgm_paired

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

    # step = (icgm_high - icgm_low) / (num_samples)

    step = 5.0

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


def generate_sim_parser_starting_glucose(original_sim_parser):

    num_history_values = 6
    for true_start_glucose in range(40, 400, 5):

        new_sim_parser = copy.deepcopy(original_sim_parser)

        sensor_glucose_dates = new_sim_parser.sensor_glucose_history.datetimes[-num_history_values:]
        sensor_glucose_values = [true_start_glucose] * num_history_values
        new_sim_parser.sensor_glucose_history = GlucoseTrace(datetimes=sensor_glucose_dates, values=sensor_glucose_values)

        patient_glucose_dates = new_sim_parser.patient_glucose_history.datetimes[-num_history_values:]
        patient_glucose_values = [true_start_glucose] * num_history_values
        new_sim_parser.patient_glucose_history = GlucoseTrace(datetimes=patient_glucose_dates, values=patient_glucose_values)

        yield new_sim_parser


def build_icgm_sim_generator(vp_scenario_dict, sim_batch_size=30):
    """
    Build simulations for the FDA 510k Loop iCGM sensitivity analysis.

    Scenario files are on Compute-2 in Cameron Summers' copy of this code base.
    """
    analysis_type_list = [
        # "temp_basal_only",
        "correction_bolus",
        # "meal_bolus"
    ]

    sim_ctr = 0
    sims = {}
    for vp_idx, (vp_id, bg_scenario_dict) in enumerate(vp_scenario_dict.items()):
        print("VP", vp_idx)
        for bg_cond_id, scenario_filename in list(bg_scenario_dict.items())[:1]:

            scenario_path = os.path.join(scenarios_dir, scenario_filename)
            original_sim_parser = ScenarioParserCSV(scenario_path)

            for sim_parser in generate_sim_parser_starting_glucose(original_sim_parser):

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
                sampled_error_values = sample_uniformly_positive_error_cgm_ranges(t0_true_bg)
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


def sim_id_without_sensor(sim_id):
    if sim_id is None:
        return

    return re.sub("\.s.*\.", "", sim_id)


def plot_icgm_results(result_dir, sim_inspect_id=None):

    all_results = load_results(result_dir, ext="tsv", max_dfs=np.inf)

    sim_groups_to_plot = defaultdict(dict)
    ideal_sims = defaultdict(dict)
    for sim_id, result_df in all_results.items():
        sensor_group_id = sim_id_without_sensor(sim_id)
        if "Ideal" in sim_id:
            ideal_sims[sensor_group_id][sim_id] = result_df
        else:
            sim_groups_to_plot[sensor_group_id][sim_id] = result_df

    sim_inspect_group_id = sim_id_without_sensor(sim_inspect_id)
    for sim_group_id, sim_group_results in sim_groups_to_plot.items():

        if sim_inspect_id is not None and sim_inspect_group_id != sim_group_id:
            continue

        plot_sim_results(sim_group_results)

        # sim_group_results = {sim_id: results_df for i, (sim_id, results_df) in enumerate(sim_group_results.items()) if i == 10}
        # sim_group_results.update({sim_id: result_df for sim_id, result_df in ideal_sims[sim_group_id].items()})
        # plot_sim_icgm_paired(sim_group_results)


def compute_sim_summary_stats(result_dir):

    sim_results = collect_sims_and_results(result_dir, sim_id_pattern="vp.*bg.*.json", max_sims=100)

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
        filename_icgm, df_results = list(df_results_dict.items())[0]
        true_bg = df_results['bg']
        true_bg[true_bg < 1] = 1
        lbgi_icgm, hbgi_icgm, brgi_icgm = blood_glucose_risk_index(true_bg)
        dkai_icgm = dka_index(df_results['iob'], df_results["sbr"])

        df_results_ideal_dict = load_result(sim_json_info_match["result_path"])
        filename_ideal, df_results_ideal = list(df_results_ideal_dict.items())[0]
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
            "start_bg_with_offset": sim_json_info["patient"]["sensor"]["start_bg_with_offset"],
            "sbr": df_results["sbr"].values[0],
            "isf": df_results["isf"].values[0],
            "cir": df_results["cir"].values[0],
            "ylw": sim_json_info["controller"]["config"]["ylw"],
            "age": sim_json_info["controller"]["config"]["ylw"],
            "max_bolus_delivered": df_results["true_bolus"].max()
        }

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)

    summary_result_filepath = "./result_summary_{}.csv".format(datetime.datetime.now().isoformat())
    summary_df.to_csv(summary_result_filepath, sep="\t")
    print("Saved summary results to", summary_result_filepath)


def compute_risk_stats(summary_df):

    # compute_dka_risk_tp_icgm(summary_df)
    # compute_lbgi_risk_tp_icgm_negative_bias(summary_df)

    score_risk_table(summary_df)


def compute_dka_risk_tp_icgm(summary_df):

    initially_ok_mask = summary_df["dkai_ideal"] == 0.0
    print(summary_df[initially_ok_mask]["dkai_icgm"].describe())


def compute_lbgi_risk_tp_icgm_negative_bias(summary_df):

    initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
    print(summary_df[initially_ok_mask]["lbgi_icgm"].describe())
    print("2.5 LBGI percentile",
          summary_df[initially_ok_mask]["lbgi_icgm"].values.searchsorted(2.5)/len(summary_df[initially_ok_mask])*100)


class TPRiskTableRev7(object):

    def __init__(self):
        self.table = np.zeros(shape=(5, 5))

        self.table_severity_indices = {
            (0.0, 2.5): 0,
            (2.5, 5.0): 1,
            (5.0, 10.0): 2,
            (10.0, 20.0): 3,
            (20.0, np.inf): 4
        }

        self.table_probability_indices = {
            (14600000, np.inf): 0,
            (1460000, 14599999): 1,
            (14600, 1459999): 2,
            (146, 14599): 3,
            (0, 146): 4,
        }

        self.acceptability_regions = {
            (0, 0): "Yellow",
            (0, 1): "Red",
            (0, 2): "Red",
            (0, 3): "Red",
            (0, 4): "Red",
            (1, 0): "Green",
            (1, 1): "Yellow",
            (1, 2): "Red",
            (1, 3): "Red",
            (1, 4): "Red",
            (2, 0): "Green",
            (2, 1): "Yellow",
            (2, 2): "Yellow",
            (2, 3): "Red",
            (2, 4): "Red",
            (3, 0): "Green",
            (3, 1): "Green",
            (3, 2): "Yellow",
            (3, 3): "Red",
            (3, 4): "Red",
            (4, 0): "Green",
            (4, 1): "Green",
            (4, 2): "Green",
            (4, 3): "Yellow",
            (4, 4): "Yellow",
        }

    def get_probability_index(self, num_events_per_100k_person_years):

        for bounds in self.table_probability_indices.keys():
            if bounds[0] <= num_events_per_100k_person_years < bounds[1]:
                return self.table_probability_indices[bounds]

        raise Exception("Probability not in indices.")

    def get_severity_index(self, severity_lbgi):

        for bounds in self.table_severity_indices.keys():
            if bounds[0] <= severity_lbgi < bounds[1]:
                return self.table_severity_indices[bounds]

        raise Exception("Severity not in indices.")

    def add(self, severity_lbgi, num_events_per_100k_person_years):

        severity_idx = self.get_severity_index(severity_lbgi)
        prob_idx = self.get_probability_index(num_events_per_100k_person_years)

        self.table[prob_idx, severity_idx] += 1

    def is_problematic(self, severity_lbgi, num_events_per_100k_person_years):

        severity_idx = self.get_severity_index(severity_lbgi)
        prob_idx = self.get_probability_index(num_events_per_100k_person_years)
        problematic = False
        if self.acceptability_regions[(prob_idx, severity_idx)] == "Red":
            problematic = True

        return problematic

    def print(self):
        print(pd.DataFrame(self.table))


class PositiveBiasiCGMRequirements():

    def __init__(self, risk_table=TPRiskTableRev7(), true_ranges=None):

        self.p_corr_bolus_given_error = 3.0 / 288.0
        self.num_cgm_per_100k_person_years = 288 * 365 * 100000

        self.true_ranges = true_ranges
        if true_ranges is None:
            self.true_ranges = [
                (0, 40),
                (40, 60),
                (61, 80),
                (81, 120),
                (121, 160),
                (161, 200),
                (201, 250),
                (251, 300),
                (301, 350),
                (351, 400),
            ]

        self.dexcom_pediatric_value_model = DexcomG6ValueModel(concurrency_table="pediatric")
        total_data_points = np.sum(self.dexcom_pediatric_value_model.comparator_totals)
        self.p_true_pediatric = np.array([v / total_data_points for v in self.dexcom_pediatric_value_model.comparator_totals])

        self.risk_table = risk_table

    def fit_positive_bias_prob(self, summary_df):

        for i, (low_true, high_true) in enumerate(self.true_ranges):

            for (low_icgm, high_icgm) in self.true_ranges[i:]:

                initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
                true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)

                icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (
                            summary_df["start_bg_with_offset"] <= high_icgm)

                concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask
                sub_df = summary_df[concurrency_square_mask]

                p_error_max = self.fit_error_probability(sub_df)
                p_requirements = self.dexcom_pediatric_value_model.get_joint_probability(low_true, low_icgm)

                print(low_true, high_true, low_icgm, high_icgm, p_error_max, p_requirements)

    def fit_positive_bias_range_and_prob(self, summary_df):

        initially_ok_mask = summary_df["lbgi_ideal"] == 0.0
        # initially_ok_mask = summary_df["lbgi_ideal"] < 0.5

        for i, (low_true, high_true) in enumerate(self.true_ranges):

            true_mask = (summary_df["true_start_bg"] >= low_true) & (summary_df["true_start_bg"] <= high_true)

            for (low_icgm, high_icgm) in self.true_ranges[i:]:

                test_high_icgms = [high_icgm - i for i in range(0, 40, 1)]
                mitigation_probs = []

                for test_high_icgm in test_high_icgms:

                    icgm_mask = (summary_df["start_bg_with_offset"] >= low_icgm) & (
                            summary_df["start_bg_with_offset"] <= test_high_icgm)

                    concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask
                    sub_df = summary_df[concurrency_square_mask]

                    # plt.hist(sub_df["true_start_bg"], alpha=0.5, label="True")
                    # plt.hist(sub_df["start_bg_with_offset"], alpha=0.5, label="iCGM")
                    # plt.legend()
                    # plt.show()

                    p_error_max = self.fit_error_probability(sub_df)

                    mitigation_probs.append(p_error_max)
                    p_requirements = self.dexcom_pediatric_value_model.get_joint_probability(low_true, low_icgm)

                    print(low_true, high_true, low_icgm, test_high_icgm, p_error_max, p_requirements)
                    print("Num sims", np.sum(concurrency_square_mask))

                    if p_error_max is not None and p_error_max < p_requirements and (test_high_icgm - low_icgm) < 5:
                        a = 1

                plt.plot(test_high_icgms, mitigation_probs, label="Max P(True, iCGM)")
                plt.axhline(p_requirements, label="Dexcom P(True, iCGM)", linestyle="--")
                plt.legend()
                plt.show()

    def fit_error_probability(self, df, max_iters=20):

        high_bound = 1.0
        low_bound = 0.0

        lbgi_data = df["lbgi_icgm"]

        if len(lbgi_data) == 0:
            return None

        # Check if ok initially
        if self.is_mitigated(lbgi_data, high_bound):
            return high_bound

        num_iters = 0
        num_iters_not_mitigated = 0
        test_bounds = []
        while True:

            test_bound = (high_bound - low_bound) / 2.0

            if num_iters >= max_iters:
                # plt.plot(test_bounds)
                # plt.show()
                # print(num_iters_not_mitigated)
                return test_bound

            test_bounds.append(test_bound)

            if not self.is_mitigated(lbgi_data, test_bound):
                high_bound = test_bound
            else:
                low_bound = test_bound
                num_iters_not_mitigated += 1

            num_iters += 1

    def is_mitigated(self, lbgi_data, region_probability):

        num_total_sims = len(lbgi_data)

        for s_idx, severity_band in enumerate([(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)], 1):

            severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])
            num_sims_in_severity_band = len(lbgi_data[severity_mask])
            severity_prob = num_sims_in_severity_band / num_total_sims
            risk_prob_sim = severity_prob * self.p_corr_bolus_given_error * region_probability
            num_risk_events_sim = risk_prob_sim * self.num_cgm_per_100k_person_years

            if self.risk_table.is_problematic(severity_band[0], num_risk_events_sim):
                return False

        return True


def score_risk_table(summary_df):

    dexcome_value_model = DexcomG6ValueModel(concurrency_table="TP_iCGM")

    summary_df["vp_id"] = summary_df["sim_id"].apply(lambda sim_id: re.search("(vp.*).bg\d", sim_id).groups()[0])

    risk_table_per_error_bin_patient_prob = TPRiskTableRev7()
    risk_table_per_error_bin_sim_prob = TPRiskTableRev7()
    risk_table_per_sim = TPRiskTableRev7()

    patient_percentages = []
    lbgi_band = []

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

        concurrency_square_mask = true_mask & icgm_mask & initially_ok_mask

        p_error = dexcome_value_model.get_joint_probability(low_true, low_icgm)
        p_corr_bolus_given_error = 3 / 288
        num_cgm_per_100k_person_years = 288 * 365 * 100000

        num_total_sims = max(1, len(summary_df[concurrency_square_mask]))

        lbgi_data = summary_df[concurrency_square_mask]["lbgi_icgm"]

        num_total_patients = max(1, len(summary_df[concurrency_square_mask]["vp_id"].unique()))
        for s_idx, severity_band in enumerate([(0.0, 2.5), (2.5, 5.0), (5.0, 10.0), (10.0, 20.0), (20.0, np.inf)], 1):
            severity_mask = (lbgi_data >= severity_band[0]) & (lbgi_data < severity_band[1])

            num_patients_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask]["vp_id"].unique())
            num_sims_in_severity_band = len(summary_df[concurrency_square_mask][severity_mask])

            patient_prob = num_patients_in_severity_band / num_total_patients
            sim_prob = num_sims_in_severity_band / num_total_sims

            risk_prob_patient = patient_prob * p_corr_bolus_given_error * p_error
            risk_prob_sim = sim_prob * p_corr_bolus_given_error * p_error

            num_risk_events_patient = risk_prob_patient * num_cgm_per_100k_person_years
            num_risk_events_sim = risk_prob_sim * num_cgm_per_100k_person_years

            patient_percentages.append(patient_prob)
            lbgi_band.append(s_idx)

            if not np.isnan(num_risk_events_patient) and num_risk_events_patient > 0.0:
                risk_table_per_error_bin_patient_prob.add(severity_band[0], num_risk_events_patient)
                # print(num_risk_events_patient)

            if not np.isnan(num_risk_events_sim) and num_risk_events_sim > 0.0:
                risk_table_per_error_bin_sim_prob.add(severity_band[0], num_risk_events_sim)
                # print(num_risk_events_sim)

        # p_settings = 1/99.0
        # for i, row in summary_df[concurrency_square_mask].iterrows():
        #
        #     sim_severity = row["lbgi_icgm"]
        #
        #     # bg_error = max(10, row["start_bg_with_offset"] - row["true_start_bg"])
        #     # p_corr_bolus_given_error /= (bg_error / 10)
        #     # print(p_corr_bolus_given_error)
        #
        #     sim_prob = p_error * p_corr_bolus_given_error * p_settings
        #     num_events_per_100k_person_years = sim_prob * num_cgm_per_100k_person_years
        #     risk_table_per_sim.add(sim_severity, num_events_per_100k_person_years)
        #
        #     if risk_table_per_sim.is_problematic(sim_severity, num_events_per_100k_person_years):
        #         print(num_events_per_100k_person_years)

        # print("Num sims excluded", num_sims_excluded)

        total_sims += num_total_sims

        # print(low_true, high_true, low_icgm, high_icgm, num_total)
        # print("Severity:", severity, "P(true range, icgm range)", p_error, "\n")

    risk_table_per_error_bin_patient_prob.print()
    risk_table_per_error_bin_sim_prob.print()
    # risk_table_per_sim.print()

    print("Total sims", total_sims)

    # print(risk_severities)
    # print(risk_severities.values())
    # print([val / sum(list(risk_severities.values())) for val in risk_severities.values()])

    # plt.scatter(lbgi_band, patient_percentages)
    # plt.show()


def general_exploration():

    # initially_ok_mask = (summary_df_positive_bias_sims["lbgi_ideal"] == 0)
    # lbgi_range_mask = summary_df_positive_bias_sims["lbgi_icgm"] < 20.0
    df = summary_df_positive_bias_sims  # [initially_ok_mask]# & lbgi_range_mask]
    df["lbgi_diff_log"] = np.log2(df["lbgi_diff"])

    # df['category'] = pd.cut(df["lbgi_icgm"], bins=[0, 2.5, 5.0, 10.0, 20.0, np.inf], labels=["1", "2", "3", "4", "5"])
    df["category"] = pd.cut(df["lbgi_diff_log"], bins=[-np.inf, 1.321928, 2.321928, 3.321928, 4.321928, np.inf],
                            labels=["1", "2", "3", "4", "5"])
    df["error"] = df["start_bg_with_offset"] - df["true_start_bg"]
    df["sbr-isf"] = df["sbr"] * df["isf"]

    # sns.kdeplot(data=df, x="true_start_bg", y="start_bg_with_offset", hue="category", alpha=0.5, fill=True)
    # plt.plot(range(40, 400), range(40, 400), color="grey", linestyle="--")

    sns.kdeplot(data=df, x="error", hue="category")
    plt.show()



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
    result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-11/"

    # On compute-1
    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-03/"  # worst case negative bias, 887 sims
    # result_dir = "./data/processed/icgm-sensitivity-analysis-results-2020-12-04/"  # temp basal case

    # plot_icgm_results(result_dir, sim_inspect_id=None)

    # compute_sim_summary_stats(result_dir)

    sim_run_887_filename_pos_bias_corr_bolus = "result_summary_positive_bias.csv"
    sim_run_200k_filename = "result_summary_2020-12-13T05:55:01.004257.csv"
    summary_df_positive_bias_sims = pd.read_csv(sim_run_200k_filename, sep="\t")

    # Compute the risk table
    # compute_risk_stats(summary_df_positive_bias_sims)

    # Fit the requirements
    # requirements_model = PositiveBiasiCGMRequirements()
    # requirements_model.fit_positive_bias_prob(summary_df_positive_bias_sims)
    # requirements_model.fit_positive_bias_range_and_prob(summary_df_positive_bias_sims)

    # To check before running
    # 1. Sensor behavior model & properties
    # 2. Future tbg prob
