__author__ = "Cameron Summers"

import os
import pdb
import json
import time
import copy
import logging
import argparse
import re
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns


print("***************** Warning: Overriding Pyloopkit with Local Copy ***************************")
import sys


# For easier dev
local_pyloopkit_path = "/Users/csummers/dev/PyLoopKit/"
if not os.path.isdir(local_pyloopkit_path):
    local_pyloopkit_path = "/mnt/cameronsummers/dev/PyLoopKit/"
assert os.path.isdir(local_pyloopkit_path)
sys.path.insert(0, local_pyloopkit_path)

# Setup Logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
LOG_FILENAME = "sim.log"
filehandler = logging.FileHandler(LOG_FILENAME)
logger.addHandler(filehandler)

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import ActionTimeline
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, \
    get_pump_config, get_variable_risk_patient_config
from tidepool_data_science_simulator.utils import timing, save_df, get_sim_results_save_dir

from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, percent_values_ge_70_le_180

from numpy.random import RandomState

SEED = 1234567890


def get_new_random_state():
    return RandomState(SEED)


@timing
def compare_physiologic_bg_change_cap(save_dir, save_results, plot_results=False, dry_run=False):
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    logger.debug("Random Seed: {}".format(SEED))

    num_patients = 30
    rate_caps = [3.0, 5.0, 7.0, 9.0, 1e12]
    duration_hrs = 4*7*24
    if dry_run:
        num_patients = 1
        rate_caps = [1e12, 3.0]
        duration_hrs = 12

    logger.debug("Running {} patients. {} rate caps. {} hours".format(num_patients, len(rate_caps), duration_hrs))

    # Setup Patients
    patient_random_state = get_new_random_state()  # Single instance generates different patients
    sims = {}
    for i in range(num_patients):
        t0, sensor_config = get_canonical_sensor_config()
        sensor_config.std_dev = patient_random_state.uniform(5, 15)

        t0, patient_config = get_variable_risk_patient_config(patient_random_state)

        patient_config.recommendation_accept_prob = patient_random_state.uniform(0.8, 0.99)
        patient_config.min_bolus_rec_threshold = patient_random_state.uniform(0.4, 0.6)
        patient_config.remember_meal_bolus_prob = patient_random_state.uniform(0.9, 1.0)
        patient_config.correct_bolus_bg_threshold = patient_random_state.uniform(140, 190)
        patient_config.correct_bolus_delay_minutes = patient_random_state.uniform(20, 40)
        patient_config.correct_carb_bg_threshold = patient_random_state.uniform(70, 90)
        patient_config.correct_carb_delay_minutes = patient_random_state.uniform(5, 15)
        patient_config.carb_count_noise_percentage = patient_random_state.uniform(0.1, 0.25)
        patient_config.report_bolus_probability = patient_random_state.uniform(1.0, 1.0)
        patient_config.report_carb_probability = patient_random_state.uniform(0.95, 1.0)

        patient_config.action_timeline = ActionTimeline()

        t0, pump_config = get_pump_config(patient_random_state)

        # Setup Controllers
        for max_rate in rate_caps:
            sim_random_state = get_new_random_state()

            sensor = NoisySensor(time=t0, sensor_config=sensor_config, random_state=sim_random_state)
            pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

            vp = VirtualPatientModel(
                time=t0,
                pump=pump,
                sensor=sensor,
                metabolism_model=SimpleMetabolismModel,
                patient_config=patient_config,
                random_state=sim_random_state,
                id=i
            )

            t0, controller_config = get_canonical_controller_config()
            controller_config.controller_settings["max_physiologic_slope"] = max_rate
            controller = LoopController(time=t0, controller_config=controller_config, random_state=sim_random_state)
            controller.name = "PyloopKit_BG_Change_Max={}".format(max_rate)

            # Setup Sims
            sim_id = "{}_{}".format(vp.name, controller.name)
            sim = Simulation(
                time=t0,
                duration_hrs=duration_hrs,  # 4 weeks
                virtual_patient=vp,
                sim_id=sim_id,
                controller=controller,
                multiprocess=True,
                random_state=sim_random_state
            )
            sims[sim_id] = sim

    # Run the sims
    num_sims = len(sims)
    sim_ctr = 1
    start_time = time.time()
    num_procs = 20
    running_sims = {}
    for sim_id, sim in sims.items():
        logger.debug("Running: {}. {} of {}".format(sim_id, sim_ctr, num_sims))
        sim.start()
        running_sims[sim_id] = sim

        if len(running_sims) >= num_procs or sim_ctr >= num_sims:
            all_results = {id: sim.queue.get() for id, sim in running_sims.items()}
            [sim.join() for id, sim in running_sims.items()]
            for id, sim in running_sims.items():
                info = sim.get_info_stateless()
                json.dump(info, open(os.path.join(results_dir, "{}.json".format(id)), "w"), indent=4)
            running_sims = {}

            logger.debug("Batch run time: {:.2f}m".format((time.time() - start_time) / 60.0))
            for sim_id, results_df in all_results.items():
                try:
                    lbgi, hbgi, brgi = blood_glucose_risk_index(results_df['bg'])
                    summary_str = "Sim {}. LBGI: {} HBGI: {} BRGI: {}".format(sim_id, lbgi, hbgi, brgi)
                    logger.debug(summary_str)
                except:
                    logger.debug("Exception in summary stats, passing {}...".format(sim_id))

                # TMP - debugging random stream sync
                print(results_df.iloc[-1]["randint"])

                if save_results:
                    save_df(results_df, sim_id, save_dir)

            if plot_results:
                plot_sim_results(all_results, save=False)

        sim_ctr += 1

    logger.debug("Full run time: {:.2f}m".format((time.time() - start_time) / 60.0))
    os.rename(LOG_FILENAME, os.path.join(results_dir, LOG_FILENAME))


def analyze_results(result_dir):

    results = []
    for root, dirs, files in os.walk(result_dir, topdown=False):
        for file in files:
            results_df = pd.read_csv(os.path.join(root, file), sep="\t")
            try:
                lbgi, hbgi, brgi = blood_glucose_risk_index(results_df['bg'])
                tir = percent_values_ge_70_le_180(results_df['bg'])
            except:
                lbgi, hbgi, brgi = (None, None, None)

            vp = re.search("vp\d+", file).group()
            rate = 11.11
            if re.search("Max=\d", file):
                rate = float(re.search("Max=(\d)", file).groups()[0])
            print(vp, rate)

            row = {
                "vp": vp,
                "rate": rate,
                "lbgi": lbgi,
                "hbgi": hbgi,
                "brgi": brgi,
                "tir": tir
            }
            results.append(row)

    summary_df = pd.DataFrame(results)

    print(summary_df.groupby("rate")["lbgi"].median())
    print(summary_df.groupby("rate")["hbgi"].median())
    print(summary_df.groupby("rate")["tir"].median())

    sns.lmplot("rate", y="tir", hue="vp", data=summary_df)
    plt.show()


def validate_random_number_streams(result_dir):

    vp_dfpath_dict = defaultdict(list)
    for root, dirs, files in os.walk(result_dir, topdown=False):
        for file in sorted(files):
            if re.search("vp\d.*.json", file):
                sim_info = json.load(open(os.path.join(root, file), "r"))
                sim_id = sim_info["sim_id"]
                vp = sim_info["patient"]["name"]
                df_file = [fn for fn in files if sim_id in fn and ".tsv" in fn][0]
                df_path = os.path.join(root, df_file)
                vp_dfpath_dict[vp].append(df_path)

    for vp, path_list in vp_dfpath_dict.items():
        for path1, path2 in combinations(path_list, 2):
            df1 = pd.read_csv(path1, sep="\t")
            df2 = pd.read_csv(path2, sep="\t")

            if sum(abs(df1.iloc[1:]["randint"] - df2.iloc[1:]["randint"]) != 0):
                print(path1, path2)
                print(np.where(df1, df1["randint"] != df2["randint"])[:10])
                break

    # print(rate_caps_results_dict)
    print(vp_dfpath_dict)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--action")#, choices=["simulate", "analyze"])
    parser.add_argument("-d", "--result_dir", help="simulation result directory to analyze")

    args = parser.parse_args()
    action = args.action
    result_dir = args.result_dir

    if action == "simulate":
        results_dir = get_sim_results_save_dir()
        compare_physiologic_bg_change_cap(save_dir=results_dir, save_results=True, plot_results=True, dry_run=True)
    elif action == "analyze":
        analyze_results(result_dir)
    else:
        validate_random_number_streams(result_dir)

