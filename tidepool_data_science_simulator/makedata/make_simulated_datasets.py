__author__ = "Cameron Summers"

import os
import datetime
import argparse
import pathlib
import json

import numpy as np
import pandas as pd

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.models.patient import VirtualPatientModel

from tidepool_data_science_simulator.models.simulation import BasalSchedule24hr, SingleSettingSchedule24Hr
from tidepool_data_science_simulator.models.measures import BasalRate, CarbInsulinRatio, InsulinSensitivityFactor


def make_patient_datasets(save_dir, num_patients=100, num_days=10, dataset_seed=1234, estimated_settings=None):
    """
    Use simulator to generate and save patient data sets. If estimated settings are provided
    override and run simulation.

    Parameters
    ----------
    save_dir
    num_patients
    num_days
    dataset_seed
    """
    sims = dict()

    # Run simulation to generate patient datasets
    np.random.seed(dataset_seed)
    for i in range(num_patients):
        patient_seed = np.random.randint(1e8, 9e8)

        t0, sim = get_canonical_simulation(
            patient_class=VirtualPatientModel,
            duration_hrs=24 * num_days,
            multiprocess=True
        )

        sim.seed = patient_seed
        sample_parameters(t0, sim)

        # If there are settings estimates, overwrite sampled distributions
        if estimated_settings is not None:
            apply_estimated_settings(t0, sim, estimated_settings)

        sims[sim.seed] = sim
        sim.start()

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

    # Do any preprocess prior to saving
    preprocess_datasets(all_results)
    save_dataset(save_dir, sims, all_results, dataset_seed)


def save_dataset(save_dir, sims, all_results, dataset_seed):
    # ================= Gather and save results =================
    patient_summaries = []

    for sim_id, results_df in all_results.items():
        # TODO: surface all params

        patient_dir = os.path.join(save_dir, "patient_{}".format(sim_id))
        os.makedirs(patient_dir)

        sim = sims[sim_id]

        # Summary Results
        total_daily_basal = round(np.sum(results_df["delivered_basal_insulin"]) / num_days, 2)
        total_daily_bolus = round(np.sum(results_df["bolus"]) / num_days, 2)
        dataset_row = {
            "id": sim_id,
            "dataset_seed": dataset_seed,
            "patient_seed": sim.seed,
            "scheduled_basal_rate": results_df["sbr"].values[-1],
            "isf": results_df["isf"].values[-1],
            "cir": results_df["cir"].values[-1],
            "pump_scheduled_basal_rate": results_df["pump_sbr"].values[-1],
            "pump_isf": results_df["pump_isf"].values[-1],
            "pump_cir": results_df["pump_cir"].values[-1],
            "carb_count_noise_percentage": sim.virtual_patient.carb_count_noise_percentage,
            "correct_bolus_bg_threshold": sim.virtual_patient.correct_bolus_bg_threshold,
            "correct_bolus_delay_minutes": sim.virtual_patient.correct_bolus_delay_minutes,
            "correct_carb_bg_threshold": sim.virtual_patient.correct_carb_bg_threshold,
            "correct_carb_delay_minutes": sim.virtual_patient.correct_carb_delay_minutes,
            "total_daily_bolus": total_daily_bolus,
            "total_daily_basal": total_daily_basal,
            "total_daily_dose": total_daily_basal + total_daily_bolus,
            "total_daily_carbs": round(np.sum(results_df["carb_value"]) / num_days, 1),
            "bg_median": round(np.median(results_df["bg"]), 1),
            "bg_variance": round(np.var(results_df["bg"]), 1)
        }
        patient_summaries.append(dataset_row)

        print(round(np.var(results_df["bg"]), 1))
        results_df.to_csv(os.path.join(patient_dir, "timeline.tsv"), sep="\t")

    patient_summaries_df = pd.DataFrame(patient_summaries)

    patient_summaries_df.to_csv(os.path.join(save_dir, "patient_summaries.tsv"), sep="\t")

    # Save dataset metadata
    with open(os.path.join(save_dir, "dataset_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump({
            "dataset_seed": dataset_seed,
            "num_days": num_days,
            "num_patients": num_patients,
            "create_timestamp": datetime.datetime.now().isoformat()

        }, f, ensure_ascii=False, indent=4)

    print("Data saved to {}".format(save_dir))


def sample_parameters(t0, sim):

    # Goal for below: Learn all parameterizations from Tidepool Data

    # === Patient Behavior Parameterization ===
    age = np.random.randint(1, 80)
    remember_meal_bolus_prob = round(np.random.uniform(0.6, 0.95), 2)
    correct_bolus_bg_threshold = round(np.random.uniform(130, 200))
    correct_bolus_delay_minutes = round(np.random.uniform(15, 45))
    correct_carb_bg_threshold = round(np.random.uniform(75, 100))
    correct_carb_delay_minutes = round(np.random.uniform(5, 20))
    carb_count_noise_percentage = round(np.random.uniform(0.05, 0.2), 2)

    sim.virtual_patient.remember_meal_bolus_prob = remember_meal_bolus_prob
    sim.virtual_patient.correct_bolus_bg_threshold = correct_bolus_bg_threshold
    sim.virtual_patient.correct_bolus_delay_minutes = correct_bolus_delay_minutes
    sim.virtual_patient.correct_carb_bg_threshold = correct_carb_bg_threshold
    sim.virtual_patient.correct_carb_delay_minutes = correct_carb_delay_minutes
    sim.virtual_patient.carb_count_noise_percentage = carb_count_noise_percentage
    sim.virtual_patient.age = age

    # ==== Pump and Patient Treatment Settings ====
    # Basal Rate
    pump_basal_rate = round(np.random.uniform(0.2, 0.8), 2)
    patient_basal_rate = round(np.random.normal(pump_basal_rate, 0.05), 2)
    sim.virtual_patient.pump.pump_config.basal_schedule = \
        SingleSettingSchedule24Hr(t0, "Basal Rate",BasalRate(pump_basal_rate, "U/hr"))
    sim.virtual_patient.patient_config.basal_schedule = \
        SingleSettingSchedule24Hr(t0, "Basal Rate", BasalRate(patient_basal_rate, "U/hr"))

    # ISF
    pump_isf = round(np.random.uniform(120, 200))
    patient_isf = round(np.random.normal(pump_isf, 10))
    sim.virtual_patient.pump.pump_config.insulin_sensitivity_schedule = \
        SingleSettingSchedule24Hr(t0, "ISF", InsulinSensitivityFactor(pump_isf, "mg/dL/U"))
    sim.virtual_patient.patient_config.insulin_sensitivity_schedule = \
        SingleSettingSchedule24Hr(t0, "ISF", InsulinSensitivityFactor(patient_isf, "mg/dL/U"))

    # Carb ratio
    pump_cir = round(np.random.uniform(10, 25), 1)
    patient_cir = round(np.random.normal(pump_cir, 5), 1)
    sim.virtual_patient.pump.pump_config.carb_ratio_schedule = \
        SingleSettingSchedule24Hr(t0, "CIR", CarbInsulinRatio(pump_cir, "g/U"))
    sim.virtual_patient.patient_config.carb_ratio_schedule = \
        SingleSettingSchedule24Hr(t0, "CIR", CarbInsulinRatio(patient_cir, "g/U"))


def apply_estimated_settings(t0, sim, estimated_settings):
    """
    Overwrite settings on pump with estimated settings.

    Parameters
    ----------
    t0
    sim
    estimated_settings
    """
    if sim.seed not in estimated_settings.index:
        raise Exception("No setting estimate found for simulation ID {}".format(sim.seed))

    pump_basal_rate = estimated_settings.loc[sim.seed]["basal_rate"]
    pump_isf = estimated_settings.loc[sim.seed]["ISF"]
    pump_cir = estimated_settings.loc[sim.seed]["CIR"]

    sim.virtual_patient.pump.pump_config.basal_schedule = \
        SingleSettingSchedule24Hr(t0, "Basal Rate", BasalRate(pump_basal_rate, "U/hr"))

    sim.virtual_patient.pump.pump_config.insulin_sensitivity_schedule = \
        SingleSettingSchedule24Hr(t0, "ISF", InsulinSensitivityFactor(pump_isf, "mg/dL/U"))

    sim.virtual_patient.pump.pump_config.carb_ratio_schedule = \
        SingleSettingSchedule24Hr(t0, "CIR", CarbInsulinRatio(pump_cir, "g/U"))


def preprocess_datasets(all_results):
    """
    Process results of raw simulation, e.g. drop data

    Parameters
    ----------
    all_results

    Returns
    -------
    dict
    """

    return all_results


def load_estimated_setting(estimated_settings_path):

    estimated_settings_df = pd.read_csv(estimated_settings_path)

    assert "id" in estimated_settings_df.columns
    assert "basal_rate" in estimated_settings_df.columns
    assert "ISF" in estimated_settings_df.columns
    assert "CIR" in estimated_settings_df.columns

    estimated_settings_df = estimated_settings_df.set_index("id")

    return estimated_settings_df


def evaluate_estimated_settings(original_dataset_path, estimated_dataset_path, patient_regex="patient_[0-9]*"):

    patient_dirs_orig = {p.name: p for p in pathlib.Path(original_dataset_path).iterdir() if p.is_dir() and p.match(patient_regex)}
    patient_dirs_estimated = {p.name: p for p in pathlib.Path(estimated_dataset_path).iterdir() if p.is_dir() and p.match(patient_regex)}

    assert sorted(patient_dirs_orig.keys()) == sorted(patient_dirs_estimated.keys())

    for patient_dirname in patient_dirs_orig.keys():
        timeline_orig_df = pd.read_csv(patient_dirs_orig[patient_dirname].joinpath("timeline.tsv"), sep='\t')
        timeline_est_df = pd.read_csv(patient_dirs_estimated[patient_dirname].joinpath("timeline.tsv"), sep='\t')

        print("METRICS: Original vs Estimated")
        print(round(timeline_orig_df['bg'].mean(), 1), round(timeline_est_df['bg'].mean()), 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("-a", "--action", choices=["create", "eval"], default="create")
    parser.add_argument("--save_dir", default="./", type=str)
    parser.add_argument("-n", "--num_patients", default=3, type=int)
    parser.add_argument("-d", "--num_days", default=10, type=int)
    parser.add_argument("-s", "--seed", default=1234, type=int)
    parser.add_argument("-e", "--estimated_settings_path", type=str)
    parser.add_argument("--eval_dataset_path", type=str)

    args = parser.parse_args()

    action = args.action

    if action == "create":
        num_patients = args.num_patients
        dataset_seed = args.seed
        num_days = args.num_days
        save_dir = args.save_dir

        save_dir = os.path.join(save_dir, "dataset_seed{}".format(dataset_seed))
        make_patient_datasets(save_dir, num_patients, num_days=num_days, dataset_seed=dataset_seed)

    elif action == "eval":
        eval_dataset_path = args.eval_dataset_path
        estimated_settings_path = args.estimated_settings_path

        if eval_dataset_path is None or estimated_settings_path is None:
            raise Exception("Eval requires dataset path and estimated settings path.")

        dataset_metadata = json.load(open(os.path.join(eval_dataset_path, "dataset_metadata.json")))
        num_patients = dataset_metadata["num_patients"]
        num_days = dataset_metadata["num_days"]
        dataset_seed = dataset_metadata["dataset_seed"]

        estimated_settings = load_estimated_setting(estimated_settings_path)

        save_dir = "../../data/raw/evaluated_datasets/dataset_seed{}_{}/".format(dataset_seed, datetime.datetime.now().isoformat())
        make_patient_datasets(save_dir, num_patients, num_days=num_days, dataset_seed=dataset_seed, estimated_settings=estimated_settings)
        evaluate_estimated_settings(eval_dataset_path, save_dir)


