__author__ = "Cameron Summers"

import os
import re
from collections import defaultdict
import logging
import json

import datetime

import pandas as pd
import numpy as np
import scipy.stats as st

from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, BasalSchedule24hr, TargetRangeSchedule24hr
)
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV, PumpConfig, PatientConfig, \
    SensorConfig
from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import CONTROLLER_MODEL_NAME_MAP
from tidepool_data_science_simulator.makedata.make_patient import SINGLE_SETTING_DURATION, DATETIME_DEFAULT, \
    get_canonical_glucose_history
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, ActionTimeline
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace, Bolus, Carb
)
from tidepool_data_science_simulator.utils import DATA_DIR

RAW_DATA_DIR = os.path.join(DATA_DIR, "raw/")
ICGM_SCENARIOS_DIR = os.path.join(DATA_DIR, "raw/icgm-sensitivity-analysis-scenarios-2020-07-10/")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed/")

ICGM_SETTINGS_FILEPATH = os.path.join(PROCESSED_DATA_DIR, "icgm", "icgm_patient_settings.json")

logger = logging.getLogger(__name__)


def sample_weight_kg_by_age(age, random_state):
    """
    Get a weight using CDC growth data tables by age.

    https://www.cdc.gov/growthcharts/percentile_data_files.htm
    """
    df = pd.read_csv(os.path.join(RAW_DATA_DIR, "cdc_weight_by_age_2-20.csv"))

    percentile = random_state.uniform()
    Z = st.norm.ppf(percentile)

    if age > 20:
        age = 20
        logger.warning("Age provided is being capped to 20 due to low variability among adults.")

    # Since things are months based, just take the first row matching the year.
    weight_stats_row = df[df["Ageyrs"].astype(int) == age].iloc[0]
    M, S, L = weight_stats_row[["M", "S", "L"]].values

    # As per instructions on CDC website for recovering weight from params
    weight_kg = M * (1.0 + L * S * Z) ** (1 / L)

    return weight_kg


def sample_total_daily_dose_by_age(age, random_state):
    """
    Get total units given in a day based on Tidepool blog post. Initially just
    returning the median, but once we get the distributions can sample from them.

    https://www.tidepool.org/blog/lets-talk-about-your-insulin-pump-data

    Returns
    -------
        float: Number of units
    """

    tdd_units = None
    if 0 <= age < 6:
        tdd_units = 15
    elif 6 <= age < 9:
        tdd_units = 18
    elif 9 <= age < 12:
        tdd_units = 28
    elif 12 <= age < 15:
        tdd_units = 46
    elif 15 <= age < 18:
        tdd_units = 50
    elif 18 <= age < 21:
        tdd_units = 42
    elif 21 <= age < 25:
        tdd_units = 44
    elif 25 <= age < 30:
        tdd_units = 45
    elif 30 <= age < 35:
        tdd_units = 39
    elif 35 <= age < 40:
        tdd_units = 45
    elif 40 <= age < 50:
        tdd_units = 40
    elif 50 <= age < 60:
        tdd_units = 41
    elif 60 <= age < 70:
        tdd_units = 37
    elif 70 <= age:
        tdd_units = 27

    return tdd_units


class iCGMPatientSettings():

    def __init__(self, settings_export):
        self.settings_export = settings_export
        self.basal_rate = float(settings_export["pump"]["basal_schedule"]["schedule"][0]["setting"])
        self.cir = float(settings_export["pump"]["carb_ratio_schedule"]["schedule"][0]["setting"])
        self.isf = float(settings_export["pump"]["insulin_sensitivity_schedule"]["schedule"][0]["setting"])

        self.age = settings_export["controller"]["age"]

        self.patient_id = settings_export["patient_id"]


def get_icgm_patient_config(icgm_patient_obj, random_state, t0=DATETIME_DEFAULT):
    """
    Get canonical patient config

    Parameters
    ----------
    icgm_patient_obj (iCGM

    Returns
    -------
    PatientConfig
    """
    patient_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    patient_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    true_bg_history = get_canonical_glucose_history(t0)

    total_hours_in_day = 24
    hours_in_day = range(total_hours_in_day)
    basal_rate = icgm_patient_obj.get_basal_rate()
    cir = icgm_patient_obj.get_carb_insulin_ratio()
    isf = icgm_patient_obj.get_insulin_sensitivity_factor()

    basal_jitter = 0.2
    basal_rates = [random_state.uniform(basal_rate - basal_jitter, basal_rate + basal_jitter) for _ in hours_in_day]

    cir_jitter = 2
    carb_ratios = [random_state.uniform(cir - cir_jitter, cir + cir_jitter) for _ in hours_in_day]

    isf_jitter = 2
    isfs = [random_state.uniform(isf - isf_jitter, isf + isf_jitter) for _ in hours_in_day]

    start_times_hourly = [datetime.time(hour=i, minute=0, second=0) for i in hours_in_day]
    durations_min_per_hour = [SINGLE_SETTING_DURATION / total_hours_in_day for _ in hours_in_day]

    logger.debug("Patient Mean Metabolism Settings. BR: {:.2f} CIR: {:.2f} ISF: {:.2f}".format(np.mean(basal_rates),
                                                                                               np.mean(carb_ratios),
                                                                                               np.mean(isfs)))

    patient_config = PatientConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=start_times_hourly,
            values=[BasalRate(rate, "mg/dL") for rate in basal_rates],
            duration_minutes=durations_min_per_hour
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=start_times_hourly,
            values=[CarbInsulinRatio(carb_ratio, "g/U") for carb_ratio in carb_ratios],
            duration_minutes=durations_min_per_hour
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=start_times_hourly,
            values=[InsulinSensitivityFactor(isf, "md/dL / U") for isf in isfs],
            duration_minutes=durations_min_per_hour
        ),
        glucose_history=true_bg_history,
        carb_event_timeline=patient_carb_timeline,
        bolus_event_timeline=patient_bolus_timeline,
        action_timeline=ActionTimeline(),
    )

    patient_config.recommendation_accept_prob = 1.0  # Does not accept any bolus recommendations
    patient_config.min_bolus_rec_threshold = 0.0  # Minimum size of bolus to accept
    patient_config.recommendation_meal_attention_time_minutes = 1e12  # Time since meal to take recommendations

    return t0, patient_config


def get_patients_by_age(min_age, max_age):
    icgm_patient_objs = get_icgm_patient_settings_objects()

    filtered_patient_settings = []
    for patient_obj in icgm_patient_objs:
        age = patient_obj.get_age()
        if min_age <= age <= max_age:
            filtered_patient_settings.append(patient_obj)

    return filtered_patient_settings


def get_icgm_patient_settings_objects():
    """
    Load exported json file that contains settings info for iCGM patients.

    Returns
    -------
        list: list of dicts of settings
    """
    settings_export_json = json.load(open(ICGM_SETTINGS_FILEPATH, "r"))

    icgm_patients = []
    for settings in settings_export_json:
        patient_obj = iCGMPatientSettings(settings)
        icgm_patients.append(patient_obj)

    return icgm_patients


def get_old_icgm_tidepool_patient_files_dict():
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

    file_names = os.listdir(ICGM_SCENARIOS_DIR)
    all_scenario_files = [filename for filename in file_names if filename.endswith('.csv')]
    logger.debug("Num scenario files: {}".format(len(all_scenario_files)))

    patient_scenario_dict = defaultdict(dict)
    for filename in all_scenario_files:
        vp_id = re.search("train_(.*)\.csv.+", filename).groups()[0]
        bg_condition = re.search("condition(\d)", filename).groups()[0]
        patient_scenario_dict[vp_id][bg_condition] = filename

    return patient_scenario_dict


def export_icgm_scenario_metadata_from_scenario_files():
    """
    Load old iCGM scenario files and export use settings for general usage.
    """
    patient_scenario_dict = get_old_icgm_tidepool_patient_files_dict()

    scenario_metadata = []
    for vp_idx, (vp_id, bg_scenario_dict) in enumerate(patient_scenario_dict.items()):
        for bg_cond_id, scenario_filename in list(bg_scenario_dict.items())[:1]:
            scenario_path = os.path.join(ICGM_SCENARIOS_DIR, scenario_filename)
            scenario_parser = ScenarioParserCSV(scenario_path)

            pump_config = scenario_parser.get_pump_config()
            controller_config = scenario_parser.get_controller_config()

            metadata = {
                "patient_id": vp_id,
                "filename": scenario_filename,
                "pump": pump_config.get_info_stateless(),
                "controller": controller_config.get_info_stateless()
            }
            scenario_metadata.append(metadata)
            logger.debug("processing {}".format(vp_idx))

    json.dump(scenario_metadata, open(ICGM_SETTINGS_FILEPATH, "w"), indent=4)


def transform_icgm_json_to_v2_parser():

    settings_export_json = json.load(open(ICGM_SETTINGS_FILEPATH, "r"))

    date_str_format = "%m/%d/%Y %H:%M:%S"  # ref: "8/15/2019 12:00:00"

    sim_json_configs = []
    for settings_export in settings_export_json:

        patient_id = settings_export["patient_id"]
        basal_rate = float(settings_export["pump"]["basal_schedule"]["schedule"][0]["setting"])
        cir = float(settings_export["pump"]["carb_ratio_schedule"]["schedule"][0]["setting"])
        isf = float(settings_export["pump"]["insulin_sensitivity_schedule"]["schedule"][0]["setting"])

        age = settings_export["controller"]["age"]
        if age < 6:
            continue

        start_time = datetime.datetime(2019, 8, 15, 12, 0, 0)

        metabolism_settings = {
            "insulin_sensitivity_factor": {
                "start_times": ["0:00:00"],
                "values": [isf]
            },
            "carb_insulin_ratio": {
                "start_times": ["0:00:00"],
                "values": [cir]
            },
            "basal_rate": {
                "start_times": ["0:00:00"],
                "values": [basal_rate]
            }
        }

        num_historical_values = 137

        earliest_datetime = start_time - datetime.timedelta(minutes=5*num_historical_values)
        glucose_history = {
            "datetime": {i: (earliest_datetime + datetime.timedelta(minutes=5*i)).strftime(date_str_format) for i in
                         range(1, num_historical_values + 1)},
            "value": {i: 0 for i in range(1, num_historical_values + 1)},  # Expecting to override this later
            "unit": {i: "mg/dL" for i in range(1, num_historical_values + 1)}
        }

        target_range = {
            "start_times": ["0:00:00"],
            "lower_values": [90],
            "upper_values": [100]
        }

        settings_export["controller"]["model"] = "rapid_acting_adult"
        # settings_export["controller"]["retrospective_correction_enabled"] = False
        json_config_v2 = {
                             "sim_id": "iCGM_Positive_Bias_{}_age={}".format(patient_id, age),
                             "patient_id": patient_id,
                             "time_to_calculate_at": start_time.strftime(date_str_format),
                             "duration_hours": 8.0,
                             "offset_applied_to_dates": 0,
                             "patient": {
                                 "age": age,
                                 "sensor": {
                                     "glucose_history": glucose_history
                                 },
                                 "pump": {
                                     "metabolism_settings": metabolism_settings,
                                     "bolus_entries": [
                                         {
                                             "type": "bolus",
                                             "time": "8/15/2019 11:55:00",
                                             "value": 0
                                         }
                                     ],
                                     "carb_entries": [
                                         {
                                             "type": "carb",
                                             "start_time": "8/15/2019 11:55:00",
                                             "value": 0
                                         }
                                     ],
                                     "target_range": target_range
                                 },
                                 "patient_model": {
                                     "metabolism_settings": metabolism_settings,
                                     "glucose_history": glucose_history,
                                     "bolus_entries": [
                                         {
                                             "type": "bolus",
                                             "time": "8/15/2019 11:55:00",
                                             "value": 0
                                         }
                                     ],
                                     "carb_entries": [
                                         {
                                             "type": "carb",
                                             "start_time": "8/15/2019 11:55:00",
                                             "value": 0
                                         }
                                     ]
                                 }
                             },
                             "controller": {
                                 "id": "pyloopkit_v1",
                                 "settings": settings_export["controller"]
                             }
                         }

        sim_json_configs.append(json_config_v2)

    return sim_json_configs


if __name__ == "__main__":
    export_icgm_scenario_metadata_from_scenario_files()
