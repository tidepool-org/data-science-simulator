__author__ = "Cameron Summers"

import os
import re
from collections import defaultdict
import logging
import json

import datetime

from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, BasalSchedule24hr, TargetRangeSchedule24hr
)
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV, PumpConfig, PatientConfig, SensorConfig
from tidepool_data_science_simulator.makedata.make_patient import SINGLE_SETTING_DURATION, DATETIME_DEFAULT, get_canonical_glucose_history
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, ActionTimeline
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace, Bolus, Carb
)

THIS_DIR = os.path.dirname(__file__)
ICGM_SCENARIOS_DIR = os.path.join(THIS_DIR, "../../data/raw/icgm-sensitivity-analysis-scenarios-2020-07-10/")
PROCESSED_DATA_DIR = os.path.join(THIS_DIR, "../../data/processed/")

ICGM_SETTINGS_FILEPATH = os.path.join(PROCESSED_DATA_DIR, "icgm", "icgm_patient_settings.json")


logger = logging.getLogger(__name__)


class iCGMPatient():

    def __init__(self, settings_export):

        self.settings_export = settings_export
        self.basal_rate = float(settings_export["pump"]["basal_schedule"]["schedule"][0]["setting"])
        self.cir = float(settings_export["pump"]["carb_ratio_schedule"]["schedule"][0]["setting"])
        self.isf = float(settings_export["pump"]["insulin_sensitivity_schedule"]["schedule"][0]["setting"])

        self.age = settings_export["controller"]["age"]

        self.patient_id = settings_export["patient_id"]

    def get_basal_rate(self):
        return self.basal_rate

    def get_carb_insulin_ratio(self):
        return self.cir

    def get_insulin_sensitivity_factor(self):
        return self.isf

    def get_age(self):
        return self.age

    def get_patient_id(self):
        return self.patient_id


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

    basal_jitter = 0.1
    basal_rates = [random_state.uniform(basal_rate - basal_jitter, basal_rate + basal_jitter) for _ in hours_in_day]

    cir_jitter = 1
    carb_ratios = [random_state.uniform(cir - cir_jitter, cir + cir_jitter) for _ in hours_in_day]

    isf_jitter = 1
    isfs = [random_state.uniform(isf - isf_jitter, isf + isf_jitter) for _ in hours_in_day]

    start_times_hourly = [datetime.time(hour=i, minute=0, second=0) for i in hours_in_day]
    durations_min_per_hour = [SINGLE_SETTING_DURATION / total_hours_in_day for _ in hours_in_day]

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

    patient_config.recommendation_accept_prob = 0.0  # Does not accept any bolus recommendations
    patient_config.min_bolus_rec_threshold = 0.5  # Minimum size of bolus to accept
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
        patient_obj = iCGMPatient(settings)
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



if __name__ == "__main__":

    export_icgm_scenario_metadata_from_scenario_files()
