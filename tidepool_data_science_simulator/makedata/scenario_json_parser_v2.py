__author__ = "Cameron Summers"

import pandas as pd
import numpy as np
import copy
import json
import os
import datetime

import logging
logger = logging.getLogger(__name__)

from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController
from tidepool_data_science_simulator.models.controller import OpenLoopController

from tidepool_data_science_simulator.legacy.read_fda_risk_input_scenarios_ORIG import input_table_to_dict
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, TempBasalTimeline, ActionTimeline, PhysicalActivityTimeline, VirtualPatientDeleteLoopData
from tidepool_data_science_simulator.models.measures import (
    Carb,
    Bolus,
    BasalRate,
    CarbInsulinRatio,
    InsulinSensitivityFactor,
    GlucoseSensitivityFactor,
    BasalBloodGlucose,
    InsulinProductionRate,
    TargetRange,
    GlucoseTrace,
    PhysicalActivity,
)

from tidepool_data_science_simulator.makedata.scenario_parser import (
    SimulationParser, SensorConfig, PatientConfig, ControllerConfig, PumpConfig
)
from tidepool_data_science_simulator.makedata.make_patient import get_heartrate_trace
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.controller import AutomationControlTimeline, LoopController, DoNothingController, AutomationControl

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

POINTER_OBJ_DIR = os.path.dirname(__file__) + "/../../scenario_configs/tidepool_risk_v2/"
DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

SWIFT_CONTROLLER_MODEL_NAME_MAP = {
    "rapid_acting_adult": "novolog",
    # "rapid_acting_child": [360, 65],
    # "walsh": [120, 15],
    # "fiasp": [360, 55],
    # "theoretical_fast_5": [20, 120],
    # "theoretical_fast_3":[20, 240],
    # "theoretical_fast_1": [29, 300],
    # "theoretical_fast_4": [20, 240],
    # "theoretical_fast_2": [29, 300],
    # "u500": [360, 1110],
    # "regular": [360, 420],
    # "nph": [480, 1320],
    # "degludec": [540, 1440],
    # "glargine": [540, 1440]
}

PYLOOPKIT_CONTROLLER_MODEL_NAME_MAP = {
    "novolog": [360, 75],
    "rapid_acting_adult": [360, 75],
    "rapid_acting_child": [360, 65],
    "walsh": [120, 15],
    "fiasp": [360, 55],
    "theoretical_fast_5": [20, 120],
    "theoretical_fast_3":[20, 240],
    "theoretical_fast_1": [29, 300],
    "theoretical_fast_4": [20, 240],
    "theoretical_fast_2": [29, 300],
    "u500": [360, 1110],
    "regular": [360, 420],
    "nph": [480, 1320],
    "degludec": [540, 1440],
    "glargine": [540, 1440]
}

class ScenarioParserV2(SimulationParser):
    """
    Redesigned scenario parser for Tidepool Risk automated pipeline, Feb 2021.
    """

    def __init__(self, path_to_json_config=None, pointer_object_dir=POINTER_OBJ_DIR):

        self.pointer_keyword = "reusable"
        self.pointer_object_dir = pointer_object_dir
        self.override_details = []

        if path_to_json_config:
            config = json.load(open(path_to_json_config))
            self.metadata = self.get_required_value(config, "metadata", dict)
            self.base_sim_config = self.get_required_value(config, "base_config", dict)

            if self.is_config_file_pointer(self.base_sim_config):  # Resolve top level pointer
                self.base_sim_config = self.load_pointer(self.base_sim_config)

            self.override_configs = self.get_required_value(config, "override_config", list)

        self.pump_model = dict()
        self.patient_model = dict()

    def get_required_value(self, obj, key, type_=None):
        """
        Function to enforce required values in scenario configuration.
        """

        value = obj.get(key, None)
        if value is None:
            raise ValueError("Required data {} not in config.".format(key))

        if type is not None and not isinstance(value, type_) and not self.is_config_file_pointer(value):
            raise TypeError("Expected value {} to be {}, but got {}".format(value, type(value), type_))

        return value

    def get_sims(self, override_json_save_dir=None):
        """
        Get simulation objects as specified by the config file.
        """
        simulations = dict()

        for i, override_delta in enumerate(self.override_configs, 1):

            override_sim_config = copy.deepcopy(self.base_sim_config)
            self.apply_config_override(override_sim_config, override_delta)

            sim_id = override_sim_config["sim_id"]

            if override_json_save_dir is not None:
                filepath = os.path.join(override_json_save_dir, sim_id + "_override_config.json")
                json.dump(override_sim_config, open(filepath, "w"), indent=4)

            sim = self.build_sim_from_config(override_sim_config)
            sim.name = sim_id
            simulations[sim_id] = sim

            logger.info(f"Created simulation: {sim_id}")

        return simulations

    def apply_config_override(self, base_sim_config, override_delta):
        """
        Modifies the config object in place and:
            1. Resolves pointer references to other files in the config
            2. Resolves the overriding leaf note configs
        """
        # Clear previous override details
        self.override_details = []

        self.resolve_pointers(base_sim_config)
        self.resolve_pointers(override_delta)

        num_overrides = self.count_leaf_nodes(override_delta)
        
        # Diagnose what will be applied
        print("\n" + "="*80)
        print(f"DIAGNOSING OVERRIDES")
        print("="*80)
        applied_paths, failed_paths = self.diagnose_override_application(
            base_sim_config, override_delta
        )
        
        print(f"\n📊 OVERRIDE SUMMARY:")
        print(f"   Total override values: {num_overrides}")
        print(f"   Expected to apply: {len(applied_paths)}")
        print(f"   Expected to fail: {len(failed_paths)}")
        
        if failed_paths:
            print(f"\n🔴 FAILED OVERRIDES ({len(failed_paths)}):")
            for path in sorted(failed_paths):
                print(f"   - {path}")
        
        if applied_paths:
            print(f"\n🟢 SUCCESSFUL OVERRIDES ({len(applied_paths)}):")
            for path in sorted(applied_paths):
                print(f"   - {path}")
        
        print("="*80 + "\n")
        
        # Now actually apply the overrides
        num_overrides_applied = self.resolve_override(base_sim_config, override_delta)

        if num_overrides_applied != num_overrides:
            raise Exception("Only applied {} of {} overriding values in {}. Check configurations.".format(
                num_overrides_applied, num_overrides, override_delta))



    def diagnose_override_application(self, base_dict, override_dict, path="", applied_paths=None, failed_paths=None):
        """
        Recursively diagnose which override paths succeed or fail.
        Returns sets of applied and failed paths.
        """
        if applied_paths is None:
            applied_paths = set()
        if failed_paths is None:
            failed_paths = set()
        
        # Handle lists - they replace entirely
        if isinstance(override_dict, list):
            current_path = path if path else "<root>"
            if isinstance(base_dict, list) or not isinstance(base_dict, dict):
                applied_paths.add(f"{current_path} (list replacement)")
            else:
                failed_paths.add(f"{current_path} (TYPE_MISMATCH: base is not a list)")
            return applied_paths, failed_paths
        
        # Handle non-dict overrides
        if not isinstance(override_dict, dict):
            return applied_paths, failed_paths
        
        for key, value in override_dict.items():
            current_path = f"{path}.{key}" if path else key
            
            if key not in base_dict:
                failed_paths.add(f"{current_path} (KEY_NOT_FOUND)")
                print(f"❌ Override FAILED: '{current_path}' - key doesn't exist in base config")
                continue
            
            base_value = base_dict[key]
            
            # If override is a dict and base is also a dict, recurse
            if isinstance(value, dict) and isinstance(base_value, dict):
                self.diagnose_override_application(base_value, value, current_path, applied_paths, failed_paths)
            # If override is a list or not a dict, it's a leaf replacement
            else:
                # Check type compatibility
                if type(value) != type(base_value) and base_value is not None:
                    failed_paths.add(f"{current_path} (TYPE_MISMATCH: expected {type(base_value).__name__}, got {type(value).__name__})")
                    print(f"❌ Override FAILED: '{current_path}' - type mismatch")
                    print(f"   Expected: {type(base_value).__name__} = {base_value}")
                    print(f"   Got: {type(value).__name__} = {value}")
                else:
                    applied_paths.add(current_path)
                    print(f"✓ Override will apply: '{current_path}' = {value}")
        
        return applied_paths, failed_paths
    
    def count_leaf_nodes(self, obj):
        """
        Count the number of non-dict values in the object. Used to later
        validate that all overrides have been applied.
        
        Lists are treated as atomic values (single leaf nodes) because they
        represent complete replacement values in overrides, not structures
        to traverse for individual leaf overrides.

        Parameters
        ----------
        obj

        Returns
        -------
        int: Number of non-dict values
        """
        # Lists are treated as single leaf nodes (atomic values)
        if isinstance(obj, list):
            return 1
        
        # Non-dict, non-list values are leaf nodes
        if not isinstance(obj, dict):
            return 1
            
        # For dicts, recursively count leaf nodes
        num_leaf_nodes = 0
        for k, v in obj.items():
            if isinstance(v, dict):
                num_leaf_nodes += self.count_leaf_nodes(v)
            else:
                # Both lists and primitive values count as 1 leaf node
                num_leaf_nodes += 1
        return num_leaf_nodes

    def resolve_pointers(self, value, key_prefix=""):
        """
        Recursively traverse the simulation config obj and replace any pointers with their objects.

        """
        # Handle lists
        if isinstance(value, list):
            for i, item in enumerate(value):
                if self.is_config_file_pointer(item):
                    value[i] = self.load_pointer(item)
                elif isinstance(item, dict):
                    self.resolve_pointers(item, f"{key_prefix}[{i}]")
                elif isinstance(item, list):
                    self.resolve_pointers(item, f"{key_prefix}[{i}]")
            return
        
        # Handle dicts (original logic)
        if not isinstance(value, dict):
            return
            
        for k, v in value.items():
            current_key = f"{key_prefix}.{k}" if key_prefix else k
            
            if self.is_config_file_pointer(v):
                loaded_value = self.load_pointer(v)
                
                # Special handling for physical_activity_entries pointing to profiles
                # If we're loading a PA profile (which has both metadata and entries),
                # and the key is 'physical_activity_entries', extract just the entries list
                if k == "physical_activity_entries" and isinstance(loaded_value, dict):
                    if "physical_activity_entries" in loaded_value:
                        # This is a profile with metadata - extract just the entries
                        value[k] = loaded_value["physical_activity_entries"]
                        logger.info(f"Extracted PA entries from profile for key '{current_key}'")
                    else:
                        # Regular pointer resolution
                        value[k] = loaded_value
                else:
                    value[k] = loaded_value
            elif isinstance(v, dict):
                self.resolve_pointers(v, current_key)
            elif isinstance(v, list):
                self.resolve_pointers(v, current_key)

    def resolve_override(self, obj, override_obj, key_prefix=""):
        """
        Recursively traverse the simulation config obj and apply specified leaf overrides.
        """
        # Handle the case where obj is a list (shouldn't happen in normal override flow,
        # but adding for robustness)
        if isinstance(obj, list) or isinstance(override_obj, list):
            # Lists are treated as leaf nodes that get replaced entirely
            return 0
        
        # Both obj and override_obj should be dicts at this point
        if not isinstance(obj, dict) or not isinstance(override_obj, dict):
            return 0
            
        num_overides_applied = 0

        for k, v in obj.items():
            current_key = f"{key_prefix}.{k}" if key_prefix else k

            if k in override_obj:
                override_value = override_obj[k]
                
                # If override value is not a dict (or is a list), treat it as a leaf node replacement
                if not isinstance(override_value, dict) or isinstance(override_value, list):
                    old_value = obj[k]
                    new_value = override_value
                    obj[k] = new_value

                    # Record the override details at instance level
                    self.override_details.append({
                        "key": current_key,
                        "old_value": old_value,
                        "new_value": new_value,
                        "value_type": type(new_value).__name__
                    })

                    num_overides_applied += 1
                else:  # key is there and it's a dict that should be explored for leaf overrides
                    # Only recurse if the base value is also a dict
                    if isinstance(v, dict):
                        sub_overrides = self.resolve_override(v, override_value, current_key)
                        num_overides_applied += sub_overrides
                    else:
                        # Base value is not a dict but override is, replace entirely
                        obj[k] = override_value
                        num_overides_applied += 1

        return num_overides_applied

    def is_config_file_pointer(self, value):
        """
        Return True if the value matches the pattern for designating a file for a config.
        """
        return isinstance(value, str) and self.pointer_keyword in value

    def load_pointer(self, pointer_string):
        """
        Load file object pointed to. Searches in subdirectories if not found in main directory.
        """
        pointer_segments = pointer_string.split(".")
        folder_path = os.path.join("/".join(pointer_segments[:-1]))
        filename_no_ext = pointer_segments[-1]
        json_filename = "{}.json".format(filename_no_ext)
        csv_filename = "{}.csv".format(filename_no_ext)

        # Define subdirectory search paths based on folder type
        subdirectories = []
        if "simulations" in folder_path:
            subdirectories = ["base", "suspend", "loop_versions", "specialized"]
        elif "metabolism_settings" in folder_path:
            subdirectories = ["profiles", "suspensions", "presets", "versions", "types"]
        
        # Build search paths: original location first, then subdirectories
        search_paths = [folder_path]
        for subdir in subdirectories:
            search_paths.append(os.path.join(folder_path, subdir))

        # Search for JSON files
        for search_path in search_paths:
            json_path = os.path.join(self.pointer_object_dir, search_path, json_filename)
            if os.path.isfile(json_path):
                obj = json.load(open(json_path, "r"))
                return obj

        # Search for CSV files
        for search_path in search_paths:
            csv_path = os.path.join(self.pointer_object_dir, search_path, csv_filename)
            if os.path.isfile(csv_path):
                obj = pd.read_csv(csv_path).to_dict()
                return obj

        # If not found anywhere, raise exception with detailed search info
        searched_paths = [os.path.join(self.pointer_object_dir, path, filename_no_ext) for path in search_paths]
        raise Exception("Could not load pointer file {}. Searched in: {}".format(
            filename_no_ext, ", ".join(searched_paths)))

        return obj

    def times_to_minutes(self, time_before, time_after):
        return int((time_after - time_before).total_seconds() / 60)

    def time_string_to_time(self, time_str):
        return datetime.datetime.strptime(time_str, '%H:%M:%S').time()

    def parse_start_times(self, start_times_str):
        """
        Take list of times in string format and return datetime.time objects and minute durations. This is
        the expected format for Pyloopkip setting schedules.
        """

        first_time = self.time_string_to_time(start_times_str[0])
        if first_time != datetime.time(0, 0, 0):
            raise Exception("First time {} for setting schedule is not 00:00:00".format(first_time))

        if len(start_times_str) > 1:
            start_times = [first_time]
            durations_minutes = []
            prev_time = first_time
            for start_time_str in start_times_str[1:]:
                time_obj = self.time_string_to_time(start_time_str)

                if time_obj < prev_time:
                    raise Exception("Setting schedule times out of order: {} and {}".format(time_obj, prev_time))

                start_times.append(time_obj)
                prev_dt = datetime.datetime.combine(datetime.datetime.today(), prev_time)
                time_dt = datetime.datetime.combine(datetime.datetime.today(), time_obj)
                duration_minutes = self.times_to_minutes(prev_dt, time_dt)
                durations_minutes.append(duration_minutes)

                prev_time = time_obj

            first_dt = datetime.datetime.combine(datetime.datetime.today() + datetime.timedelta(days=1), first_time)
            durations_minutes.append(self.times_to_minutes(time_dt, first_dt))

        else:
            start_times = [first_time]
            durations_minutes = [1440*10]  # minutes in 24 hours * 10 

        return start_times, durations_minutes

    def get_scalar_setting_schedule_info(self, schedule_config, validation_func):
        """
        Get necessary info for creating setting schedule objects.
        """
        start_times, durations_minutes = self.parse_start_times(schedule_config.get("start_times"))
        values = schedule_config.get("values")

        if not (len(start_times) == len(durations_minutes) == len(values)):
            raise ValueError("Setting schedule does not have matching values.")

        [validation_func(value) for value in values]

        return start_times, durations_minutes, values

    def get_range_setting_schedule_info(self, schedule_config, validation_func):
        start_times, durations_minutes = self.parse_start_times(schedule_config.get("start_times"))
        upper_values = schedule_config.get("upper_values")
        lower_values = schedule_config.get("lower_values")

        if not (len(lower_values) == len(upper_values) == len(start_times) == len(durations_minutes)):
            raise ValueError("Different number of values passed in")

        [validation_func(lower_val, upper_val) for lower_val, upper_val in zip(lower_values, upper_values)]

        return start_times, durations_minutes, lower_values, upper_values

    def validate_basal_rate(self, basal_rate):
        """
        Validate a basal rate in the config.
        """
        if not isinstance(basal_rate, float):
            raise ValueError("Value type should be float")

        if not 0 <= basal_rate <= 100:
            raise ValueError("Value {} exceeds expected range, likely an error.".format(basal_rate))

    def validate_carb_ratio(self, carb_ratio):

        float(carb_ratio)

        if not 0 < carb_ratio <= 231:
            raise ValueError("Value {} exceeds expected range, likely an error.".format(carb_ratio))

    def validate_insulin_sensitivity(self, insulin_sensitivity):

        float(insulin_sensitivity)

        if not 0 < insulin_sensitivity <= 1200:
            raise ValueError("Value {} exceeds expected range, likely an error.".format(insulin_sensitivity))

    def validate_target_range(self, lower_val, upper_val):

        float(lower_val)
        float(upper_val)

        if lower_val > upper_val:
            raise ValueError("Expected lower val {} to be greater than upper val {}".format(lower_val, upper_val))

        if lower_val < 0 or upper_val < 0:
            raise ValueError("Target range values must be greater than zero.")

    def validate_carb_entry(self):
        pass

    def validate_glucose_sensitivity_factor(self, glucose_sensitivity_factor):
        """
        Validate a basal secretion rate in the config.
        """
        if not isinstance(glucose_sensitivity_factor, float) and \
            not isinstance(glucose_sensitivity_factor, int):
            raise ValueError("Value type should be float or int")

        if not 0 <= glucose_sensitivity_factor <= 500:
            raise ValueError("Value {} exceeds expected range, likely an error.".format(glucose_sensitivity_factor))
        
    def validate_basal_blood_glucose(self, basal_blood_glucose):
        """
        Validate a basal secretion rate in the config.
        """
        if not isinstance(basal_blood_glucose, float):
            raise ValueError("Value type should be float")

        if not 0 <= basal_blood_glucose <= 500:
            raise ValueError("Value {} exceeds expected range, likely an error.".format(basal_blood_glucose))

    def validate_insulin_production_rate(self, insulin_production_rate):
        """
        Validate a basal secretion rate in the config.
        """
        if not isinstance(insulin_production_rate, float):
            raise ValueError("Value type should be float")

        if not 0 <= insulin_production_rate <= 5:
            raise ValueError("Value {} exceeds expected range, likely an error.".format(insulin_production_rate))
            
    def carb_entries_to_timeline(self, carb_entries):

        carb_datetimes = []
        carb_events = []
        for carb_entry in carb_entries:
            carb_datetime = datetime.datetime.strptime(carb_entry["start_time"], DATETIME_FORMAT)
            carb_value = carb_entry["value"]
            carb_duration = carb_entry.get("duration", 180)
            carb_obj = Carb(carb_value, "g", carb_duration)

            carb_datetimes.append(carb_datetime)
            carb_events.append(carb_obj)

        return CarbTimeline(carb_datetimes, carb_events)

    def bolus_entries_to_timeline(self, bolus_entries):

        insulin_datetimes = []
        insulin_events = []
        for insulin_entry in bolus_entries:
            insulin_dt = datetime.datetime.strptime(insulin_entry["time"], DATETIME_FORMAT)
            insulin_value = insulin_entry["value"]
            bolus = Bolus(insulin_value, "U")

            insulin_datetimes.append(insulin_dt)
            insulin_events.append(bolus)

        return BolusTimeline(insulin_datetimes, insulin_events)

    def automation_control_entries_to_timeline(self, automation_control_entries):

        automation_control_datetimes = []
        automation_control_events = []
        for entry in automation_control_entries:
            entry_time = datetime.datetime.strptime(entry["time"], DATETIME_FORMAT)
            dosing_enabled = entry["dosing_enabled"]
            automation_control_obj = AutomationControl(dosing_enabled, entry_time)

            automation_control_datetimes.append(entry_time)
            automation_control_events.append(automation_control_obj)

        return AutomationControlTimeline(automation_control_datetimes, automation_control_events)

    def build_components_from_config(self, sim_config, sensor=None, pump=None):

        sim_start_time_str = self.get_required_value(sim_config, "time_to_calculate_at", str)
        sim_start_time = datetime.datetime.strptime(sim_start_time_str, DATETIME_FORMAT)

        duration_hrs = self.get_required_value(sim_config, "duration_hours", float)
        # Store timing variables for use in get_patient_config (Phase 2)
        self.sim_start_time = sim_start_time
        self.duration_hrs = duration_hrs


        self.pump_model = self.build_model_from_config(sim_start_time, sim_config["patient"]["pump"])
        self.patient_model = self.build_model_from_config(sim_start_time, sim_config["patient"]["patient_model"])

        self.sensor_glucose_history = self.build_glucose_history(sim_config["patient"]["sensor"]["glucose_history"])
        self.patient_model_glucose_history = self.build_glucose_history(
            sim_config["patient"]["patient_model"]["glucose_history"])

        controller = self.get_controller(sim_start_time, sim_config)

        if pump is None:
            pump = ContinuousInsulinPump(time=sim_start_time, pump_config=self.get_pump_config())

        if sensor is None:
            sensor = IdealSensor(time=sim_start_time, sensor_config=self.get_sensor_config())

        # TODO: The JSON parser is not flexible enough to accomodate different VirtualPatient models
        virtual_patient = VirtualPatient(
            sim_start_time,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=self.get_patient_config(),
        )

        return sim_start_time, duration_hrs, virtual_patient, controller

    def build_sim_from_config(self, sim_config):

        sim_start_time, duration_hrs, virtual_patient, controller = self.build_components_from_config(sim_config)

        sim = Simulation(sim_start_time,
                         duration_hrs=duration_hrs,
                         virtual_patient=virtual_patient,
                         controller=controller,
                         multiprocess=True,
                         sim_id=self.metadata["simulation_id"]
                         )
        return sim
    
    def build_model_from_config(self, sim_start_time, model_config):

        model = dict()
        
        metabolism_settings = model_config["metabolism_settings"]

        # model patient's actual insulin type
        patient_insulin_type = metabolism_settings.get("patient_insulin_type", "rapid_acting_adult")
        model["patient_insulin_type"] = patient_insulin_type

        basal_rate_schedule = metabolism_settings["basal_rate"]
        basal_start_times, basal_durations_minutes, basal_values = self.get_scalar_setting_schedule_info(basal_rate_schedule, self.validate_basal_rate)
                
        model["basal_rate_schedule"] = BasalSchedule24hr(
            sim_start_time,
            start_times=basal_start_times,
            values=[
                BasalRate(rate, units)
                for rate, units in zip(
                    basal_values,
                    ["U/hr"] * len(basal_values),  # NOTE: Assuming these units for now to reduce on config verbosity
                )
            ],
            duration_minutes=basal_durations_minutes,
        )

        carb_ratio_schedule = metabolism_settings["carb_insulin_ratio"]
        carb_ratio_start_times, carb_ratio_durations_minutes, carb_ratio_values = self.get_scalar_setting_schedule_info(carb_ratio_schedule, self.validate_carb_ratio)

        model["carb_ratio_schedule"] = SettingSchedule24Hr(
            sim_start_time,
            "Carb Insulin Ratio",
            start_times=carb_ratio_start_times,
            values=[
                CarbInsulinRatio(value, units)
                for value, units in zip(
                    carb_ratio_values,
                    ["g/U"] * len(carb_ratio_values),
                )
            ],
            duration_minutes=carb_ratio_durations_minutes
        )

        # Insulin sensitivity schedule
        insulin_sensitivity_schedule = metabolism_settings["insulin_sensitivity_factor"]
        insulin_sensitivity_start_times, insulin_sensitivity_durations_minutes, insulin_sensitivity_values = \
            self.get_scalar_setting_schedule_info(insulin_sensitivity_schedule, self.validate_insulin_sensitivity)
        
        model["insulin_sensitivity_schedule"] = SettingSchedule24Hr(
            sim_start_time,
            "Insulin Sensitivity",
            start_times=insulin_sensitivity_start_times,
            values=[
                InsulinSensitivityFactor(value, units)
                for value, units in zip(insulin_sensitivity_values, ["mg/dL / U"] * len(insulin_sensitivity_values))
            ],
            duration_minutes=insulin_sensitivity_durations_minutes
        )

        # Type 2 insulin model
        # Sets to default Type 1 if these parameters are not specified in the config file to maintain backward compatibility with 
        # older config files.
        if "glucose_sensitivity_factor" in metabolism_settings:
            glucose_sensitivity_factor_schedule = metabolism_settings["glucose_sensitivity_factor"]
            glucose_sensitivity_factor_start_times, glucose_sensitivity_factor_duration_minutes, glucose_sensitivity_factor_values = \
                self.get_scalar_setting_schedule_info(glucose_sensitivity_factor_schedule, self.validate_glucose_sensitivity_factor)
            
        else:
            glucose_sensitivity_factor_start_times = [datetime.time(0, 0)]
            glucose_sensitivity_factor_duration_minutes = [1440]
            glucose_sensitivity_factor_values = [0.0]
            
        model["glucose_sensitivity_factor_schedule"] = SettingSchedule24Hr(
            sim_start_time,
            "Glucose Sensitivity Factor",
            start_times=glucose_sensitivity_factor_start_times,
            values=[
                GlucoseSensitivityFactor(value, units)
                for value, units in zip(glucose_sensitivity_factor_values, ["U / mg/dL"] * len(glucose_sensitivity_factor_values))
            ],
            duration_minutes=glucose_sensitivity_factor_duration_minutes
        )

        if "basal_blood_glucose" in metabolism_settings:
            basal_blood_glucose_schedule = metabolism_settings["basal_blood_glucose"]
            basal_blood_glucose_start_times, basal_blood_glucose_duration_minutes, basal_blood_glucose_values = \
                self.get_scalar_setting_schedule_info(basal_blood_glucose_schedule, self.validate_basal_blood_glucose)
            
        else:
            basal_blood_glucose_start_times = [datetime.time(0, 0)]
            basal_blood_glucose_duration_minutes = [1440]
            basal_blood_glucose_values = [100.0]

        model["basal_blood_glucose_schedule"] = SettingSchedule24Hr(
            sim_start_time,
            "Basal Blood Glucose",
            start_times=basal_blood_glucose_start_times,
            values=[
                BasalBloodGlucose(value, units)
                for value, units in zip(basal_blood_glucose_values, ["mg/dL"] * len(basal_blood_glucose_values))
            ],
            duration_minutes=basal_blood_glucose_duration_minutes
        )

        if "insulin_production_rate" in metabolism_settings:
            insulin_production_rate_schedule = metabolism_settings["insulin_production_rate"]
            insulin_production_rate_start_times, insulin_production_rate_duration_minutes, insulin_production_rate_values = \
                self.get_scalar_setting_schedule_info(insulin_production_rate_schedule, self.validate_insulin_production_rate)
            
        else:
            insulin_production_rate_start_times = [datetime.time(0, 0)]
            insulin_production_rate_duration_minutes = [1440]
            insulin_production_rate_values = [0.0]

        model["insulin_production_rate_schedule"] = SettingSchedule24Hr(
            sim_start_time,
            "Insulin Production Rate",
            start_times=insulin_production_rate_start_times,
            values=[
                InsulinProductionRate(value, units)
                for value, units in zip(insulin_production_rate_values, ["U/min"] * len(insulin_production_rate_values))
            ],
            duration_minutes=insulin_production_rate_duration_minutes
        )
        
        # Physical activity processing - ONLY for patient model
        pa_entries = model_config.get("physical_activity_entries", [])
        
        # If pa_entries is a string, it's a reusable profile reference - wrap it in a list
        if isinstance(pa_entries, str) and pa_entries.startswith("reusable."):
            pa_entries = [pa_entries]  # Wrap in list for processing
        
        # Extract metabolism parameters from PA profiles if available
        pa_metabolism_params = self.extract_metabolism_params_from_pa_profiles(pa_entries)
        
        # Physical activity and metabolism model parameters with defaults
        # Priority: explicit model_config > explicit metabolism_settings > profile defaults > global defaults
        # IMPORTANT: Check both metabolism_settings AND model_config top level for PA parameters
        # This ensures PA parameters are preserved when metabolism_settings is overridden
        model["w_hr"] = model_config.get("w_hr", 
                          metabolism_settings.get("w_hr", 
                              pa_metabolism_params.get("w_hr", 0.0)))
        model["a"] = model_config.get("a",
                       metabolism_settings.get("a",
                           pa_metabolism_params.get("a", 1.0)))
        model["tau"] = model_config.get("tau",
                         metabolism_settings.get("tau",
                             pa_metabolism_params.get("tau", 60.0)))
        model["n"] = model_config.get("n",
                       metabolism_settings.get("n",
                           pa_metabolism_params.get("n", 1.0)))

        # Specific to pump
        if "target_range" in model_config:
            target_range_schedule = model_config["target_range"]
            target_range_start_times, target_range_durations_minutes, target_range_lower_values, target_range_upper_values = \
                self.get_range_setting_schedule_info(target_range_schedule, self.validate_target_range)

            model["target_range_schedule"] = TargetRangeSchedule24hr(
                sim_start_time,
                start_times=target_range_start_times,
                values=[
                    TargetRange(min_value, max_value, units)
                    for min_value, max_value, units in zip(
                        target_range_lower_values,
                        target_range_upper_values,
                        ["mg/dL"] * len(target_range_lower_values),
                    )
                ],
                duration_minutes=target_range_durations_minutes
            )

        
        carb_entries = model_config["carb_entries"]
        model["carb_timeline"] = self.carb_entries_to_timeline(carb_entries)

        bolus_entries = model_config["bolus_entries"]
        model["bolus_timeline"] = self.bolus_entries_to_timeline(bolus_entries)

        # Physical activity timeline (pa_entries already extracted above for parameter extraction)
        model["pa_timeline"] = self.enhanced_physical_activity_entries_to_timeline(pa_entries)

        model["action_timeline"] = ActionTimeline()

        return model

    def build_glucose_history(self, history_obj):

        glucose_trace_obj = GlucoseTrace(
            datetimes=[datetime.datetime.strptime(value, DATETIME_FORMAT) for value in history_obj["datetime"].values()],
            values=list(history_obj["value"].values()),
        )

        return glucose_trace_obj

    def get_controller(self, sim_start_time, sim_config):

        controller = DoNothingController(sim_start_time, controller_config=None)

        if sim_config.get("controller") is not None:
            controller_settings = sim_config["controller"]["settings"]

            automation_control_entries = []
            if "automation_control_timeline" in sim_config["controller"].keys():
                automation_control_entries = sim_config["controller"]["automation_control_timeline"]

            automation_control_timeline = self.automation_control_entries_to_timeline(automation_control_entries)

            controller_config = ControllerConfig(
                bolus_event_timeline=self.pump_model["bolus_timeline"],
                carb_event_timeline=self.pump_model["carb_timeline"],
                controller_settings=controller_settings
            )
            
            model_name = controller_settings["model"]
            controller_id = sim_config['controller']['id']
            
            if 'swift' in controller_id:
                if model_name in SWIFT_CONTROLLER_MODEL_NAME_MAP:
                    model_name = SWIFT_CONTROLLER_MODEL_NAME_MAP[model_name]
                controller_config.controller_settings['model'] = model_name
                controller = SwiftLoopController(sim_start_time, controller_config, automation_control_timeline)

            elif 'py' in controller_id:
                if model_name in PYLOOPKIT_CONTROLLER_MODEL_NAME_MAP:
                    model_name = PYLOOPKIT_CONTROLLER_MODEL_NAME_MAP[model_name]
                controller_config.controller_settings['model'] = model_name
                controller = LoopController(sim_start_time, controller_config, automation_control_timeline)

            elif 'open' in controller_id:
                controller = OpenLoopController(sim_start_time, controller_config, automation_control_timeline)

        return controller


    # ===== PHASE 2 & 3: PHYSICAL ACTIVITY SUPPORT WITH VALIDATION =====
    
    def validate_metabolism_parameters(self, params):
        """
        Validate PA metabolism parameters.
        
        Parameters
        ----------
        params : dict
            Dictionary of metabolism parameters to validate
            
        Returns
        -------
        list
            List of validation errors (empty if valid)
        """
        errors = []
        
        if 'w_hr' in params:
            try:
                w_hr = float(params['w_hr'])
                if not -10.0 <= w_hr <= 10.0:
                    errors.append(f"w_hr {w_hr} outside expected range [-10, 10]")
            except (ValueError, TypeError):
                errors.append(f"w_hr must be numeric, got: {params['w_hr']}")
        
        if 'a' in params:
            try:
                a = float(params['a'])
                if not -1.0 <= a <= 1.0:
                    errors.append(f"a {a} outside expected range [-1, 1]")
            except (ValueError, TypeError):
                errors.append(f"a must be numeric, got: {params['a']}")
        
        if 'tau' in params:
            try:
                tau = float(params['tau'])
                if not 0.0 < tau <= 1000.0:
                    errors.append(f"tau {tau} outside expected range (0, 1000]")
            except (ValueError, TypeError):
                errors.append(f"tau must be numeric, got: {params['tau']}")
        
        if 'n' in params:
            try:
                n = float(params['n'])
                if not 0.0 < n <= 100.0:
                    errors.append(f"n {n} outside expected range (0, 100]")
            except (ValueError, TypeError):
                errors.append(f"n must be numeric, got: {params['n']}")
        
        return errors
    
    def extract_metabolism_params_from_pa_profiles(self, pa_entries):
        """
        Extract metabolism parameters from PA profile configurations.
        
        Parameters
        ----------
        pa_entries : list
            List of PA entries (may include reusable profile references)
            
        Returns
        -------
        dict
            Dictionary of metabolism parameters found in profiles (may be empty)
        """
        pa_metabolism_params = {}
        
        if not pa_entries:
            return pa_metabolism_params
        
        # Check each entry for profile references with metabolism parameters
        for entry in pa_entries:
            try:
                # Check if this is a profile reference string
                if isinstance(entry, str) and entry.startswith('reusable.'):
                    profile_config = self.load_reusable_pa_config(entry)
                    
                    # Extract metabolism_parameters if present
                    if 'metabolism_parameters' in profile_config:
                        profile_params = profile_config['metabolism_parameters']
                        
                        # Validate parameters
                        validation_errors = self.validate_metabolism_parameters(profile_params)
                        if validation_errors:
                            error_msg = f"Metabolism parameter validation errors in {entry}:\n" + "\n".join(validation_errors)
                            logger.error(error_msg)
                            raise ValueError(error_msg)
                        
                        # Use parameters from first profile with metabolism_parameters
                        # (Priority is given to the first profile encountered)
                        if not pa_metabolism_params:
                            pa_metabolism_params = profile_params.copy()
                            logger.info(f"Extracted metabolism parameters from PA profile: {entry}")
                            break
                        
            except ValueError:
                # Re-raise validation errors
                raise
            except Exception as e:
                # Log but don't fail for other errors - just skip this entry
                logger.warning(f"Could not extract metabolism parameters from PA entry: {e}")
                continue
        
        return pa_metabolism_params
    
    def physical_activity_entries_to_timeline(self, pa_entries):
        """
        Convert physical activity entries from JSON to PhysicalActivityTimeline
        
        Parameters
        ----------
        pa_entries: list
            List of physical activity entry dictionaries
            
        Returns
        -------
        PhysicalActivityTimeline
        """
        pa_datetimes = []
        pa_events = []
        
        for pa_entry in pa_entries:
            pa_datetime = datetime.datetime.strptime(pa_entry["start_time"], DATETIME_FORMAT)
            activity_name = pa_entry.get("activity", "exercise")
            duration_minutes = pa_entry.get("duration", 30)
            
            pa_obj = PhysicalActivity(activity=activity_name, duration=duration_minutes)
            
            pa_datetimes.append(pa_datetime)
            pa_events.append(pa_obj)
        
        return PhysicalActivityTimeline(pa_datetimes, pa_events)
    
    def load_reusable_pa_config(self, config_path):
        """
        Load a reusable physical activity configuration file.
        
        Parameters
        ----------
        config_path : str
            Path to the reusable configuration in dot notation
            (e.g., "reusable.physical_activities.profiles.moderate_exercise_v1")
            
        Returns
        -------
        dict
            The loaded configuration
        """
        # Convert dot notation to file path
        path_parts = config_path.split('.')
        if path_parts[0] != 'reusable':
            raise ValueError(f"Reusable config path must start with 'reusable', got: {config_path}")
        
        # Build file path - keep 'reusable' in path and add .json extension
        # pointer_object_dir already points to scenario_configs/tidepool_risk_v2/
        # so we need to include 'reusable' in the relative path
        relative_path = '/'.join(path_parts) + '.json'
        file_path = os.path.join(self.pointer_object_dir, relative_path)
        
        try:
            with open(file_path, 'r') as f:
                config = json.load(f)
            
            logger.info(f"Loaded reusable PA config: {config_path} from {file_path}")
            return config
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Reusable PA config not found: {file_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in reusable PA config {file_path}: {e}")
    
    def validate_pa_entry(self, pa_entry, entry_index=None):
        """
        Validate a physical activity entry.
        
        Parameters
        ----------
        pa_entry : dict
            Physical activity entry to validate
        entry_index : int, optional
            Index of entry for error reporting
            
        Returns
        -------
        list
            List of validation errors (empty if valid)
        """
        errors = []
        entry_desc = f"PA entry {entry_index}" if entry_index is not None else "PA entry"
        
        # Required fields
        required_fields = ['start_time']
        for field in required_fields:
            if field not in pa_entry:
                errors.append(f"{entry_desc}: Missing required field '{field}'")
        
        # Must have either 'activity' or 'activity_ref'
        if 'activity' not in pa_entry and 'activity_ref' not in pa_entry:
            errors.append(f"{entry_desc}: Must have either 'activity' or 'activity_ref'")
        
        # Validate activity reference format
        if 'activity_ref' in pa_entry:
            ref = pa_entry['activity_ref']
            if not isinstance(ref, str) or not ref.startswith('reusable.'):
                errors.append(f"{entry_desc}: Invalid activity_ref format: {ref}")
        
        # Validate start_time format
        if 'start_time' in pa_entry:
            try:
                datetime.datetime.strptime(pa_entry['start_time'], DATETIME_FORMAT)
            except ValueError:
                errors.append(f"{entry_desc}: Invalid start_time format. Expected {DATETIME_FORMAT}")
        
        # Validate duration if present
        if 'duration' in pa_entry:
            try:
                duration = float(pa_entry['duration'])
                if duration <= 0 or duration > 480:  # Max 8 hours
                    errors.append(f"{entry_desc}: Duration must be between 0 and 480 minutes, got: {duration}")
            except (ValueError, TypeError):
                errors.append(f"{entry_desc}: Duration must be numeric, got: {pa_entry['duration']}")
        
        # Validate intensity if present
        if 'intensity' in pa_entry:
            valid_intensities = ['light', 'moderate', 'high']
            if pa_entry['intensity'] not in valid_intensities:
                errors.append(f"{entry_desc}: Invalid intensity '{pa_entry['intensity']}'. Must be one of: {valid_intensities}")
        
        # Validate expected_hr_increase if present
        if 'expected_hr_increase' in pa_entry:
            try:
                hr_increase = float(pa_entry['expected_hr_increase'])
                if hr_increase < 0 or hr_increase > 200:  # Reasonable heart rate increase range
                    errors.append(f"{entry_desc}: Heart rate increase must be between 0 and 200 bpm, got: {hr_increase}")
            except (ValueError, TypeError):
                errors.append(f"{entry_desc}: Heart rate increase must be numeric, got: {pa_entry['expected_hr_increase']}")
        
        return errors
    
    def resolve_pa_activity_ref(self, pa_entry):
        """
        Resolve physical activity reference to actual configuration.
        
        Parameters
        ----------
        pa_entry : dict
            PA entry that may contain an activity_ref
            
        Returns
        -------
        dict
            Resolved PA entry with activity_ref replaced by actual activity data
        """
        if 'activity_ref' not in pa_entry:
            return pa_entry
        
        # Load the referenced configuration
        activity_ref = pa_entry['activity_ref']
        try:
            referenced_config = self.load_reusable_pa_config(activity_ref)
            
            # Create resolved entry by merging reference with overrides
            resolved_entry = copy.deepcopy(pa_entry)
            del resolved_entry['activity_ref']  # Remove the reference
            
            # Apply defaults from referenced config, but don't override explicit values
            if 'physical_activity_entries' in referenced_config and referenced_config['physical_activity_entries']:
                ref_entry = referenced_config['physical_activity_entries'][0]  # Use first entry as template
                
                for key, value in ref_entry.items():
                    if key not in resolved_entry:  # Don't override explicit values
                        resolved_entry[key] = value
            
            logger.debug(f"Resolved activity reference {activity_ref} for PA entry")
            return resolved_entry
            
        except Exception as e:
            raise ValueError(f"Failed to resolve activity reference '{activity_ref}': {e}")
    
    def process_pa_entries_with_validation(self, pa_entries):
        """
        Process physical activity entries with validation and reference resolution.
        
        Parameters
        ----------
        pa_entries : list
            List of PA entries (may include reusable references)
            
        Returns
        -------
        list
            Processed and validated PA entries
        """
        if not pa_entries:
            return []
        
        processed_entries = []
        all_errors = []
        
        for i, entry in enumerate(pa_entries):
            try:
                # Check if this is a profile reference
                if isinstance(entry, str) and entry.startswith('reusable.physical_activities.profiles.'):
                    # Load entire profile configuration
                    profile_config = self.load_reusable_pa_config(entry)
                    
                    # Validate profile has required structure
                    if 'physical_activity_entries' not in profile_config:
                        all_errors.append(f"Profile {entry}: Missing 'physical_activity_entries'")
                        continue
                    
                    # Add all entries from the profile
                    for profile_entry in profile_config['physical_activity_entries']:
                        # Validate each entry from the profile
                        errors = self.validate_pa_entry(profile_entry, f"{i} (from profile {entry})")
                        if errors:
                            all_errors.extend(errors)
                            continue
                        
                        processed_entries.append(profile_entry)
                
                else:
                    # Validate entry
                    errors = self.validate_pa_entry(entry, i)
                    if errors:
                        all_errors.extend(errors)
                        continue
                    
                    # Resolve activity references
                    resolved_entry = self.resolve_pa_activity_ref(entry)
                    processed_entries.append(resolved_entry)
                    
            except Exception as e:
                all_errors.append(f"PA entry {i}: {str(e)}")
        
        # Report validation errors
        if all_errors:
            error_msg = "Physical Activity validation errors:\n" + "\n".join(all_errors)
            logger.error(error_msg)
            raise ValueError(error_msg)
        
        logger.info(f"Successfully processed {len(processed_entries)} PA entries")
        return processed_entries
    
    def enhanced_physical_activity_entries_to_timeline(self, pa_entries):
        """
        Enhanced version of physical_activity_entries_to_timeline with Phase 3 features.
        
        Parameters
        ----------
        pa_entries : list
            List of physical activity entries (may include reusable references)
            
        Returns
        -------
        PhysicalActivityTimeline
            Timeline with resolved and validated activities
        """
        if not pa_entries:
            return PhysicalActivityTimeline()
        
        # Process entries with validation and reference resolution
        processed_entries = self.process_pa_entries_with_validation(pa_entries)
        
        # Convert to timeline using the existing method
        return self.physical_activity_entries_to_timeline(processed_entries)
    
    # ===== END PHASE 2 & 3 =====

    def get_sensor_config(self):
        return SensorConfig(self.sensor_glucose_history)

    def get_patient_config(self):

        pa_timeline_from_model = self.patient_model.get("pa_timeline", PhysicalActivityTimeline())

        patient_config = PatientConfig(
            basal_schedule=self.patient_model["basal_rate_schedule"],
            carb_ratio_schedule=self.patient_model["carb_ratio_schedule"],
            insulin_sensitivity_schedule=self.patient_model["insulin_sensitivity_schedule"],
            glucose_sensitivity_factor_schedule=self.patient_model["glucose_sensitivity_factor_schedule"],
            basal_blood_glucose_schedule=self.patient_model["basal_blood_glucose_schedule"],
            insulin_production_rate_schedule=self.patient_model["insulin_production_rate_schedule"],
            glucose_history=self.patient_model_glucose_history,
            carb_event_timeline=self.patient_model["carb_timeline"],
            bolus_event_timeline=self.patient_model["bolus_timeline"],
            action_timeline=self.patient_model["action_timeline"],
            pa_timeline=pa_timeline_from_model,  # Phase 2: Add PA timeline
            patient_insulin_type=self.patient_model.get("patient_insulin_type", "rapid_acting_adult"),
            # Physical activity and metabolism model parameters with sensible defaults
            w_hr=self.patient_model.get("w_hr", 0.0),  # Heart rate weight parameter
            a=self.patient_model.get("a", 1.0),        # Metabolism model parameter
            tau=self.patient_model.get("tau", 60.0),   # Time constant parameter
            n=self.patient_model.get("n", 1.0),        # Exponential parameter
        )

        # Phase 2: Generate heart rate trace based on physical activity
        if hasattr(self, 'sim_start_time') and hasattr(self, 'duration_hrs'):
            patient_config.hr_trace = get_heartrate_trace(
                pa_timeline=patient_config.pa_timeline,
                t0=self.sim_start_time,
                sim_length=self.duration_hrs,
                heart_rate_trace=None
            )
        else:
            # Fallback for cases where timing variables aren't set
            from tidepool_data_science_simulator.models.measures import HeartRateTrace
            patient_config.hr_trace = HeartRateTrace()

        patient_config.recommendation_accept_prob = 0  # Currently, all bolus are specified
        return patient_config

    def get_pump_config(self):

        return PumpConfig(
            basal_schedule=self.pump_model["basal_rate_schedule"],
            carb_ratio_schedule=self.pump_model["carb_ratio_schedule"],
            insulin_sensitivity_schedule=self.pump_model["insulin_sensitivity_schedule"],
            target_range_schedule=self.pump_model["target_range_schedule"],
            carb_event_timeline=self.pump_model["carb_timeline"],
            bolus_event_timeline=self.pump_model["bolus_timeline"]
        )


