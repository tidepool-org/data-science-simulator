__author__ = "Cameron Summers"

import pandas as pd
import numpy as np
import copy
import json
import os
import datetime

import logging
logger = logging.getLogger(__name__)

from tidepool_data_science_simulator.legacy.read_fda_risk_input_scenarios_ORIG import input_table_to_dict
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, TempBasalTimeline, ActionTimeline, VirtualPatientDeleteLoopData
from tidepool_data_science_simulator.models.measures import (
    Carb,
    Bolus,
    BasalRate,
    CarbInsulinRatio,
    InsulinSensitivityFactor,
    TargetRange,
    GlucoseTrace,
)

from tidepool_data_science_simulator.makedata.scenario_parser import (
    SimulationParser, SensorConfig, PatientConfig, ControllerConfig, PumpConfig
)
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.controller import LoopController, DoNothingController
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


POINTER_OBJ_DIR = os.path.dirname(__file__) + "/../../scenario_configs/tidepool_risk_v2/"
DATETIME_FORMAT = "%m/%d/%Y %H:%M:%S"

CONTROLLER_MODEL_NAME_MAP = {
    "rapid_acting_adult": [360, 75],
    "rapid_acting_child": [360, 65],
    "walsh": [120, 15],
    "fiasp": [360, 55]
}


class ScenarioParserV2(SimulationParser):
    """
    Redesigned scenario parser for Tidepool Risk automated pipeline, Feb 2021.
    """

    def __init__(self, path_to_json_config=None):

        self.pointer_keyword = "reusable"

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
        Get simulation objects as specified by the config file. By design there is one simulation for
        each override config. The base simulation configuration is not built, but if an empty dict
        is passed in the override list, this is a sim with the base config and no overrides.

        Returns
        -------
            list: [Simulation]
        """
        simulations = dict()
        for override_delta in self.override_configs:

            override_sim_config = copy.deepcopy(self.base_sim_config)  # override alters the config in place, so start fresh to avoid issues
            self.apply_config_override(override_sim_config, override_delta)

            sim_id = override_sim_config["sim_id"]

            if override_json_save_dir is not None:
                filepath = os.path.join(override_json_save_dir, sim_id + "_override_config.json")
                json.dump(override_sim_config, open(filepath, "w"), indent=4)

            sim = self.build_sim_from_config(override_sim_config)
            sim.name = sim_id
            simulations[sim_id] = sim

        return simulations

    def apply_config_override(self, base_sim_config, override_delta):
        """
        Modifies the config object in place and:
            1. Resolves pointer references to other files in the config
            2. Resolves the overriding leaf note configs
        """

        self.resolve_pointers(base_sim_config)
        self.resolve_pointers(override_delta)

        num_overrides = self.count_leaf_nodes(override_delta)
        num_overrides_applied = self.resolve_override(base_sim_config, override_delta)

        if num_overrides_applied != num_overrides:
            raise Exception("Only applied {} of {} overriding values in {}. Check configurations.".format(num_overrides_applied, num_overrides, override_delta))

    def count_leaf_nodes(self, obj):
        """
        Count the number of non-dict values in the object. Used to later
        validate that all overrides have been applied.

        Parameters
        ----------
        obj

        Returns
        -------
        int: Number of non-dict values
        """
        num_leaf_nodes = 0
        for k, v in obj.items():
            if not isinstance(v, dict):
                num_leaf_nodes += 1
            else:
                num_leaf_nodes += self.count_leaf_nodes(v)
        return num_leaf_nodes

    def resolve_pointers(self, value):
        """
        Recursively traverse the simulation config obj and replace any pointers with their objects.
        The relied on assumption to make this simply is the value is a pointer if its type is string
        and it has the pointer keyword in it.
        """
        for k, v in value.items():
            if self.is_config_file_pointer(v):
                value[k] = self.load_pointer(v)
            elif isinstance(v, dict):
                self.resolve_pointers(v)

    def resolve_override(self, obj, override_obj):
        """
        Recursively traverse the simulation config obj and apply specified leaf overrides. The
        relied on assumption to make this simple is only values in the override that are not
        dicts themselves (ie leaf nodes) are overridden.
        """
        num_overides_applied = 0
        for k, v in obj.items():

            if k in override_obj:
                if not isinstance(override_obj[k], dict):  # key is there and it's a leaf node
                    obj[k] = override_obj[k]
                    # logger.debug("Applied override {}: {}".format(k, override_obj[k]))
                    num_overides_applied += 1
                else:  # key is there and it's an object that should be explored for leaf overrides
                    num_overides_applied += self.resolve_override(v, override_obj[k])

        return num_overides_applied

    def is_config_file_pointer(self, value):
        """
        Return True if the value matches the pattern for designating a file for a config.
        """
        return (isinstance(value, str) and self.pointer_keyword in value)

    def load_pointer(self, pointer_string):
        """
        Load file object pointed to.
        """
        pointer_segments = pointer_string.split(".")
        folder_path = os.path.join("/".join(pointer_segments[:-1]))
        filename_no_ext = pointer_segments[-1]
        json_filename = "{}.json".format(filename_no_ext)
        csv_filename = "{}.csv".format(filename_no_ext)

        json_path = os.path.join(POINTER_OBJ_DIR, folder_path, json_filename)
        csv_path = os.path.join(POINTER_OBJ_DIR, folder_path, csv_filename)
        if os.path.isfile(json_path):
            obj = json.load(open(json_path, "r"))
        elif os.path.isfile(csv_path):
            obj = pd.read_csv(csv_path).to_dict()
        else:
            raise Exception("Could not load pointer file {}/{}".format(folder_path,filename_no_ext))

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
            durations_minutes = [1440]  # minutes in 24 hours

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

        if not 0 < carb_ratio <= 150:
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

    def build_components_from_config(self, sim_config, sensor=None, pump=None):

        sim_start_time_str = self.get_required_value(sim_config, "time_to_calculate_at", str)
        sim_start_time = datetime.datetime.strptime(sim_start_time_str, DATETIME_FORMAT)

        duration_hrs = self.get_required_value(sim_config, "duration_hours", float)

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

            # Get model parameters from passed string in config
            model_name = controller_settings["model"]
            if model_name not in CONTROLLER_MODEL_NAME_MAP:
                raise ValueError("{} not a recognized model. Available models: {}".format(model_name,
                                                                                          CONTROLLER_MODEL_NAME_MAP.keys()))

            model_params = CONTROLLER_MODEL_NAME_MAP[model_name]
            controller_settings["model"] = model_params

            controller_config = ControllerConfig(
                bolus_event_timeline=self.pump_model["bolus_timeline"],
                carb_event_timeline=self.pump_model["carb_timeline"],
                controller_settings=controller_settings
            )

            controller = LoopController(sim_start_time, controller_config)

        return controller

    def get_sensor_config(self):
        return SensorConfig(self.sensor_glucose_history)

    def get_patient_config(self):

        patient_config = PatientConfig(
            basal_schedule=self.patient_model["basal_rate_schedule"],
            carb_ratio_schedule=self.patient_model["carb_ratio_schedule"],
            insulin_sensitivity_schedule=self.patient_model["insulin_sensitivity_schedule"],
            glucose_history=self.patient_model_glucose_history,
            carb_event_timeline=self.patient_model["carb_timeline"],
            bolus_event_timeline=self.patient_model["bolus_timeline"],
            action_timeline=self.patient_model["action_timeline"]
        )

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
