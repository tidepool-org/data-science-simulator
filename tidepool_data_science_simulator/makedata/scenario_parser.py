__author__ = "Cameron Summers"

"""
Parsers are interfaces for gathering information from configuration files
into a normalized format for the simulation.
"""

import pandas as pd
import numpy as np

from tidepool_data_science_simulator.legacy.read_fda_risk_input_scenarios_ORIG import input_table_to_dict
from tidepool_data_science_simulator.models.simulation import EventTimeline, SettingSchedule24Hr
from tidepool_data_science_simulator.models.measures import (
    Carb,
    Bolus,
    BasalRate,
    CarbInsulinRatio,
    InsulinSensitivityFactor,
    TargetRange,
    GlucoseTrace,
)


class SimulationParser(object):
    def get_simulation_config(self):
        raise NotImplementedError

    def get_pump_config(self):
        raise NotImplementedError

    def get_sensor_config(self):
        raise NotImplementedError

    def get_patient_config(self):
        raise NotImplementedError

    def get_controller_config(self):
        raise NotImplementedError

    def get_simulation_start_time(self):
        raise NotImplementedError

    def get_simulation_duration_hours(self):
        raise NotImplementedError


class ScenarioParserCSV(SimulationParser):
    """
    Parser for scenarios for FDA risk analysis May 2020.
    """

    def __init__(self, path_to_csv=None):
        self.csv_path = path_to_csv
        data = pd.read_csv(path_to_csv, sep="\t")
        custom_table_df = data.set_index("setting_name")
        self.tmp_dict = input_table_to_dict(custom_table_df)

    def get_simulation_config(self):
        """
        Shortcut for passing the Loop specific format of current scenario files.

        Returns
        -------
        dict
            Scenario dict from the original code
        """
        return self.tmp_dict

    def get_pump_config(self):
        """
        Get a config object with relevant pump information. Pump takes settings
        from scenario that are programmed and may differ from the patient or "actual" settings.

        Returns
        -------
        PumpConfig
        """

        time = self.get_simulation_start_time()

        basal_schedule = SettingSchedule24Hr(
            time,
            "Basal Rate",
            start_times=self.tmp_dict.get("basal_rate_start_times"),
            values=[
                BasalRate(rate, units)
                for rate, units in zip(
                    self.tmp_dict.get("basal_rate_values"),
                    self.tmp_dict.get("basal_rate_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get("basal_rate_minutes"),
        )

        carb_ratio_schedule = SettingSchedule24Hr(
            time,
            "Carb Insulin Ratio",
            start_times=self.tmp_dict.get("carb_ratio_start_times"),
            values=[
                CarbInsulinRatio(value, units)
                for value, units in zip(
                    self.tmp_dict.get("carb_ratio_values"),
                    self.tmp_dict.get("carb_ratio_value_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get(
                "carb_ratio_minutes", [1440]
            ),  # TODO: hack
        )

        insulin_sensitivity_schedule = SettingSchedule24Hr(
            time,
            "Insulin Sensitivity",
            start_times=self.tmp_dict.get("sensitivity_ratio_start_times"),
            values=[
                InsulinSensitivityFactor(value, units)
                for value, units in zip(
                    self.tmp_dict.get("sensitivity_ratio_values"),
                    self.tmp_dict.get("sensitivity_ratio_value_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get(
                "sensitivity_ratio_minutes", [1440]
            ),  # TODO: hack
        )

        target_range_schedule = SettingSchedule24Hr(
            time,
            "Target Range",
            start_times=self.tmp_dict.get("target_range_start_times"),
            values=[
                TargetRange(min_value, max_value, units)
                for min_value, max_value, units in zip(
                    self.tmp_dict.get("target_range_minimum_values"),
                    self.tmp_dict.get("target_range_maximum_values"),
                    self.tmp_dict.get("target_range_value_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get(
                "target_range_minutes", [1440]
            ),  # TODO: hack
        )

        carb_events = EventTimeline(
            datetimes=self.tmp_dict["carb_dates"],
            events=[
                Carb(value, units, duration)
                for value, units, duration in zip(
                    self.tmp_dict["carb_values"],
                    self.tmp_dict["carb_value_units"],
                    self.tmp_dict["carb_absorption_times"],
                )
            ],
        )

        insulin_events = EventTimeline(
            datetimes=self.tmp_dict["dose_start_times"],
            events=[
                Bolus(value, units)
                for value, units in zip(
                    self.tmp_dict["dose_values"], self.tmp_dict["dose_value_units"]
                )
            ],
        )

        glucose_history = GlucoseTrace(
            datetimes=self.tmp_dict["glucose_dates"],
            values=self.tmp_dict["glucose_values"],
        )

        pump_config = PumpConfig(
            basal_schedule=basal_schedule,
            carb_ratio_schedule=carb_ratio_schedule,
            insulin_sensitivity_schedule=insulin_sensitivity_schedule,
            target_range_schedule=target_range_schedule,
            carb_events=carb_events,
            insulin_events=insulin_events,
            glucose_history=glucose_history,
        )

        return pump_config

    def get_sensor_config(self):
        """
        Get a glucose trace object to give to sensor.

        Returns
        -------
        GlucoseTrace
        """

        sensor_glucose_history = GlucoseTrace(
            datetimes=self.tmp_dict["glucose_dates"],
            values=self.tmp_dict["glucose_values"],
        )

        return sensor_glucose_history

    def get_patient_config(self):
        """
        Get a config object with relevant patient information. Patient takes settings
        from scenario that are "actual" settings.

        Returns
        -------
        PatientConfig
        """

        time = self.get_simulation_start_time()

        basal_schedule = SettingSchedule24Hr(
            time,
            "Basal Rate",
            start_times=self.tmp_dict.get("basal_rate_start_times"),
            values=[
                BasalRate(rate, units)
                for rate, units in zip(
                    self.tmp_dict.get("actual_basal_rates"),
                    self.tmp_dict.get("basal_rate_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get("basal_rate_minutes"),
        )

        carb_ratio_schedule = SettingSchedule24Hr(
            time,
            "Carb Insulin Ratio",
            start_times=self.tmp_dict.get("carb_ratio_start_times"),
            values=[
                CarbInsulinRatio(value, units)
                for value, units in zip(
                    self.tmp_dict.get("actual_carb_ratios"),
                    self.tmp_dict.get("carb_ratio_value_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get(
                "carb_ratio_minutes", [1440]
            ),  # TODO: hack
        )

        insulin_sensitivity_schedule = SettingSchedule24Hr(
            time,
            "Insulin Sensitivity",
            start_times=self.tmp_dict.get("sensitivity_ratio_start_times"),
            values=[
                InsulinSensitivityFactor(value, units)
                for value, units in zip(
                    self.tmp_dict.get("actual_sensitivity_ratios"),
                    self.tmp_dict.get("sensitivity_ratio_value_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get(
                "sensitivity_ratio_minutes", [1440]
            ),  # TODO: hack
        )

        target_range_schedule = SettingSchedule24Hr(
            time,
            "Target Range",
            start_times=self.tmp_dict.get("target_range_start_times"),
            values=[
                TargetRange(min_value, max_value, units)
                for min_value, max_value, units in zip(
                    self.tmp_dict.get("target_range_minimum_values"),
                    self.tmp_dict.get("target_range_maximum_values"),
                    self.tmp_dict.get("target_range_value_units"),
                )
            ],
            duration_minutes=self.tmp_dict.get(
                "target_range_minutes", [1440]
            ),  # TODO: hack
        )

        carb_events = EventTimeline(
            datetimes=self.tmp_dict["carb_dates"],
            events=[
                Carb(value, units, duration)
                for value, units, duration in zip(
                    self.tmp_dict["actual_carbs"],
                    self.tmp_dict["carb_value_units"],
                    self.tmp_dict["carb_absorption_times"],
                )
            ],
        )

        insulin_events = EventTimeline(
            datetimes=self.tmp_dict["dose_start_times"],
            events=[
                Bolus(value, units)
                for value, units in zip(
                    self.tmp_dict["actual_doses"], self.tmp_dict["dose_value_units"]
                )
            ],
        )

        glucose_history = GlucoseTrace(
            datetimes=self.tmp_dict["glucose_dates"],
            values=self.tmp_dict["actual_blood_glucose"],
        )

        patient_config = PatientConfig(
            basal_schedule=basal_schedule,
            carb_ratio_schedule=carb_ratio_schedule,
            insulin_sensitivity_schedule=insulin_sensitivity_schedule,
            target_range_schedule=target_range_schedule,
            carb_events=carb_events,
            insulin_events=insulin_events,
            glucose_history=glucose_history,
        )

        return patient_config

    def get_controller_config(self):
        """
        Get the Loop controller configuration.

        TODO: This is a shortcut to get loop working. We'll want to construct a general
                config object for this.

        Returns
        -------
        dict
            The settings dictionary
        """
        controller_dict = self.tmp_dict["settings_dictionary"]
        return controller_dict

    def get_simulation_start_time(self):
        """
        Get the simulation start time.

        Returns
        -------
        datetime
            t=0 for simulation
        """
        return self.tmp_dict["time_to_calculate_at"]

    def get_simulation_duration_hours(self):
        """
        Get the length of the simulation

        TODO: read from scenario config

        Returns
        -------
        float
            Length in hours of the simulation
        """
        return 8.0


class PatientConfig(object):
    def __init__(
        self,
        basal_schedule,
        carb_ratio_schedule,
        insulin_sensitivity_schedule,
        target_range_schedule,
        glucose_history,
        carb_events,
        insulin_events,
    ):

        self.basal_schedule = basal_schedule
        self.carb_ratio_schedule = carb_ratio_schedule
        self.insulin_sensitivity_schedule = insulin_sensitivity_schedule
        self.target_range_schedule = target_range_schedule

        self.insulin_events = insulin_events
        self.carb_events = carb_events

        self.glucose_history = glucose_history


class PumpConfig(object):
    def __init__(
        self,
        basal_schedule,
        carb_ratio_schedule,
        insulin_sensitivity_schedule,
        target_range_schedule,
        glucose_history,
        carb_events,
        insulin_events,
    ):

        self.basal_schedule = basal_schedule
        self.carb_ratio_schedule = carb_ratio_schedule
        self.insulin_sensitivity_schedule = insulin_sensitivity_schedule
        self.target_range_schedule = target_range_schedule

        self.insulin_events = insulin_events
        self.carb_events = carb_events

        self.glucose_history = glucose_history

        # FIXME - these should be explicit somewhere
        self.max_temp_basal = np.inf
        self.max_bolus = np.inf
