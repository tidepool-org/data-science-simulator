__author__ = "Cameron Summers"

"""
Parsers are interfaces for gathering information from configuration files
into a normalized format for the simulation.
"""

import pandas as pd
import numpy as np
import copy

from tidepool_data_science_simulator.legacy.read_fda_risk_input_scenarios_ORIG import input_table_to_dict
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, TempBasalTimeline, ActionTimeline
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
    Parser for the monolithic scenario files.
    """

    def __init__(self, path_to_csv=None):
        if path_to_csv[-3:] == 'csv':
            separator=","
        else:
            separator = "\t"
        self.csv_path = path_to_csv
        data = pd.read_csv(path_to_csv, sep=separator)
        custom_table_df = data.set_index("setting_name")
        self.tmp_dict = input_table_to_dict(custom_table_df)

        time = self.get_simulation_start_time()

        # ========== Pump =============
        self.pump_basal_schedule = BasalSchedule24hr(
            time,
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

        self.pump_carb_ratio_schedule = SettingSchedule24Hr(
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

        self.pump_insulin_sensitivity_schedule = SettingSchedule24Hr(
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

        self.pump_target_range_schedule = TargetRangeSchedule24hr(
            time,
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

        self.pump_carb_events = CarbTimeline(
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

        self.pump_bolus_events = BolusTimeline(
            datetimes=self.tmp_dict["dose_start_times"],
            events=[
                Bolus(value, units)
                for value, units in zip(
                    self.tmp_dict["dose_values"], self.tmp_dict["dose_value_units"]
                )
            ],
        )

        # ======== Patient ==========
        self.patient_basal_schedule = BasalSchedule24hr(
            time,
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

        self.patient_carb_ratio_schedule = SettingSchedule24Hr(
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

        self.patient_insulin_sensitivity_schedule = SettingSchedule24Hr(
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

        self.patient_target_range_schedule = TargetRangeSchedule24hr(
            time,
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

        self.patient_carb_events = CarbTimeline(
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

        self.patient_bolus_events = BolusTimeline(
            datetimes=self.tmp_dict["dose_start_times"],
            events=[
                Bolus(value, units)
                for value, units in zip(
                    self.tmp_dict["actual_doses"], self.tmp_dict["dose_value_units"]
                )
            ],
        )

        self.patient_glucose_history = GlucoseTrace(
            datetimes=self.tmp_dict["glucose_dates"],
            values=self.tmp_dict["actual_blood_glucose"],
        )

        self.sensor_glucose_history = GlucoseTrace(
            datetimes=self.tmp_dict["glucose_dates"],
            values=self.tmp_dict["glucose_values"],
        )

    def get_pump_config(self):
        """
        Get a config object with relevant pump information. Pump takes settings
        from scenario that are programmed and may differ from the patient or "actual" settings.

        Returns
        -------
        PumpConfig
        """

        pump_config = PumpConfig(
            basal_schedule=self.pump_basal_schedule,
            carb_ratio_schedule=self.pump_carb_ratio_schedule,
            insulin_sensitivity_schedule=self.pump_insulin_sensitivity_schedule,
            target_range_schedule=self.pump_target_range_schedule,
            carb_event_timeline=self.pump_carb_events,
            bolus_event_timeline=self.pump_bolus_events,
        )

        return pump_config

    def get_sensor_config(self):
        """
        Get a glucose trace object to give to sensor.

        Returns
        -------
        GlucoseTrace
        """
        return SensorConfig(copy.deepcopy(self.sensor_glucose_history))

    def get_patient_config(self):
        """
        Get a config object with relevant patient information. Patient takes settings
        from scenario that are "actual" settings.

        Returns
        -------
        PatientConfig
        """

        patient_config = PatientConfig(
            basal_schedule=self.patient_basal_schedule,
            carb_ratio_schedule=self.patient_carb_ratio_schedule,
            insulin_sensitivity_schedule=self.patient_insulin_sensitivity_schedule,
            carb_event_timeline=self.patient_carb_events,
            bolus_event_timeline=self.patient_bolus_events,
            glucose_history=copy.deepcopy(self.patient_glucose_history),
            action_timeline=ActionTimeline()
        )

        patient_config.recommendation_accept_prob = 0.0  # Does not accept any bolus recommendations
        patient_config.min_bolus_rec_threshold = 0.5  # Minimum size of bolus to accept
        patient_config.recommendation_meal_attention_time_minutes = 1e12  # Time since meal to take recommendations

        return patient_config

    def get_controller_config(self):
        """
        Get the Loop controller configuration.

        Returns
        -------
        ControllerConfig
        """
        controller_settings = self.tmp_dict["settings_dictionary"]

        controller_config = ControllerConfig(
            bolus_event_timeline=BolusTimeline(),
            carb_event_timeline=CarbTimeline(),
            # bolus_event_timeline=self.pump_bolus_events,
            # carb_event_timeline=self.pump_carb_events,
            controller_settings=controller_settings
        )

        return controller_config

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

        Returns
        -------
        float
            Length in hours of the simulation
        """
        return 8.0


class ControllerConfig(object):

    def __init__(
        self,
        bolus_event_timeline,
        carb_event_timeline,
        controller_settings
     ):
        self.bolus_event_timeline = bolus_event_timeline
        self.carb_event_timeline = carb_event_timeline
        self.temp_basal_event_timeline = TempBasalTimeline()  # No existing scenario specifies temp basal events
        self.controller_settings = controller_settings

    def get_info_stateless(self):

        return self.controller_settings


class PatientConfig(object):
    def __init__(
        self,
        basal_schedule,
        carb_ratio_schedule,
        insulin_sensitivity_schedule,
        glucose_sensitivity_factor_schedule,
        basal_blood_glucose_schedule,
        insulin_production_rate_schedule,
        glucose_history,
        carb_event_timeline,
        bolus_event_timeline,
        action_timeline,
    ):
        """
        Configuration object for virtual patient.

        Parameters
        ----------
        basal_schedule: SettingSchedule24Hr
            Basal schedule representing equivalent endogenous glucose production

        carb_ratio_schedule: SettingSchedule24Hr
            True carb ratio schedule

        insulin_sensitivity_schedule: SettingSchedule24Hr
            True insulin sentivity schedule

        glucose_history: GlucoseTrace
            Historical glucose previous to t=0

        carb_event_timeline: CarbTimeline
            Timeline of true carb events

        bolus_event_timeline: BolusTimeline
            Timeline of true bolus events
        """

        self.basal_schedule = basal_schedule
        self.carb_ratio_schedule = carb_ratio_schedule
        self.insulin_sensitivity_schedule = insulin_sensitivity_schedule
        self.glucose_sensitivity_factor_schedule = glucose_sensitivity_factor_schedule
        self.basal_blood_glucose_schedule = basal_blood_glucose_schedule
        self.insulin_production_rate_schedule = insulin_production_rate_schedule
        self.bolus_event_timeline = bolus_event_timeline
        self.carb_event_timeline = carb_event_timeline
        self.action_timeline = action_timeline

        self.glucose_history = glucose_history

    def get_info_stateless(self):

        stateless_info = {
            "basal_schedule": self.basal_schedule.get_info_stateless(),
            "carb_ratio_schedule": self.carb_ratio_schedule.get_info_stateless(),
            "insulin_sensitivity_schedule": self.insulin_sensitivity_schedule.get_info_stateless()
        }

        return stateless_info


class PumpConfig(object):
    def __init__(
        self,
        basal_schedule,
        carb_ratio_schedule,
        insulin_sensitivity_schedule,
        target_range_schedule,
        carb_event_timeline,
        bolus_event_timeline,
    ):
        """
        Configuration for pump

        Parameters
        ----------
        basal_schedule: SettingSchedule24Hr
            Basal schedule on the pump

        carb_ratio_schedule: SettingSchedule24Hr
            Carb ratio schedule on the pump

        insulin_sensitivity_schedule: SettingSchedule24Hr
            Insulin sensitivity schedule on the pump

        target_range_schedule: SettingSchedule24Hr
            Target range schedule on the pump

        carb_event_timeline: CarbTimeline
            Carb events on the pump

        bolus_event_timeline: BolusTimeline
            Bolus events on the pump
        """

        self.basal_schedule = basal_schedule
        self.carb_ratio_schedule = carb_ratio_schedule
        self.insulin_sensitivity_schedule = insulin_sensitivity_schedule
        self.target_range_schedule = target_range_schedule

        self.bolus_event_timeline = bolus_event_timeline
        self.carb_event_timeline = carb_event_timeline

        # FIXME - these should be explicit somewhere
        self.max_temp_basal = np.inf
        self.max_bolus = np.inf

    def get_info_stateless(self):

        stateless_info = {
            "basal_schedule": self.basal_schedule.get_info_stateless(),
            "carb_ratio_schedule": self.carb_ratio_schedule.get_info_stateless(),
            "insulin_sensitivity_schedule": self.insulin_sensitivity_schedule.get_info_stateless()
        }

        return stateless_info


class SensorConfig(object):

    def __init__(self, sensor_bg_history=None):

        self.sensor_bg_history = sensor_bg_history

    def get_info_stateless(self):

        return dict()
