__author__ = "Cameron Summers"

"""
Parsers are interfaces for gathering information from configuration files
into a normalized format for the simulation.
"""

import pandas as pd
import numpy as np
import copy
import datetime

from pyloopkit.dose import DoseType

from tidepool_data_science_simulator.legacy.read_fda_risk_input_scenarios_ORIG import input_table_to_dict
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, TempBasalTimeline
from tidepool_data_science_simulator.models.measures import (
    Carb,
    Bolus,
    BasalRate,
    TempBasal,
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
    Parser for scenarios for FDA risk analysis May 2020.
    """

    def __init__(self, path_to_csv=None):
        if path_to_csv[-3:] == 'csv':
            separator=","
        else:
            separator = "\t"
        self.csv_path = path_to_csv
        data = pd.read_csv(path_to_csv, sep=separator)
        custom_table_df = data.set_index("setting_name")
        self.loop_inputs_dict = input_table_to_dict(custom_table_df)

        self.loop_inputs_to_sim_config()

    def loop_inputs_to_sim_config(self):

        time = self.get_simulation_start_time()

        self.transform_pump(time)
        self.transform_patient(time)
        self.transform_sensor()

    def transform_pump(self, time):

        # ========== Pump =============
        self.pump_basal_schedule = BasalSchedule24hr(
            time,
            start_times=self.loop_inputs_dict.get("basal_rate_start_times"),
            values=[
                BasalRate(rate, units)
                for rate, units in zip(
                    self.loop_inputs_dict.get("basal_rate_values"),
                    self.loop_inputs_dict.get("basal_rate_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get("basal_rate_minutes"),
        )

        self.pump_carb_ratio_schedule = SettingSchedule24Hr(
            time,
            "Carb Insulin Ratio",
            start_times=self.loop_inputs_dict.get("carb_ratio_start_times"),
            values=[
                CarbInsulinRatio(value, units)
                for value, units in zip(
                    self.loop_inputs_dict.get("carb_ratio_values"),
                    self.loop_inputs_dict.get("carb_ratio_value_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get(
                "carb_ratio_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["carb_ratio_start_times"])
            ),
        )

        self.pump_insulin_sensitivity_schedule = SettingSchedule24Hr(
            time,
            "Insulin Sensitivity",
            start_times=self.loop_inputs_dict.get("sensitivity_ratio_start_times"),
            values=[
                InsulinSensitivityFactor(value, units)
                for value, units in zip(
                    self.loop_inputs_dict.get("sensitivity_ratio_values"),
                    self.loop_inputs_dict.get("sensitivity_ratio_value_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get(
                "sensitivity_ratio_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["sensitivity_ratio_start_times"])
            ),
        )

        self.pump_target_range_schedule = TargetRangeSchedule24hr(
            time,
            start_times=self.loop_inputs_dict.get("target_range_start_times"),
            values=[
                TargetRange(min_value, max_value, units)
                for min_value, max_value, units in zip(
                    self.loop_inputs_dict.get("target_range_minimum_values"),
                    self.loop_inputs_dict.get("target_range_maximum_values"),
                    self.loop_inputs_dict.get("target_range_value_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get(
                "target_range_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["target_range_start_times"])
            ),
        )

        self.pump_carb_events = CarbTimeline(
            datetimes=self.loop_inputs_dict["carb_dates"],
            events=[
                Carb(value, units, duration)
                for value, units, duration in zip(
                    self.loop_inputs_dict["carb_values"],
                    self.loop_inputs_dict["carb_value_units"],
                    self.loop_inputs_dict["carb_absorption_times"],
                )
            ],
        )

        # Separate bolus and temp basal events
        bolus_start_times, bolus_values, bolus_units, bolus_dose_types, bolus_delivered_units = ([], [], [], [], [])
        temp_basal_start_times, temp_basal_duration_minutes, temp_basal_values, \
            temp_basal_units, temp_basal_dose_types, temp_basal_delivered_units = ([], [], [], [], [], [])
        for start_time, end_time, value, units, dose_type, delivered_units in zip(
                self.loop_inputs_dict["dose_start_times"],
                self.loop_inputs_dict["dose_end_times"],
                self.loop_inputs_dict["dose_values"],
                self.loop_inputs_dict["dose_value_units"],
                self.loop_inputs_dict["dose_types"],
                self.loop_inputs_dict.get("delivered_units", [None]*len(self.loop_inputs_dict["dose_start_times"]))
        ):
            if dose_type == DoseType.bolus:
                bolus_start_times.append(start_time)
                bolus_values.append(value)
                bolus_units.append(units)
                bolus_dose_types.append(dose_type)
                bolus_delivered_units.append(delivered_units)
            elif dose_type == DoseType.tempbasal or dose_type == DoseType.basal:
                duration_minutes = (end_time - start_time).total_seconds() / 60
                temp_basal_start_times.append(start_time)
                temp_basal_duration_minutes.append(duration_minutes)
                temp_basal_values.append(value)
                temp_basal_units.append(units)
                temp_basal_dose_types.append(dose_type)
                temp_basal_delivered_units.append(delivered_units)
            else:
                raise Exception("Unknown dose type")

        self.pump_bolus_events = BolusTimeline(
            datetimes=bolus_start_times,
            events=[
                Bolus(value, units)
                for value, units, dose_type in zip(
                    bolus_values,
                    bolus_units,
                    bolus_dose_types
                )
            ],
        )

        self.pump_temp_basal_events = TempBasalTimeline(
            datetimes=temp_basal_start_times,
            events=[
                TempBasal(start_time, value, duration_minutes, units, delivered_units=delivered_units)
                for start_time, value, units, duration_minutes, delivered_units in zip(
                    temp_basal_start_times,
                    temp_basal_values,
                    temp_basal_units,
                    temp_basal_duration_minutes,
                    temp_basal_delivered_units
                )
            ],
        )

    def transform_patient(self, time):

        self.patient_basal_schedule = BasalSchedule24hr(
            time,
            start_times=self.loop_inputs_dict.get("basal_rate_start_times"),
            values=[
                BasalRate(rate, units)
                for rate, units in zip(
                    self.loop_inputs_dict.get("actual_basal_rates"),
                    self.loop_inputs_dict.get("basal_rate_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get("basal_rate_minutes"),
        )

        self.patient_carb_ratio_schedule = SettingSchedule24Hr(
            time,
            "Carb Insulin Ratio",
            start_times=self.loop_inputs_dict.get("carb_ratio_start_times"),
            values=[
                CarbInsulinRatio(value, units)
                for value, units in zip(
                    self.loop_inputs_dict.get("actual_carb_ratios"),
                    self.loop_inputs_dict.get("carb_ratio_value_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get(
                "carb_ratio_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["carb_ratio_start_times"])
            ),
        )

        self.patient_insulin_sensitivity_schedule = SettingSchedule24Hr(
            time,
            "Insulin Sensitivity",
            start_times=self.loop_inputs_dict.get("sensitivity_ratio_start_times"),
            values=[
                InsulinSensitivityFactor(value, units)
                for value, units in zip(
                    self.loop_inputs_dict.get("actual_sensitivity_ratios"),
                    self.loop_inputs_dict.get("sensitivity_ratio_value_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get(
                "sensitivity_ratio_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["sensitivity_ratio_start_times"])
            ),
        )

        self.patient_target_range_schedule = TargetRangeSchedule24hr(
            time,
            start_times=self.loop_inputs_dict.get("target_range_start_times"),
            values=[
                TargetRange(min_value, max_value, units)
                for min_value, max_value, units in zip(
                    self.loop_inputs_dict.get("target_range_minimum_values"),
                    self.loop_inputs_dict.get("target_range_maximum_values"),
                    self.loop_inputs_dict.get("target_range_value_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get(
                "target_range_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["target_range_start_times"])
            ),
        )

        self.patient_carb_events = CarbTimeline(
            datetimes=self.loop_inputs_dict["carb_dates"],
            events=[
                Carb(value, units, duration)
                for value, units, duration in zip(
                    self.loop_inputs_dict["actual_carbs"],
                    self.loop_inputs_dict["carb_value_units"],
                    self.loop_inputs_dict["carb_absorption_times"],
                )
            ],
        )

        self.patient_bolus_events = BolusTimeline(
            datetimes=self.loop_inputs_dict["dose_start_times"],
            events=[
                Bolus(value, units)
                for value, units in zip(
                    self.loop_inputs_dict["actual_doses"], self.loop_inputs_dict["dose_value_units"]
                )
            ],
        )

        self.patient_glucose_history = GlucoseTrace(
            datetimes=self.loop_inputs_dict["glucose_dates"],
            values=self.loop_inputs_dict["actual_blood_glucose"],
        )

    def transform_sensor(self):

        self.sensor_glucose_history = GlucoseTrace(
            datetimes=self.loop_inputs_dict["glucose_dates"],
            values=self.loop_inputs_dict["glucose_values"],
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
            temp_basal_event_timeline=self.pump_temp_basal_events
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
        )

        return patient_config

    def get_controller_config(self):
        """
        Get the Loop controller configuration.

        Returns
        -------
        dict
            The settings dictionary
        """
        controller_settings = self.loop_inputs_dict["settings_dictionary"]

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
        return self.loop_inputs_dict["time_to_calculate_at"]

    def get_simulation_duration_hours(self):
        """
        Get the length of the simulation

        Returns
        -------
        float
            Length in hours of the simulation
        """
        return 8.0


def start_times_to_minutes_durations(start_times):
    """
    Transform start times for settings to minute durations.

    Parameters
    ----------
    start_times: [datetime]
        Ordered start times of settings. Assumed first value is 00:00.

    Returns
    -------
    list
        Minute durations between start times.
    """
    assert start_times[0] == datetime.time(0,0,0)

    minutes_durations = []
    for i in range(1, len(start_times)):
        datetime_end = datetime.datetime.combine(datetime.datetime.today(), start_times[i])
        datetime_start = datetime.datetime.combine(datetime.datetime.today(), start_times[i-1])
        delta_minutes = (datetime_end - datetime_start).total_seconds() / 60
        minutes_durations.append(delta_minutes)

    # Last setting of the day to
    datetime_start = datetime.datetime.combine(datetime.datetime.today(), start_times[-1])
    datetime_end = datetime.datetime.combine(datetime.datetime.today(), start_times[0])
    last_delta_minutes = (datetime_end - datetime_start).total_seconds() / 60
    minutes_durations.append(last_delta_minutes)

    return minutes_durations


class ControllerConfig(object):

    def __init__(
        self,
        bolus_event_timeline,
        carb_event_timeline,
        controller_settings,
        temp_basal_timeline=TempBasalTimeline()
     ):
        self.bolus_event_timeline = bolus_event_timeline
        self.carb_event_timeline = carb_event_timeline
        self.temp_basal_event_timeline = temp_basal_timeline  # No existing scenario specifies temp basal events
        self.controller_settings = controller_settings


class PatientConfig(object):
    def __init__(
        self,
        basal_schedule,
        carb_ratio_schedule,
        insulin_sensitivity_schedule,
        glucose_history,
        carb_event_timeline,
        bolus_event_timeline,
        real_glucose=None,
        action_timeline=None,
        recommendation_accept_prob=1.0,
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

        recommendation_accept_prob: float
            Probability of patient accepting a bolus recommendation
        """

        self.basal_schedule = basal_schedule
        self.carb_ratio_schedule = carb_ratio_schedule
        self.insulin_sensitivity_schedule = insulin_sensitivity_schedule

        self.bolus_event_timeline = bolus_event_timeline
        self.carb_event_timeline = carb_event_timeline
        self.action_timeline = action_timeline

        self.glucose_history = glucose_history
        self.real_glucose = real_glucose

        self.recommendation_accept_prob = recommendation_accept_prob


class PumpConfig(object):
    def __init__(
        self,
        basal_schedule,
        carb_ratio_schedule,
        insulin_sensitivity_schedule,
        target_range_schedule,
        carb_event_timeline,
        bolus_event_timeline,
        temp_basal_event_timeline,
        max_temp_basal=np.inf,
        max_bolus=np.inf
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
        self.temp_basal_event_timeline = temp_basal_event_timeline

        # FIXME - these should be explicit somewhere
        self.max_temp_basal = max_temp_basal
        self.max_bolus = max_bolus


class SensorConfig(object):

    def __init__(self, sensor_bg_history=None):

        self.sensor_bg_history = sensor_bg_history