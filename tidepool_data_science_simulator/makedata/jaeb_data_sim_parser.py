__author__ = "Eden Grown-Haeberli"

import json
import os

from pyloopkit import pyloop_parser
import pandas as pd
import datetime
from math import isnan
from pyloopkit.dose import DoseType

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.legacy.read_fda_risk_input_scenarios_ORIG import input_table_to_dict
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.makedata.scenario_parser import start_times_to_minutes_durations
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

class JaebDataSimParser(ScenarioParserCSV):

    def __init__(self, path_to_settings, path_to_time_series_data, t0=None):

        issue_report_settings = pd.read_csv(path_to_settings, sep=",")
        # TODO: Parse path_to_time_series to choose appropriate settings row
        file_name = path_to_time_series_data.split(sep="/")[-1]
        user_numbers = "LOOP-" + file_name.split(sep="-")[1]
        report_num = int(file_name.split(sep="-")[3])
        settings_data = (issue_report_settings.loc[
                        issue_report_settings['loop_id'] == user_numbers])
        settings_data = settings_data.loc[settings_data['report_num'] == report_num]
        carb_data = pd.read_csv(path_to_time_series_data, sep=",", usecols=['rounded_local_time', 'carbs'])
        target_data = pd.read_csv(path_to_time_series_data, sep=",", usecols=['rounded_local_time',
                                                                              'bg_target_lower', 'bg_target_upper'])

        self.parse_settings_and_carbs(settings_df=settings_data, carb_df=carb_data, target_df=target_data)

        # TODO: load settings
        if t0 is not None:
            time = t0
        else:
            time = self.get_simulation_start_time()

        self.transform_pump(time)
        self.transform_patient(time)
        self.transform_sensor()

    def parse_settings_and_carbs(self, settings_df, carb_df, target_df):
        jaeb_inputs = parse_settings(settings_df=settings_df)

        target_data = target_df.dropna()
        carb_data = carb_df.dropna()
        jaeb_inputs['target_range_minimum_values'] = target_data['bg_target_lower'].drop_duplicates().values
        jaeb_inputs['target_range_maximum_values'] = target_data['bg_target_upper'].drop_duplicates().values
        jaeb_inputs['target_range_start_times'] = [datetime.time(0, 0, 0)]
        jaeb_inputs['carb_values'] = carb_data['carbs'].values
        jaeb_inputs['carb_dates'] = carb_data['rounded_local_time'].map(
            lambda time: datetime.datetime.fromisoformat(time)
        ).values

        # add any other important values
        # TODO: Make this flexible
        jaeb_inputs["basal_rate_units"] = ["U" for _ in jaeb_inputs["basal_rate_values"]]
        jaeb_inputs["carb_ratio_value_units"] = [jaeb_inputs["carb_value_units"] + "/U" for _ in jaeb_inputs[
            "carb_ratio_values"]]
        jaeb_inputs["target_range_value_units"] = ["mg/dL" for _ in jaeb_inputs["target_range_minimum_values"]]
        jaeb_inputs["sensitivity_ratio_value_units"] = ["mg/dL/U" for _ in jaeb_inputs["sensitivity_ratio_values"]]
        jaeb_inputs["carb_value_units"] = [jaeb_inputs["carb_value_units"] for _ in jaeb_inputs['carb_values']]
       # fixme: in the future, pull duration values from the data

        self.loop_inputs_dict = jaeb_inputs

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
            duration_minutes=self.loop_inputs_dict.get(
                "basal_rate_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["basal_rate_start_times"])
            )
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

        # TODO: Check in about this - bg_target is part of time series data. should I load in the first set?
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

        # Import bolus and basal data
        # Separate bolus and temp basal events
        # bolus_start_times, bolus_values, bolus_units, bolus_dose_types, bolus_delivered_units = ([], [], [], [], [])
        # temp_basal_start_times, temp_basal_duration_minutes, temp_basal_values, \
        #     temp_basal_units, temp_basal_dose_types, temp_basal_delivered_units = ([], [], [], [], [], [])
        # for start_time, end_time, value, units, dose_type, delivered_units in zip(
        #         self.loop_inputs_dict["dose_start_times"],
        #         self.loop_inputs_dict["dose_end_times"],
        #         self.loop_inputs_dict["dose_values"],
        #         self.loop_inputs_dict["dose_value_units"],
        #         self.loop_inputs_dict["dose_types"],
        #         self.loop_inputs_dict.get("delivered_units", [None]*len(self.loop_inputs_dict["dose_start_times"]))
        # ):
        #     if dose_type == DoseType.bolus:
        #         bolus_start_times.append(start_time)
        #         bolus_values.append(value)
        #         bolus_units.append(units)
        #         bolus_dose_types.append(dose_type)
        #         bolus_delivered_units.append(delivered_units)
        #     elif dose_type == DoseType.tempbasal or dose_type == DoseType.basal:
        #         duration_minutes = (end_time - start_time).total_seconds() / 60
        #         temp_basal_start_times.append(start_time)
        #         temp_basal_duration_minutes.append(duration_minutes)
        #         temp_basal_values.append(value)
        #         temp_basal_units.append(units)
        #         temp_basal_dose_types.append(dose_type)
        #         temp_basal_delivered_units.append(delivered_units)
        #     else:
        #         raise Exception("Unknown dose type")

        self.pump_bolus_events = BolusTimeline(
            datetimes=[],
            events=[]
        )

        self.pump_temp_basal_events = TempBasalTimeline(
            datetimes=[],
            events=[]
        )

    def transform_patient(self, time):

        self.patient_basal_schedule = BasalSchedule24hr(
            time,
            start_times=self.loop_inputs_dict.get("basal_rate_start_times"),
            values=[
                BasalRate(rate, units)
                for rate, units in zip(
                    self.loop_inputs_dict.get("basal_rate_values"),
                    self.loop_inputs_dict.get("basal_rate_units"),
                )
            ],
            duration_minutes=self.loop_inputs_dict.get("basal_rate_minutes", start_times_to_minutes_durations(
                self.loop_inputs_dict["basal_rate_start_times"])),
        )

        self.patient_carb_ratio_schedule = SettingSchedule24Hr(
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

        self.patient_insulin_sensitivity_schedule = SettingSchedule24Hr(
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
                    self.loop_inputs_dict["carb_values"],
                    self.loop_inputs_dict["carb_value_units"],
                    self.loop_inputs_dict["carb_absorption_times"],
                )
            ],
        )

        self.patient_bolus_events = BolusTimeline(
            datetimes=[],
            events=[],
        )

        self.patient_glucose_history = GlucoseTrace(
            datetimes=[],
            values=[],
        )

    def transform_sensor(self):

        self.sensor_glucose_history = GlucoseTrace(
            datetimes=[],
            values=[],
        )


def parse_settings(settings_df):
    dict_ = dict()
    for col, val in settings_df.iteritems():
        if col == "insulin_model":
            pass
        elif "report_timestamp" in col:
            dict_["time_to_calculate_at"] = datetime.datetime.fromisoformat(
                pd.to_datetime(settings_df[col].values[0]).isoformat()
            )
        elif col == "carb_ratio_unit":
            dict_["carb_value_units"] = val.values[0]
        elif "maximum_" in col:
            col = col.replace("maximum", "max")
            dict_[col] = val.values[0]
        elif "suspend_threshold" == col:
            dict_[col] = val.values[0]
        elif "retrospective_correction_enabled" == col:
            dict_[col] = val.values[0]
        elif "carb_default_absorption" in col:
            if "carb_absorption_times" not in dict_:
                dict_["carb_absorption_times"] = [0, 0, 0]
            value = float(val.values[0])
            length = values.split("_")[-1]
            if length == "slow":
                dict_["carb_absorption_times"][2] = value
            elif length == "medium":
                dict_["carb_absorption_times"][1] = value
            elif length == "fast":
                dict_["carb_absorption_times"][0] = value
        elif col.endswith("_schedule"):
            schedule_type = col.replace("schedule", "")
            if "sensitivity" in col:
                schedule_type = "sensitivity_ratio_"
            values = val.values[0]

            if "}, {" in values:
                events = values.split("}, {")
            else:
                events = [values]

            for event in events: # parse strings
                event = event.replace("'", "")
                event = (event.replace("{", "")).replace("}", "")
                event = (event.replace("[", "")).replace("]", "")
                event_descriptors = event.split(", ")
                start_time = event_descriptors[0]
                value = event_descriptors[1]

                time_in_hours = int(int(eval(start_time.split(": ")[1])) / 3600)
                minutes = int((int(eval(start_time.split(": ")[1]) % 3600) / 3600) * 60)
                if schedule_type + "start_times" in dict_:
                    dict_[schedule_type + "start_times"].append(datetime.time(hour=time_in_hours, minute=minutes))
                    dict_[schedule_type + "values"].append(float(value.split(": ")[1]))
                else:
                    dict_[schedule_type + "start_times"] = [datetime.time(hour=time_in_hours, minute=minutes)]
                    dict_[schedule_type + "values"] = [float(value.split(": ")[1])]

    return dict_
