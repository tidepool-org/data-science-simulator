__author__ = "Eden Grown-Haeberli"

import pandas as pd
import numpy as np
import datetime
import copy
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.makedata.scenario_parser import start_times_to_minutes_durations
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, TempBasalTimeline
from tidepool_data_science_simulator.models.measures import (
    Carb,
    BasalRate,
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
        report_num = int(file_name.split(sep="/")[-1].split(sep="-")[3])
        settings_data = (issue_report_settings.loc[
                        issue_report_settings['loop_id'] == user_numbers])
        settings_data = settings_data.loc[settings_data['report_num'] == report_num]

        # TODO: Does this take up unnecessary runtime?
        carb_data = pd.read_csv(path_to_time_series_data, sep=",", usecols=['rounded_local_time', 'carbs'])
        target_data = pd.read_csv(path_to_time_series_data, sep=",", usecols=['rounded_local_time',
                                                                              'bg_target_lower', 'bg_target_upper'])
        cgm_data = pd.read_csv(path_to_time_series_data, sep=",", usecols=['rounded_local_time', 'cgm'])
        self.parse_settings(settings_df=settings_data, target_df=target_data)
        self.parse_carbs(carb_df=carb_data)

        self.parse_cgm(cgm_data)

        if t0 is not None:
            time = t0
        else:
            time = self.get_simulation_start_time()

        self.transform_pump(time)
        self.transform_patient(time)
        self.transform_sensor()

    # Note: rename this and parse_settings to more appropriate names, refactor
    def parse_settings(self, settings_df, target_df, start_time=None):
        jaeb_inputs = parse_settings(settings_df=settings_df)

        if start_time is not None:
            jaeb_inputs['time_to_calculate_at'] = start_time
        else:
            issue_report_start = datetime.datetime.fromisoformat(
                pd.to_datetime(target_df['rounded_local_time'].values[0]).isoformat())
            rounded_start_time = issue_report_start + datetime.timedelta(days=1)
            rounded_start_time += datetime.timedelta(minutes=2, seconds=30)
            rounded_start_time -= datetime.timedelta(minutes=rounded_start_time.minute % 5,
                                                     seconds=rounded_start_time.second,
                                                     microseconds=rounded_start_time.microsecond)

            jaeb_inputs['time_to_calculate_at'] = rounded_start_time

        target_data = target_df.dropna()
        jaeb_inputs['target_range_minimum_values'] = list(target_data['bg_target_lower'].drop_duplicates().values)
        jaeb_inputs['target_range_maximum_values'] = list(target_data['bg_target_upper'].drop_duplicates().values)
        jaeb_inputs['target_range_start_times'] = [datetime.time(0, 0, 0)]

        # add any other important values
        # TODO: Make this flexible
        jaeb_inputs["basal_rate_units"] = ["U" for _ in jaeb_inputs["basal_rate_values"]]
        jaeb_inputs["carb_ratio_value_units"] = [jaeb_inputs["carb_value_units"] + "/U" for _ in jaeb_inputs[
            "carb_ratio_values"]]
        jaeb_inputs["target_range_value_units"] = ["mg/dL" for _ in jaeb_inputs["target_range_minimum_values"]]
        jaeb_inputs["sensitivity_ratio_value_units"] = ["mg/dL/U" for _ in jaeb_inputs["sensitivity_ratio_values"]]
       # fixme: in the future, pull duration values from the data

        self.loop_inputs_dict = jaeb_inputs

    def parse_carbs(self, carb_df):
        carb_data = carb_df.dropna()
        for col, _ in carb_df.items():
            temp_df = carb_data[col]
            temp_array = []
            if col == 'rounded_local_time':
                for v in temp_df.values:
                    if ":" in v:
                        if len(v) == 7:
                            obj = datetime.time.fromisoformat(
                                pd.to_datetime(v).strftime("%H:%M:%S")
                            )
                        elif len(v) == 8:
                            obj = datetime.time.fromisoformat(v)
                        elif len(v) > 8:
                            obj = datetime.datetime.fromisoformat(
                                pd.to_datetime(v).isoformat()
                            )
                        else:
                            obj = np.safe_eval(v)
                    else:
                        obj = np.safe_eval(v)

                    temp_array = np.append(temp_array, obj)

                self.loop_inputs_dict['carb_dates'] = list(temp_array)
            elif col == 'carbs':
                self.loop_inputs_dict['carb_values'] = list(temp_df.values)

        self.loop_inputs_dict["carb_value_units"] = [self.loop_inputs_dict["carb_value_units"] for _ in self.loop_inputs_dict[
            'carb_values']]
        self.loop_inputs_dict["carb_absorption_times"] = [self.loop_inputs_dict["settings_dictionary"][
                                                              "default_absorption_times"][1] for _ in self.loop_inputs_dict[
            'carb_values']]

    def parse_cgm(self, cgm_df):
        cgm_data = cgm_df.dropna()
        history_end_time = self.get_simulation_start_time()

        bg_dates = [datetime.datetime.fromisoformat(
            pd.to_datetime(date).isoformat()) for date in cgm_data["rounded_local_time"].values if
            datetime.datetime.fromisoformat(date) <= history_end_time]
        glucose_values = [cgm_data["cgm"].values[i] for i in range(
            len(cgm_data["cgm"].values)) if datetime.datetime.fromisoformat(cgm_data["rounded_local_time"].values[i]) <=
            history_end_time]

        self.loop_inputs_dict["cgm_glucose_values"] = glucose_values
        self.loop_inputs_dict["cgm_glucose_dates"] = bg_dates

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

        # TODO: Import bolus and basal data

        self.pump_bolus_events = BolusTimeline(
            datetimes=[],
            events=[]
        )

        self.pump_temp_basal_events = TempBasalTimeline(
            datetimes=[],
            events=[]
        )

    def transform_patient(self, time):
        """
        Loop issue report gives no true patient information.

        What do we want to do here for replay?
        """
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
            duration_minutes=self.loop_inputs_dict.get(
                "basal_rate_minutes", start_times_to_minutes_durations(self.loop_inputs_dict["basal_rate_start_times"])
            )
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
            events=[]
        )

        self.patient_glucose_history = GlucoseTrace(
            datetimes=self.loop_inputs_dict["cgm_glucose_dates"],
            values=self.loop_inputs_dict["cgm_glucose_values"],
        )

    def transform_sensor(self):

        self.sensor_glucose_history = GlucoseTrace(
            datetimes=self.loop_inputs_dict["cgm_glucose_dates"],
            values=self.loop_inputs_dict["cgm_glucose_values"],
        )


def parse_settings(settings_df):
    dict_ = dict()
    dict_["settings_dictionary"] = dict()
    for col, val in settings_df.iteritems():
        if col == "insulin_model":
            model = val.values[0]
            obj = []
            if model == "fiasp":
                obj = [360, 55]
            elif "humalog" in model:
                if "child" in model:
                    obj = [360, 65]
                elif "adult" in model:
                    obj = [360, 75]
            dict_["settings_dictionary"]["model"] = obj
        elif col == "carb_ratio_unit":
            dict_["carb_value_units"] = val.values[0]
        elif "maximum_" in col:
            col = col.replace("maximum", "max")
            dict_["settings_dictionary"][col] = val.values[0]
        elif "suspend_threshold" == col or "retrospective_correction_enabled" == col:
            dict_["settings_dictionary"][col] = val.values[0]
        elif "carb_default_absorption" in col:
            if "default_absorption_times" not in dict_["settings_dictionary"]:
                dict_["settings_dictionary"]["default_absorption_times"] = [0, 0, 0]
            value = float(val.values[0])
            if "slow" in col:
                dict_["settings_dictionary"]["default_absorption_times"][2] = value / 60 # convert to minutes
            elif "medium" in col:
                dict_["settings_dictionary"]["default_absorption_times"][1] = value / 60
            elif "fast" in col:
                dict_["settings_dictionary"]["default_absorption_times"][0] = value / 60
        elif col.endswith("_schedule"):
            schedule_type = col.replace("schedule", "")
            if "sensitivity" in col:
                schedule_type = "sensitivity_ratio_"
            values = val.values[0]
            values = values.replace("'", "")
            values = (values.replace("[", "")).replace("]", "")

            if "}, {" in values:
                events = values.split("}, {")
            else:
                events = [values]

            for event in events: # parse strings
                event = (event.replace("{", "")).replace("}", "")
                event_descriptors = event.split(", ")
                start_time = ""
                value = ""
                for descriptor in event_descriptors:
                    if 'start' in descriptor:
                        start_time = descriptor
                    elif 'value' in descriptor:
                        value = descriptor

                time_in_hours = int(int(eval(start_time.split(": ")[1])) / 3600)
                minutes = int((int(eval(start_time.split(": ")[1]) % 3600) / 3600) * 60)
                if schedule_type + "start_times" in dict_:
                    dict_[schedule_type + "start_times"].append(datetime.time(hour=time_in_hours, minute=minutes))
                    dict_[schedule_type + "values"].append(float(value.split(": ")[1]))
                else:
                    dict_[schedule_type + "start_times"] = [datetime.time(hour=time_in_hours, minute=minutes)]
                    dict_[schedule_type + "values"] = [float(value.split(": ")[1])]

    return dict_
