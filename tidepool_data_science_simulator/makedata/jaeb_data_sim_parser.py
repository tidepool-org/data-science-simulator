__author__ = "Eden Grown-Haeberli"

import pandas as pd
import numpy as np
import datetime
import copy
import pytz

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV, PatientConfig, PumpConfig
from pyloopkit.dose import DoseType
from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, TargetRangeSchedule24hr, BasalSchedule24hr
)
from tidepool_data_science_simulator.makedata.scenario_parser import start_times_to_minutes_durations
from tidepool_data_science_simulator.models.events import (
    CarbTimeline,
    BolusTimeline,
    TempBasalTimeline,
    ActionTimeline,
)
from tidepool_data_science_simulator.models.measures import (
    Carb,
    BasalRate,
    Bolus,
    TempBasal,
    CarbInsulinRatio,
    InsulinSensitivityFactor,
    TargetRange,
    GlucoseTrace,
)

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


class JaebDataSimParser(ScenarioParserCSV):
    """
    A scenario parser meant to pull data from the parsed Jaeb data files.
    """

    def __init__(self, path_to_settings, path_to_time_series_data, t0=None, days_in=None):
        issue_report_settings = pd.read_csv(path_to_settings, sep=",")

        file_name = path_to_time_series_data.split(sep="/")[-1]
        user_numbers = "LOOP-" + file_name.split(sep="-")[1]
        report_num = int(file_name.split(sep="/")[-1].split(sep="-")[3])
        settings_data = (issue_report_settings.loc[
                        issue_report_settings['loop_id'] == user_numbers])
        settings_data = settings_data.loc[settings_data['report_num'] == report_num]

        self.loop_version = settings_data['loop_version']

        time_series_df = pd.read_csv(path_to_time_series_data, sep=",")

        self.parse_settings(
            settings_df=settings_data,
            target_df=time_series_df[['rounded_local_time', 'bg_target_lower', 'bg_target_upper']],
            start_time=t0,
            days_in=days_in
        )
        self.parse_carbs(
            carb_df=time_series_df[['rounded_local_time', 'carbs']]
        )
        self.parse_insulin(
            insulin_df=time_series_df[[
                'rounded_local_time',
                'bolus',
                'set_basal_rate',
                'total_insulin_delivered',
                'basal_pulse_delivered'
            ]]
        )
        self.parse_cgm(
            cgm_df=time_series_df[['rounded_local_time', 'cgm']]
        )

        if t0 is not None:
            time = t0
        else:
            time = self.get_simulation_start_time()

        self.transform_pump(time)
        self.transform_patient(time)
        self.transform_sensor()

        #TODO: Integrate into patient
        self.patient_id = user_numbers
        self.report_num = report_num

    # Note: rename this and parse_settings to more appropriate names, refactor
    def parse_settings(self, settings_df, target_df, start_time=None, days_in=None):
        jaeb_inputs = self.input_settings_from_df(settings_df=settings_df)

        tr_sched = parse_tr_schedule_from_time_series(target_df)
        tr_sched_normal = tr_sched[list(tr_sched.keys())[0]]
        jaeb_inputs['target_range_start_times'] = tr_sched_normal[0]
        jaeb_inputs['target_range_minimum_values'] = tr_sched_normal[1]
        jaeb_inputs['target_range_maximum_values'] = tr_sched_normal[2]

        if days_in is None:
            days_in = 7

        if start_time is not None:
            jaeb_inputs['time_to_calculate_at'] = start_time
        else:
            issue_report_start = datetime.datetime.fromisoformat(
                pd.to_datetime(target_df['rounded_local_time'].values[0]).isoformat())
            rounded_start_time = issue_report_start + datetime.timedelta(days=days_in)
            rounded_start_time += datetime.timedelta(minutes=2, seconds=30)
            rounded_start_time -= datetime.timedelta(minutes=rounded_start_time.minute % 5,
                                                     seconds=rounded_start_time.second,
                                                     microseconds=rounded_start_time.microsecond)

            jaeb_inputs['time_to_calculate_at'] = rounded_start_time

        # fixme: in the future, pull duration values from the data
        jaeb_inputs["basal_rate_units"] = ["U" for _ in jaeb_inputs["basal_rate_values"]]
        jaeb_inputs["carb_ratio_value_units"] = [jaeb_inputs["carb_value_units"] + "/U" for _ in jaeb_inputs[
            "carb_ratio_values"]]
        jaeb_inputs["target_range_value_units"] = ["mg/dL" for _ in jaeb_inputs["target_range_minimum_values"]]
        jaeb_inputs["sensitivity_ratio_value_units"] = ["mg/dL/U" for _ in jaeb_inputs["sensitivity_ratio_values"]]
        jaeb_inputs["settings_dictionary"]["dynamic_carb_absorption_enabled"] = True

        self.loop_inputs_dict = jaeb_inputs

    def input_settings_from_df(self, settings_df):
        """
        Gets settings from dataframe.
        """
        dict_ = dict()
        dict_["settings_dictionary"] = dict()
        for col, val in settings_df.iteritems():
            if col == "insulin_model":
                model = val.values[0]
                obj = []
                if model == "fiasp":
                    obj = [360, 55]
                elif "humalog" in model:
                    if "Child" in model:
                        obj = [360, 65]
                    elif "Adult" in model:
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
                    dict_["settings_dictionary"]["default_absorption_times"][2] = value / 60  # convert to minutes
                elif "medium" in col:
                    dict_["settings_dictionary"]["default_absorption_times"][1] = value / 60
                elif "fast" in col:
                    dict_["settings_dictionary"]["default_absorption_times"][0] = value / 60
            elif col.endswith("_schedule"):
                schedule_type = col.replace("schedule", "")
                if "sensitivity" in col:
                    schedule_type = "sensitivity_ratio_"
                elif "correction" in col:
                    continue
                values = val.values[0]
                values = values.replace("'", "")
                values = (values.replace("[", "")).replace("]", "")

                if "}, {" in values:
                    events = values.split("}, {")
                else:
                    events = [values]

                for event in events:  # parse strings
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

    def parse_carbs(self, carb_df):
        """
        Get the carb start_times, carb_values, units and absorption times from the time series data.
        """
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
        """
        Get glucose values up to the
        """
        cgm_data = cgm_df.dropna()
        history_end_time = self.get_simulation_start_time()

        self.loop_inputs_dict["glucose_values"] = [cgm_data["cgm"].values[i] for i in range(
            len(cgm_data["cgm"].values)) if datetime.datetime.fromisoformat(cgm_data["rounded_local_time"].values[i]) <=
            history_end_time]
        self.loop_inputs_dict["glucose_dates"] = [datetime.datetime.fromisoformat(
            pd.to_datetime(date).isoformat()) for date in cgm_data["rounded_local_time"].values if
            datetime.datetime.fromisoformat(date) <= history_end_time]

        self.loop_inputs_dict["full_glucose_values"] = [val for val in cgm_data["cgm"].values]
        self.loop_inputs_dict["full_glucose_dates"] = [datetime.datetime.fromisoformat(
            pd.to_datetime(date).isoformat()) for date in cgm_data["rounded_local_time"].values]

    def parse_insulin(self, insulin_df):
        insulin_df = insulin_df.set_index('rounded_local_time')
        temp_basal = insulin_df['set_basal_rate'].dropna()
        bolus = insulin_df['bolus'].dropna()


        bolus_start_times, bolus_values, bolus_units, bolus_dose_types, bolus_delivered_units = ([], [], [], [], [])
        temp_basal_start_times, temp_basal_duration_minutes, temp_basal_values, \
        temp_basal_units, temp_basal_dose_types, temp_basal_delivered_units = ([], [], [], [], [], [])
        for date, dose in bolus.items():
            bolus_date = datetime.datetime.fromisoformat(pd.to_datetime(date).isoformat())
            bolus_start_times.append(bolus_date)
            bolus_values.append(dose)
            bolus_dose_types.append(DoseType.bolus)
            bolus_units.append("U")
            bolus_delivered_units.append(insulin_df[date]['total_insulin_delivered']-insulin_df[date][
                'basal_pulse_delivered'])

        previous = None
        for date, dose in temp_basal.items():
            date_as_date = datetime.datetime.fromisoformat(pd.to_datetime(date).isoformat())
            if date_as_date > self.get_simulation_start_time():
                break
            temp_basal_start_times.append(date_as_date)

            if not previous:
                temp_basal_values.append(dose)
                temp_basal_units.append("U/hr")
                temp_basal_delivered_units.append(insulin_df[date]['basal_pulse_delivered'])
                temp_basal_dose_types.append(DoseType.tempbasal)
                temp_basal_duration_minutes.append(5)
            else:
                if (previous[1] == dose) and (date_as_date - previous[0] == datetime.timedelta(minutes=5)):
                    temp_basal_duration_minutes[-1] = temp_basal_duration_minutes[-1] + 5
                    temp_basal_delivered_units[-1] = temp_basal_delivered_units + (insulin_df[date]['basal_pulse_delivered'])
                else:
                    temp_basal_values.append(dose)
                    temp_basal_units.append("U/hr")
                    temp_basal_delivered_units.append(insulin_df[date]['basal_pulse_delivered'])
                    temp_basal_dose_types.append(DoseType.tempbasal)
                    temp_basal_duration_minutes.append(5)

            previous = date_as_date, dose

        self.loop_inputs_dict['bolus_start_times'] = bolus_start_times
        self.loop_inputs_dict['bolus_values'] = bolus_values
        self.loop_inputs_dict['bolus_units'] = bolus_units
        self.loop_inputs_dict['bolus_delivered_units'] = bolus_delivered_units
        self.loop_inputs_dict['bolus_dose_types'] = bolus_dose_types
        self.loop_inputs_dict['temp_basal_start_times'] = temp_basal_start_times
        self.loop_inputs_dict['temp_basal_values'] = temp_basal_values
        self.loop_inputs_dict['temp_basal_duration_minutes'] = temp_basal_duration_minutes
        self.loop_inputs_dict['temp_basal_dose_types'] = temp_basal_dose_types
        self.loop_inputs_dict['temp_basal_units'] = temp_basal_units
        self.loop_inputs_dict['temp_basal_delivered_units'] = temp_basal_delivered_units


    def transform_pump(self, time):

        # ========== Pump =============
        self.transform_pump_settings(time)

        self.pump_bolus_events = BolusTimeline(
            datetimes=self.loop_inputs_dict["bolus_start_times"],
            events=[
                Bolus(value, units)
                for value, units, dose_type in zip(
                    self.loop_inputs_dict["bolus_values"],
                    self.loop_inputs_dict["bolus_units"],
                    self.loop_inputs_dict["bolus_dose_types"]
                )
            ],
        )

        self.pump_temp_basal_events = TempBasalTimeline(
            datetimes=self.loop_inputs_dict["temp_basal_start_times"],
            events=[
                TempBasal(start_time, value, duration_minutes, units, delivered_units)
                for start_time, value, units, duration_minutes, delivered_units in zip(
                    self.loop_inputs_dict["temp_basal_start_times"],
                    self.loop_inputs_dict["temp_basal_values"],
                    self.loop_inputs_dict["temp_basal_units"],
                    self.loop_inputs_dict["temp_basal_duration_minutes"],
                    self.loop_inputs_dict["temp_basal_delivered_units"]
                )
            ],
        )

        self.pump_max_basal_rate = self.loop_inputs_dict['settings_dictionary']['max_basal_rate']
        self.pump_max_bolus = self.loop_inputs_dict['settings_dictionary']['max_bolus']

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
            datetimes=self.loop_inputs_dict["bolus_start_times"],
            events=[
                Bolus(value, units)
                for value, units, dose_type in zip(
                    self.loop_inputs_dict["bolus_values"],
                    self.loop_inputs_dict["bolus_units"],
                    self.loop_inputs_dict["bolus_dose_types"]
                )
            ],
        )

        self.patient_glucose_history = GlucoseTrace(
            datetimes=self.loop_inputs_dict["glucose_dates"],
            values=self.loop_inputs_dict["glucose_values"],
        )

        self.patient_full_glucose_values = GlucoseTrace(
            datetimes=self.loop_inputs_dict["full_glucose_dates"],
            values=self.loop_inputs_dict["full_glucose_values"]
        )

        self.patient_actions = self.get_action_timeline()

    def get_action_timeline(self):
        """
        Imports actions from time series data, i.e. suspend, settings change, etc.
        fixme: these settings don't change, update them
        """
        datetimes = []
        events = []

        timeline = ActionTimeline(datetimes=datetimes, events=events)
        return timeline

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
            temp_basal_event_timeline=self.pump_temp_basal_events,
            max_temp_basal=self.pump_max_basal_rate,
            max_bolus=self.pump_max_bolus
        )

        return pump_config

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
            action_timeline=self.patient_actions
        )

        return patient_config


def parse_tr_schedule_from_time_series(schedule_df):
    """
    Parse target schedule from the time series.
    """
    target_range_schedule = {}

    schedule_df = schedule_df.drop_duplicates()
    for i in schedule_df.index:
        date_and_time = datetime.datetime.fromisoformat(schedule_df['rounded_local_time'][i])
        date = date_and_time.date()
        if date in target_range_schedule:
            if target_range_schedule[date][1][-1] != schedule_df['bg_target_lower'][i] or \
                    target_range_schedule[date][2][-1] != schedule_df['bg_target_upper'][i]:
                date_and_time = datetime.datetime.fromisoformat(schedule_df['rounded_local_time'][i])
                time = date_and_time.time()
                target_range_schedule[date][0].append(time)
                target_range_schedule[date][1].append(schedule_df['bg_target_lower'][i])
                target_range_schedule[date][2].append(schedule_df['bg_target_upper'][i])
        else:
            time = date_and_time.time()
            if time == datetime.time(0, 0, 0):
                target_range_schedule[date] = [
                    [time],
                    [schedule_df['bg_target_lower'][i]],
                    [schedule_df['bg_target_upper'][i]]]

    return target_range_schedule


class JaebReplayParser(JaebDataSimParser):
    def transform_patient(self, time):
        """
        Loop issue report gives no true patient information.

        """
        self.patient_basal_schedule = None
        self.patient_carb_ratio_schedule = None
        self.patient_insulin_sensitivity_schedule = None
        self.patient_target_range_schedule = None

        self.patient_carb_events = None
        self.patient_bolus_events = None
        self.patient_glucose_history = None

        self.patient_actions = None

    def transform_sensor(self):
        """
        Captures full patient glucose history as sensor glucose history
        """

        self.sensor_glucose_history = GlucoseTrace(
            datetimes=self.loop_inputs_dict["full_glucose_dates"],
            values=self.loop_inputs_dict["full_glucose_values"],
        )

        assert self.sensor_glucose_history.get_last()[0] >= self.get_simulation_start_time()

    def parse_insulin(self, insulin_df):
        insulin_df['basal_end_dates'] = insulin_df['rounded_local_time'].apply(
            lambda x: (datetime.datetime.fromisoformat(pd.to_datetime(x).isoformat()) + datetime.timedelta(
                minutes=5)).isoformat()
        )
        insulin_df = insulin_df.set_index('rounded_local_time')
        temp_basal = insulin_df[['set_basal_rate', 'basal_pulse_delivered', 'basal_end_dates']].dropna()
        bolus = insulin_df['bolus'].dropna()

        bolus_start_times, bolus_values, bolus_units, bolus_dose_types, bolus_delivered_units = ([], [], [], [], [])
        temp_basal_start_times, temp_basal_duration_minutes, temp_basal_values, \
        temp_basal_units, temp_basal_dose_types, temp_basal_delivered_units = ([], [], [], [], [], [])
        for date, dose in bolus.items():
            bolus_date = datetime.datetime.fromisoformat(pd.to_datetime(date).isoformat())
            bolus_start_times.append(bolus_date)
            bolus_values.append(dose)
            bolus_dose_types.append(DoseType.bolus)
            bolus_units.append("U")
            bolus_delivered_units.append(insulin_df.at[date, 'total_insulin_delivered'] - insulin_df.at[date,
                'basal_pulse_delivered'])

        previous = None
        for date, dose in temp_basal.iterrows():
            date_as_date = datetime.datetime.fromisoformat(pd.to_datetime(date).isoformat())
            if not previous:
                temp_basal_start_times.append(date_as_date)
                temp_basal_values.append(dose['set_basal_rate'])
                temp_basal_units.append("U/hr")
                temp_basal_delivered_units.append(dose['basal_pulse_delivered'])
                temp_basal_dose_types.append(DoseType.tempbasal)
                temp_basal_duration_minutes.append(5)
                current_date = copy.deepcopy(date_as_date) + datetime.timedelta(minutes=5)
                last_date = datetime.datetime.fromisoformat(pd.to_datetime(insulin_df.index[-1]).isoformat())
                if current_date < last_date:
                    while insulin_df.at[current_date.isoformat().replace("T", " "), 'basal_pulse_delivered'] == dose['basal_pulse_delivered']:
                        pulse = dose['basal_pulse_delivered']
                        new_pulse = insulin_df.at[current_date.isoformat().replace("T", " "),'basal_pulse_delivered']
                        temp_basal_duration_minutes[-1] = temp_basal_duration_minutes[-1] + 5
                        temp_basal_delivered_units[-1] = temp_basal_delivered_units[-1] + dose['basal_pulse_delivered']
                        current_date = current_date + datetime.timedelta(minutes=5)
                        if current_date >= last_date:
                            break
            else:
                if (previous[1]['set_basal_rate'] == dose['set_basal_rate']) and (date_as_date - previous[0] ==
                                                                                  datetime.timedelta(
                                                                                      minutes=5)):
                    pass
                else:
                    temp_basal_values.append(dose['set_basal_rate'])
                    temp_basal_units.append("U/hr")
                    temp_basal_start_times.append(date_as_date)
                    temp_basal_delivered_units.append(dose['basal_pulse_delivered'])
                    temp_basal_dose_types.append(DoseType.tempbasal)
                    temp_basal_duration_minutes.append(5)
                    current_date = copy.deepcopy(date_as_date) + datetime.timedelta(minutes=5)
                    if current_date < last_date:
                        while insulin_df.at[current_date.isoformat().replace("T", " "), 'basal_pulse_delivered'] == dose['basal_pulse_delivered']:
                            temp_basal_duration_minutes[-1] = temp_basal_duration_minutes[-1] + 5
                            temp_basal_delivered_units[-1] = temp_basal_delivered_units[-1] + dose['basal_pulse_delivered']
                            current_date = current_date + datetime.timedelta(minutes=5)
                            if current_date >= last_date:
                                break

            previous = date_as_date, dose

        self.loop_inputs_dict['bolus_start_times'] = bolus_start_times
        self.loop_inputs_dict['bolus_values'] = bolus_values
        self.loop_inputs_dict['bolus_units'] = bolus_units
        self.loop_inputs_dict['bolus_delivered_units'] = bolus_delivered_units
        self.loop_inputs_dict['bolus_dose_types'] = bolus_dose_types
        self.loop_inputs_dict['temp_basal_start_times'] = temp_basal_start_times
        self.loop_inputs_dict['temp_basal_values'] = temp_basal_values
        self.loop_inputs_dict['temp_basal_duration_minutes'] = temp_basal_duration_minutes
        self.loop_inputs_dict['temp_basal_dose_types'] = temp_basal_dose_types
        self.loop_inputs_dict['temp_basal_units'] = temp_basal_units
        self.loop_inputs_dict['temp_basal_delivered_units'] = temp_basal_delivered_units


class RawDatasetJaebParser(JaebReplayParser):
    def __init__(self, data_path, settings_data, ir_number=0):
        """
        Initialize raw dataset parser object.

        Parameters
        ----------
        data_path: String
        settings_data: String
        ir_number: Int
        """
        user_data_df = pd.read_csv(data_path, sep="\t", compression="gzip", low_memory=False, index_col=[0])
        self.loop_version = settings_data['loop_version']

        carb_data_df = user_data_df[['time', 'nutrition.carbohydrate.net', 'nutrition.carbohydrate.units',
                                  'payload.com.loudnate.CarbKit.HKMetadataKey.AbsorptionTimeMinutes', 'type']]
        carb_data_df = carb_data_df[carb_data_df['nutrition.carbohydrate.net'].notna()]
        bg_types = ["smbg", "cbg"]
        cgm_data_df = user_data_df[['time', 'value', 'type', 'units']]
        cgm_data_df = cgm_data_df[cgm_data_df['type'].isin(bg_types)]
        insulin_types = ["bolus", "basal"]
        insulin_data_df = user_data_df[user_data_df['type'].isin(insulin_types)]
        insulin_data_df = insulin_data_df[['time', 'value', 'type', 'units', 'deliveryType', "duration", 'rate',
                                        'payload.com.loopkit.InsulinKit.MetadataKeyProgrammedTempBasalRate',
                                        'payload.com.loopkit.InsulinKit.MetadataKeyScheduledBasalRate']]

        issue_report_dates = settings_data['issue_report_date'].unique()
        data_around_issue_report = {}
        for ir_date in issue_report_dates:
            data_start = (pd.to_datetime(ir_date) - datetime.timedelta(hours=8)).strftime('%Y-%m-%dT%H:%M:%d')
            data_end = (pd.to_datetime(ir_date) + datetime.timedelta(days=1)).strftime('%Y-%m-%dT%H:%M:%d')
            has_cgm_df = (
                user_data_df[((user_data_df["time"] >= data_start) & (user_data_df["time"] < data_end))]
                    .copy()
                    .dropna(axis=1, how="all")
                    .sort_values(["time"])
                    .reset_index()
            )

            local_timezone = self.get_timezone_from_settings(has_cgm_df)
            insulin_carb_5min_ts = self.process_daily_insulin_and_carb_data(has_cgm_df, data_start, local_timezone)
            data_around_issue_report[ir_date] = insulin_carb_5min_ts

        all_time = list(data_around_issue_report.keys())
        t0 = all_time[ir_number]
        self.parse_settings(settings_df=settings_data[settings_data["date"].isin(issue_report_dates)], start_time=t0)
        self.parse_carbs(carb_df=carb_data_df)
        self.parse_insulin(insulin_df=insulin_data_df)
        self.parse_cgm(cgm_data_df)

        if t0 is not None:
            time = t0
        else:
            time = self.get_simulation_start_time()

        self.transform_pump(time)
        self.transform_patient(time)
        self.transform_sensor()

        #TODO: Integrate into patient
        # self.patient_id = user_numbers
        # self.report_num = report_num

    def get_timezone_from_settings(self, data):
        """
        The local time is needed to match up setting schedules to the data.
        To make this conversion, the local timezone is needed and often provided by the issue report.
        Pulled from the data-science-explore-jaeb-data repo
        Parameters
        ----------
        data : pandas.DataFrame
            The entire Jaeb dataset from the study participant.
            The timezoneOffset column is used in case the issue report does not have a timezone set

        Returns
        -------
        local_timezone : str
            The name of the local timezone name calculated from the issue report's timezone offset

        """
        if "timezoneOffset" in data.columns:
            most_common_data_offset = data["timezoneOffset"].mode()[0]
            utc_offset = datetime.timedelta(minutes=most_common_data_offset)
        else:
            # print("No timezone could be calculated. Defaulting to GMT-6")
            utc_offset = datetime.timedelta(minutes=360)

        local_timezone = [tz for tz in pytz.all_timezones if utc_offset == pytz.timezone(tz)._utcoffset][0]
        # print("{} -- {} timezone calculated: {}".format(loop_id, single_report["file_name"], local_timezone))

        return local_timezone

    def process_daily_insulin_and_carb_data(self, insulin_report, sample_start_time, local_timezone):
        """
        All bolus, basal, and carb information must be extracted, deduplicated, and merged into a common time series
        Pulled from the data-science-explore-jaeb-data repo

        Parameters
        ----------
        insulin_report: pandas.DataFrame
        sample_start_time : datetime
        local_timezone : str


        Returns
        -------
        insulin_carb_5min_ts : pandas.DataFrame

        """
        nonrounded_5min_ts = pd.DataFrame(insulin_report['time'], columns=['time'])
        nonrounded_5min_ts = self.process_basal_data(insulin_report, nonrounded_5min_ts)
        nonrounded_5min_ts = self.process_bolus_data(insulin_report, nonrounded_5min_ts)
        nonrounded_5min_ts = self.process_carb_data(insulin_report, nonrounded_5min_ts)
        nonrounded_5min_ts["total_insulin_delivered"] = nonrounded_5min_ts["basal_pulse_delivered"] + nonrounded_5min_ts[
            "normal"
        ].fillna(0)

        # Get insulin on board for all basal/bolus data
        nonrounded_5min_ts = self.calculate_iob_for_timeseries(nonrounded_5min_ts)

        local_start_time = pd.to_datetime(sample_start_time).tz_localize(local_timezone)
        insulin_carb_5min_ts = nonrounded_5min_ts[(pd.to_datetime(nonrounded_5min_ts["time"])) >=
                                                   local_start_time].copy()
        insulin_carb_5min_ts["date"] = pd.to_datetime(insulin_carb_5min_ts["time"]).dt.date

        return insulin_carb_5min_ts

    def calculate_iob_for_timeseries(self, daily_5min_ts):
        """
        Insulin-on-board is an important calculation for assessing DKA risk and other metrics.
        The Simple Diabetes Metabolism Model can be used to add all the iob effects of evey insulin delivery pulse
        within a time series.
        Pulled from the data-science-explore-jaeb-data repo

        Parameters
        ----------
        daily_5min_ts : pandas.DataFrame
            A 5-min rounded time series containing the combined bolus and basal insulin delivered

        Returns
        -------
        daily_5min_ts : pandas.DataFrame
            The same dataframe, now with the "iob" data at every time step

        """
        smm = SimpleMetabolismModel(insulin_sensitivity_factor=1, carb_insulin_ratio=1)
        _, _, _, insulin_decay_vector = smm.run(carb_amount=0, insulin_amount=1)
        all_iob_arrays = daily_5min_ts["total_insulin_delivered"].apply(lambda x: x * insulin_decay_vector)
        decay_size = len(insulin_decay_vector)
        final_iob_array = [0] * len(all_iob_arrays) + [0] * decay_size

        for i in range(len(all_iob_arrays)):
            final_iob_array[i: (i + decay_size)] += all_iob_arrays[i]

        daily_5min_ts["iob"] = final_iob_array[:-decay_size]

        return daily_5min_ts

    def process_basal_data(self, buffered_sample_data, daily_5min_ts):
        """
        The basal data must be extracted, deduplicated, and merged into a 5-minute rounded local time series
        Pulled from the data-science-explore-jaeb-data repo

        Parameters
        ----------
        buffered_sample_data : pandas.DataFrame
        daily_5min_ts: pandas.DataFrame

        Returns
        -------
        daily_5min_ts : pandas.DataFrame

        """
        basal_data = buffered_sample_data[buffered_sample_data["type"] == "basal"].copy()
        basals_before_deduplication = len(basal_data)

        if basals_before_deduplication > 0:
            basal_data.sort_values(by=["time", "uploadId"], ascending=False, inplace=True)
            basal_data = basal_data[~basal_data["time"].duplicated()].sort_values("time", ascending=True)

            basal_rates = basal_data[["time", "rate"]]

            daily_5min_ts = pd.merge(daily_5min_ts, basal_rates, how="left", on="time")

            daily_5min_ts["basal_pulse_delivered"] = daily_5min_ts["rate"] / 12
            daily_5min_ts["basal_pulse_delivered"].ffill(limit=288, inplace=True)

        else:
            daily_5min_ts["basal_pulse_delivered"] = np.nan

        return daily_5min_ts

    def process_bolus_data(self, buffered_sample_data, daily_5min_ts):
        """
        The bolus data must be extracted, deduplicated, and merged into a 5-minute rounded local time series
        Pulled from the data-science-explore-jaeb-data repo

        Parameters
        ----------
        buffered_sample_data : pandas.DataFrame
        daily_5min_ts: pandas.DataFrame

        Returns
        -------
        single_report : pandas.Series
        daily_5min_ts : pandas.DataFrame

        """
        bolus_data = buffered_sample_data[buffered_sample_data["type"] == "bolus"].copy()
        boluses_before_deduplication = len(bolus_data)

        if boluses_before_deduplication > 0:
            bolus_data.sort_values(by=["time", "uploadId"], ascending=False, inplace=True)
            bolus_data = bolus_data[~bolus_data["time"].duplicated()].sort_values("time", ascending=True)

            # Merge boluses within 5-min together
            bolus_data = pd.DataFrame(bolus_data.groupby("time")["normal"].sum()).reset_index()
            boluses = bolus_data[["time", "normal"]]
            daily_5min_ts = pd.merge(daily_5min_ts, boluses, how="left", on="time")
        else:
            daily_5min_ts["normal"] = np.nan

        return daily_5min_ts

    def process_carb_data(self, buffered_sample_data, daily_5min_ts):
        """
        The carb data must be extracted, deduplicated, and merged into a 5-minute rounded local time series

        Parameters
        ----------
        single_report : pandas.Series
        buffered_sample_data : pandas.DataFrame
        sample_start_time : datetime

        Returns
        -------
        single_report : pandas.Series
        daily_5min_ts : pandas.DataFrame

        """
        carb_data = buffered_sample_data[buffered_sample_data["type"] == "food"].copy()
        carb_entries_before_deduplication = len(carb_data)

        # Carbs may come from two difference sources: nutrition.carbohydrate(s), combine into one
        carb_data["carbs"] = 0
        if "nutrition.carbohydrates.net" in carb_data.columns:
            carb_data["carbs"] += carb_data["nutrition.carbohydrates.net"]

        if "nutrition.carbohydrate.net" in carb_data.columns:
            carb_data["carbs"] += carb_data["nutrition.carbohydrate.net"]

        if carb_entries_before_deduplication > 0:
            carb_data.sort_values(by=["time", "uploadId"], ascending=False, inplace=True)
            carb_data = carb_data[~carb_data["time"].duplicated()].sort_values("time", ascending=True)

            carb_entries = pd.DataFrame(carb_data.groupby("time")["carbs"].sum()).reset_index()

            # Merge carb absorption data if it exists
            carb_absorption_col = "payload.com.loudnate.CarbKit.HKMetadataKey.AbsorptionTimeMinutes"

            if carb_absorption_col in carb_data.columns:
                carb_data.drop_duplicates("time", keep="last", inplace=True)

                carb_entries = pd.merge(
                    carb_entries,
                    carb_data[["time", carb_absorption_col]],
                    how="left",
                    on="time",
                )

                carb_entries.rename(columns={carb_absorption_col: "carb_absorption_minutes"}, inplace=True)
            else:
                carb_entries["carb_absorption_minutes"] = np.nan

            # Merge carbs within 5-min together
            daily_5min_ts = pd.merge(daily_5min_ts, carb_entries, how="left", on="time")
        else:
            daily_5min_ts["carbs"] = np.nan

        return daily_5min_ts

    def parse_settings(self, settings_df, start_time=None):
        """
        Parse settings into a timeline from the settings raw data

        Parameters
        ----------
        settings_df: pd.DataFrame
        start_time: datetime.datetime
        """
        jaeb_inputs = self.input_settings_from_df(settings_df=settings_df)

        if start_time is not None:
            jaeb_inputs['time_to_calculate_at'] = datetime.datetime.fromisoformat(
                pd.to_datetime(start_time).isoformat())
        else:
            issue_report_start = datetime.datetime.fromisoformat(
                pd.to_datetime(settings_df['time'].values[0]).isoformat())
            rounded_start_time = issue_report_start + datetime.timedelta(days=days_in)
            rounded_start_time += datetime.timedelta(minutes=2, seconds=30)
            rounded_start_time -= datetime.timedelta(minutes=rounded_start_time.minute % 5,
                                                     seconds=rounded_start_time.second,
                                                     microseconds=rounded_start_time.microsecond)

            jaeb_inputs['time_to_calculate_at'] = rounded_start_time

        # fixme: in the future, pull duration values from the data
        jaeb_inputs["basal_rate_units"] = ["U" for _ in jaeb_inputs["basal_rate_values"]]
        jaeb_inputs["carb_ratio_value_units"] = [jaeb_inputs["carb_value_units"] + "/U" for _ in jaeb_inputs[
            "carb_ratio_values"]]
        jaeb_inputs["target_range_value_units"] = ["mg/dL" for _ in jaeb_inputs["target_range_minimum_values"]]
        jaeb_inputs["sensitivity_ratio_value_units"] = ["mg/dL/U" for _ in jaeb_inputs["sensitivity_ratio_values"]]
        jaeb_inputs["settings_dictionary"]["dynamic_carb_absorption_enabled"] = True

        self.loop_inputs_dict = jaeb_inputs

    def input_settings_from_df(self, settings_df):
        """
        Gets settings from dataframe.

        Parameters
        ----------
        settings_df: pd.DataFrame
        """

        dict_ = dict()
        dict_["settings_dictionary"] = dict()
        settings_names = [
            "basal_rate",
            "suspend_threshold",
            "correction_target_lower",
            "correction_target_upper",
            "maximum_basal_rate",
            "insulin_sensitivity_factor",
            "carb_ratio",
            "maximum_bolus"
        ]

        for col, val in settings_df.iteritems():
            if col == "insulin_model":
                model = val.values[0]
                obj = []
                if model == "fiasp":
                    obj = [360, 55]
                elif "humalog" in model:
                    if "Child" in model:
                        obj = [360, 65]
                    elif "Adult" in model:
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
                    dict_["settings_dictionary"]["default_absorption_times"][2] = value / 60  # convert to minutes
                elif "medium" in col:
                    dict_["settings_dictionary"]["default_absorption_times"][1] = value / 60
                elif "fast" in col:
                    dict_["settings_dictionary"]["default_absorption_times"][0] = value / 60
            elif col in settings_names:
                if "correction_target" in col:
                    if col == "correction_target_lower":
                        other_col = "correction_target_upper"
                    else:
                        other_col = "correction_target_lower"
                    schedule = settings_df[[col, other_col, "hour"]].loc[((settings_df[col].shift() != settings_df[
                        col]) | (settings_df[other_col].shift() != settings_df[other_col]))]

                    for i in schedule.index:  # parse strings
                        time = schedule.loc[i, "hour"]
                        lower_value = schedule.loc[i, "correction_target_lower"]
                        upper_value = schedule.loc[i, "correction_target_upper"]

                        time_in_hours = int(time)
                        minutes = int((time - time_in_hours) * 60)
                        if 'target_range_start_times' in dict_:
                            dict_['target_range_start_times'].append(
                                datetime.time(hour=time_in_hours, minute=minutes))
                            dict_['target_range_minimum_values'].append(lower_value)
                            dict_['target_range_maximum_values'].append(upper_value)
                        else:
                            dict_['target_range_start_times'] = [datetime.time(hour=time_in_hours, minute=minutes)]
                            dict_['target_range_minimum_values'] = [lower_value]
                            dict_['target_range_maximum_values'] = [upper_value]
                else:
                    schedule = settings_df[[col, "hour"]].loc[settings_df[col].shift() != settings_df[
                        col]]
                    schedule_type = col.replace("schedule", "")
                    if "sensitivity" in col:
                        schedule_type = "sensitivity_ratio"

                    for i in schedule.index:  # parse strings
                        time = schedule.loc[i, "hour"]
                        value = schedule.loc[i, col]

                        time_in_hours = int(time)
                        minutes = int((time-time_in_hours) * 60)
                        if schedule_type + "start_times" in dict_:
                            dict_[schedule_type + "_start_times"].append(datetime.time(hour=time_in_hours, minute=minutes))
                            dict_[schedule_type + "_values"].append(value)
                        else:
                            dict_[schedule_type + "_start_times"] = [datetime.time(hour=time_in_hours, minute=minutes)]
                            dict_[schedule_type + "_values"] = [value]

        return dict_

    def parse_carbs(self, carb_df):
        """
        Get the carb start_times, carb_values, units and absorption times from the time series data.

        Parameters
        ----------
        carb_df: pd.DataFrame
        """
        for col, _ in carb_df.items():
            temp_df = carb_df[col]
            temp_array = []
            if col == 'time':
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
            elif 'carbohydrate.net' in col:
                self.loop_inputs_dict['carb_values'] = list(temp_df.values)
            elif 'carbohydrate.unit' in col:
                self.loop_inputs_dict['carb_units'] = list(temp_df.values)
            elif 'carbohydrate.unit' in col:
                self.loop_inputs_dict['carb_units'] = list(temp_df.values)
            elif "AbsorptionTime" in col:
                for v in temp_df.values:
                    if pd.isna(v):
                        obj = 7200
                    else:
                        obj = v

                    temp_array = np.append(temp_array, obj)

                self.loop_inputs_dict["carb_absorption_times"] = list(temp_array)

    def parse_cgm(self, cgm_df):
        """
        Get glucose values up to the end time.

        Parameters
        ----------
        cgm_df: pd.DataFrame
        """
        cgm_data = cgm_df.dropna()
        history_end_time = self.get_simulation_start_time()

        self.loop_inputs_dict["glucose_values"] = [cgm_data["cgm"].values[i] for i in range(
            len(cgm_data["cgm"].values)) if datetime.datetime.fromisoformat(cgm_data["rounded_local_time"].values[i]) <=
            history_end_time]
        self.loop_inputs_dict["glucose_dates"] = [datetime.datetime.fromisoformat(
            pd.to_datetime(date).isoformat()) for date in cgm_data["rounded_local_time"].values if
            datetime.datetime.fromisoformat(date) <= history_end_time]

        self.loop_inputs_dict["full_glucose_values"] = [val for val in cgm_data["cgm"].values]
        self.loop_inputs_dict["full_glucose_dates"] = [datetime.datetime.fromisoformat(
            pd.to_datetime(date).isoformat()) for date in cgm_data["rounded_local_time"].values]

    def parse_insulin(self, insulin_df):
        """
        Divide insulin records into bolus and temp basal timelines and record.

        Parameters
        ----------
        insulin_df: pd.DataFrame
        """
        #TODO: Test and debug - make sure that all insulin records from the raw datasets are copied over

        insulin_df['basal_end_dates'] = insulin_df['rounded_local_time'].apply(
            lambda x: (datetime.datetime.fromisoformat(pd.to_datetime(x).isoformat()) + datetime.timedelta(
                minutes=5)).isoformat()
        )
        insulin_df = insulin_df.set_index('rounded_local_time')
        temp_basal = insulin_df[['set_basal_rate', 'basal_pulse_delivered', 'basal_end_dates']].dropna()
        bolus = insulin_df['bolus'].dropna()

        bolus_start_times, bolus_values, bolus_units, bolus_dose_types, bolus_delivered_units = ([], [], [], [], [])
        temp_basal_start_times, temp_basal_duration_minutes, temp_basal_values, \
        temp_basal_units, temp_basal_dose_types, temp_basal_delivered_units = ([], [], [], [], [], [])
        for date, dose in bolus.items():
            bolus_date = datetime.datetime.fromisoformat(pd.to_datetime(date).isoformat())
            bolus_start_times.append(bolus_date)
            bolus_values.append(dose)
            bolus_dose_types.append(DoseType.bolus)
            bolus_units.append("U")
            bolus_delivered_units.append(
                insulin_df.at[date, 'total_insulin_delivered'] - insulin_df.at[date,'basal_pulse_delivered']
            )

        previous = None
        for date, dose in temp_basal.iterrows():
            date_as_date = datetime.datetime.fromisoformat(pd.to_datetime(date).isoformat())
            if not previous:
                temp_basal_start_times.append(date_as_date)
                temp_basal_values.append(dose['set_basal_rate'])
                temp_basal_units.append("U/hr")
                temp_basal_delivered_units.append(dose['basal_pulse_delivered'])
                temp_basal_dose_types.append(DoseType.tempbasal)
                temp_basal_duration_minutes.append(5)
                current_date = copy.deepcopy(date_as_date) + datetime.timedelta(minutes=5)
                last_date = datetime.datetime.fromisoformat(pd.to_datetime(insulin_df.index[-1]).isoformat())
                if current_date < last_date:
                    while insulin_df.at[current_date.isoformat().replace("T", " "), 'basal_pulse_delivered'] == dose['basal_pulse_delivered']:
                        temp_basal_duration_minutes[-1] = temp_basal_duration_minutes[-1] + 5
                        temp_basal_delivered_units[-1] = temp_basal_delivered_units[-1] + dose['basal_pulse_delivered']
                        current_date = current_date + datetime.timedelta(minutes=5)
                        if current_date >= last_date:
                            break
            else:
                if (previous[1]['set_basal_rate'] == dose['set_basal_rate']) and (date_as_date - previous[0] ==
                                                                                  datetime.timedelta(
                                                                                      minutes=5)):
                    pass
                else:
                    temp_basal_values.append(dose['set_basal_rate'])
                    temp_basal_units.append("U/hr")
                    temp_basal_start_times.append(date_as_date)
                    temp_basal_delivered_units.append(dose['basal_pulse_delivered'])
                    temp_basal_dose_types.append(DoseType.tempbasal)
                    temp_basal_duration_minutes.append(5)
                    current_date = copy.deepcopy(date_as_date) + datetime.timedelta(minutes=5)
                    if current_date < last_date:
                        while insulin_df.at[current_date.isoformat().replace("T", " "), 'basal_pulse_delivered'] == \
                                dose['basal_pulse_delivered']:
                            temp_basal_duration_minutes[-1] = temp_basal_duration_minutes[-1] + 5
                            temp_basal_delivered_units[-1] = temp_basal_delivered_units[-1] + dose[
                                'basal_pulse_delivered']
                            current_date = current_date + datetime.timedelta(minutes=5)
                            if current_date >= last_date:
                                break

            previous = date_as_date, dose

        self.loop_inputs_dict['bolus_start_times'] = bolus_start_times
        self.loop_inputs_dict['bolus_values'] = bolus_values
        self.loop_inputs_dict['bolus_units'] = bolus_units
        self.loop_inputs_dict['bolus_delivered_units'] = bolus_delivered_units
        self.loop_inputs_dict['bolus_dose_types'] = bolus_dose_types
        self.loop_inputs_dict['temp_basal_start_times'] = temp_basal_start_times
        self.loop_inputs_dict['temp_basal_values'] = temp_basal_values
        self.loop_inputs_dict['temp_basal_duration_minutes'] = temp_basal_duration_minutes
        self.loop_inputs_dict['temp_basal_dose_types'] = temp_basal_dose_types
        self.loop_inputs_dict['temp_basal_units'] = temp_basal_units
        self.loop_inputs_dict['temp_basal_delivered_units'] = temp_basal_delivered_units