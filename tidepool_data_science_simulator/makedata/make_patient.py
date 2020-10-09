__author__ = "Cameron Summers"

import datetime
import numpy as np

from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, BasalSchedule24hr, TargetRangeSchedule24hr
)
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline, ActionTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import PumpConfig, PatientConfig, SensorConfig
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace, Bolus, Carb
)
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.models.controller import LoopController

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

SINGLE_SETTING_START_TIME = datetime.time(hour=0, minute=0, second=0)
SINGLE_SETTING_DURATION = 1440
DATETIME_DEFAULT = datetime.datetime(year=2019, month=8, day=15, hour=12, minute=0, second=0)


def get_canonical_risk_pump_config(t0=DATETIME_DEFAULT):
    """
    Get canonical pump config

    Parameters
    ----------
    t0

    Returns
    -------
    PumpConfig
    """

    pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    pump_config = PumpConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(0.3, "U/hr")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(20.0, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(150.0, "mg/dL/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        target_range_schedule=TargetRangeSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_event_timeline=pump_carb_timeline,
        bolus_event_timeline=pump_bolus_timeline
    )

    return t0, pump_config


def get_canonical_sensor_config(t0=DATETIME_DEFAULT):
    """
    Get canonical sensor config

    Parameters
    ----------
    t0

    Returns
    -------
    SensorConfig
    """

    num_values = 137
    sensor_bg_dates = [t0 - datetime.timedelta(minutes=i * 5) for i in range(num_values)]
    sensor_bg_dates.reverse()
    sensor_bg_values = [110.0] * num_values
    sensor_bg_history = GlucoseTrace(sensor_bg_dates, sensor_bg_values)

    sensor_config = SensorConfig(sensor_bg_history)
    return t0, sensor_config


def get_canonical_risk_patient_config(t0=DATETIME_DEFAULT):
    """
    Get canonical patient config

    Parameters
    ----------
    t0

    Returns
    -------
    PatientConfig
    """

    patient_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    patient_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    num_glucose_values = 137
    true_bg_dates = [t0 - datetime.timedelta(minutes=i * 5) for i in range(num_glucose_values)]
    true_bg_dates.reverse()
    true_bg_values = [110.0] * num_glucose_values
    true_bg_history = GlucoseTrace(true_bg_dates, true_bg_values)

    patient_config = PatientConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(0.3, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(20.0, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(150.0, "md/dL / U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        glucose_history=true_bg_history,
        carb_event_timeline=patient_carb_timeline,
        bolus_event_timeline=patient_bolus_timeline,
        action_timeline=ActionTimeline(),
        recommendation_accept_prob=0.0  # Does not accept any recommendations
    )

    return t0, patient_config


def get_canonical_risk_patient(t0=DATETIME_DEFAULT,
                               patient_class=VirtualPatient,
                               patient_config=None,
                               pump_class=ContinuousInsulinPump,
                               pump_config=None,
                               sensor_class=IdealSensor,
                               sensor_config=None,
                               ):
    """
    Get patient with settings from the FDA risk analysis scenario template.
    Putting here to simplify testing and decouple from specific scenarios.

    Returns
    -------
    (datetime.datetime, VirtualPatient)
        Current time of patient, Canonical patient
    """

    if pump_config is None:
        t0, pump_config = get_canonical_risk_pump_config(t0)

    if patient_config is None:
        t0, patient_config = get_canonical_risk_patient_config(t0)

    if sensor_config is None:
        t0, sensor_config = get_canonical_sensor_config(t0)

    pump = pump_class(
        time=t0,
        pump_config=pump_config
    )

    sensor = sensor_class(
        time=t0,
        sensor_config=sensor_config
    )

    virtual_patient = patient_class(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config,
    )

    return t0, virtual_patient


def get_variable_risk_patient_config(random_state, t0=DATETIME_DEFAULT):
    """
    Get canonical patient config

    Parameters
    ----------
    t0

    Returns
    -------
    PatientConfig
    """
    patient_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    patient_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    num_glucose_values = 137
    true_bg_dates = [t0 - datetime.timedelta(minutes=i * 5) for i in range(num_glucose_values)]
    true_bg_dates.reverse()
    true_bg_values = [110.0] * num_glucose_values
    true_bg_history = GlucoseTrace(true_bg_dates, true_bg_values)

    total_hours_in_day = 24
    hours_in_day = range(total_hours_in_day)
    carb_gain = random_state.uniform(7, 9)
    basal_rates = [random_state.uniform(0.2, 0.4) for _ in hours_in_day]
    carb_ratios = [random_state.uniform(18, 22) for _ in hours_in_day]
    isfs = [carb_gain * carb_ratio for carb_ratio in carb_ratios]

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
        recommendation_accept_prob=0.0  # Does not accept any recommendations
    )

    return t0, patient_config


def get_pump_config_from_patient(random_state, patient_config, risk_level=0, t0=DATETIME_DEFAULT):
    """
    Get pump config that is the mean of the true patient config values.

    Parameters
    ----------
    t0

    Returns
    -------
    PumpConfig
    """
    pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
    pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    # Risk Configurable values
    patient_basal_values = [br.value for br in patient_config.basal_schedule.schedule.values()]
    patient_basal_mean = np.mean(patient_basal_values)
    patient_basal_std = np.std(patient_basal_values)

    patient_cir_values = [cir.value for cir in patient_config.carb_ratio_schedule.schedule.values()]
    patient_cir_mean = np.mean(patient_cir_values)
    patient_cir_std = np.std(patient_cir_values)

    patient_isf_values = [isf.value for isf in patient_config.insulin_sensitivity_schedule.schedule.values()]
    patient_isf_mean = np.mean(patient_isf_values)
    patient_isf_std = np.std(patient_isf_values)

    pump_br = random_state.normal(patient_basal_mean, patient_basal_std * risk_level)
    pump_cir = random_state.normal(patient_cir_mean, patient_cir_std * risk_level)
    pump_isf =random_state.normal(patient_isf_mean, patient_isf_std * risk_level)

    pump_config = PumpConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(pump_br, "U/hr")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[CarbInsulinRatio(pump_cir, "g/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[SINGLE_SETTING_START_TIME],
            values=[InsulinSensitivityFactor(pump_isf, "mg/dL/U")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        target_range_schedule=TargetRangeSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        ),
        carb_event_timeline=pump_carb_timeline,
        bolus_event_timeline=pump_bolus_timeline
    )

    return t0, pump_config
