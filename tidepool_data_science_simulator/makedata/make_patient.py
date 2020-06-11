__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.models.simulation import SettingSchedule24Hr, CarbTimeline, BolusTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import PumpConfig, PatientConfig
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace
)
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.models.controller import LoopController

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

# TODO Soon: Use builder pattern instead of this with if statements
def get_canonical_risk_patient(t0=None,
                               patient_class=VirtualPatient,
                               pump_class=ContinuousInsulinPump,
                               sensor_class=IdealSensor):
    """
    Get patient with settings from the FDA risk analysis scenario template.
    Putting here to simplify testing and decouple from specific scenarios.

    Returns
    -------
    (datetime.datetime, VirtualPatient)
        Current time of patient, Canonical patient
    """

    if t0 is None:
        t0 = datetime.datetime(year=2020, month=1, day=1, hour=0, minute=0, second=0)

    setting_start_time = datetime.time(hour=0, minute=0, second=0)

    glucose_dates = [t0 + datetime.timedelta(minutes=i * 5) for i in range(144)]
    glucose_dates.reverse()
    glucose_values = [110] * 144  # 12 hours of glucose data
    glucose_history = GlucoseTrace(glucose_dates, glucose_values)

    pump_config = PumpConfig(
        basal_schedule=SettingSchedule24Hr(
            t0,
            "basal",
            start_times=[setting_start_time],
            values=[BasalRate(0.3, "U/hr")],
            duration_minutes=[1440]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[setting_start_time],
            values=[CarbInsulinRatio(20, "g/U")],
            duration_minutes=[1440]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[setting_start_time],
            values=[InsulinSensitivityFactor(150, "mg/dL/U")],
            duration_minutes=[1440]
        ),
        target_range_schedule=SettingSchedule24Hr(
            t0,
            "TargetRange",
            start_times=[setting_start_time],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[1440]
        ),
        carb_event_timeline=CarbTimeline(),
        bolus_event_timeline=BolusTimeline()
    )

    pump = pump_class(
        time=t0,
        pump_config=pump_config
    )

    sensor = sensor_class(sensor_config=None)

    patient_config = PatientConfig(
        basal_schedule=SettingSchedule24Hr(
            t0,
            "basal",
            start_times=[setting_start_time],
            values=[BasalRate(0.3, "mg/dL")],
            duration_minutes=[1440]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[setting_start_time],
            values=[CarbInsulinRatio(20, "g/U")],
            duration_minutes=[1440]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[setting_start_time],
            values=[InsulinSensitivityFactor(150, "md/dL / U")],
            duration_minutes=[1440]
        ),
        glucose_history=glucose_history,
        carb_event_timeline=CarbTimeline(),
        bolus_event_timeline=BolusTimeline(),
        recommendation_accept_prob=0.0  # Does not accept any recommendations
    )

    virtual_patient = patient_class(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config,
    )

    return t0, virtual_patient

