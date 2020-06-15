__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, CarbTimeline, BolusTimeline, BasalSchedule24hr, TargetRangeSchedule24hr
)
from tidepool_data_science_simulator.makedata.scenario_parser import PumpConfig, PatientConfig
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace, Bolus, Carb
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
                               sensor_class=IdealSensor,
                               include_initial_events=False):
    """
    Get patient with settings from the FDA risk analysis scenario template.
    Putting here to simplify testing and decouple from specific scenarios.

    Returns
    -------
    (datetime.datetime, VirtualPatient)
        Current time of patient, Canonical patient
    """

    if t0 is None:
        t0 = datetime.datetime(year=2019, month=8, day=15, hour=12, minute=0, second=0)

    if include_initial_events:
        pump_carb_timeline = CarbTimeline([t0], [Carb(40.0, "g", 180)])
        pump_bolus_timeline =BolusTimeline([t0], [Bolus(2.0, "U")])

        patient_carb_timeline = CarbTimeline([t0], [Carb(20.0, "g", 180)])
        patient_bolus_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
    else:
        pump_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
        pump_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

        patient_carb_timeline = CarbTimeline([t0], [Carb(0.0, "g", 180)])
        patient_bolus_timeline = BolusTimeline([t0], [Bolus(0.0, "U")])

    setting_start_time = datetime.time(hour=0, minute=0, second=0)

    num_glucose_values = 137
    glucose_dates = [t0 - datetime.timedelta(minutes=i * 5) for i in range(num_glucose_values)]
    glucose_dates.reverse()
    glucose_values = [110.0] * num_glucose_values
    glucose_history = GlucoseTrace(glucose_dates, glucose_values)
    sensor_glucose_history = glucose_history

    pump_config = PumpConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[setting_start_time],
            values=[BasalRate(0.3, "U/hr")],
            duration_minutes=[1440.0]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[setting_start_time],
            values=[CarbInsulinRatio(20.0, "g/U")],
            duration_minutes=[1440.0]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[setting_start_time],
            values=[InsulinSensitivityFactor(150.0, "mg/dL/U")],
            duration_minutes=[1440.0]
        ),
        target_range_schedule=TargetRangeSchedule24hr(
            t0,
            start_times=[setting_start_time],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[1440.0]
        ),
        carb_event_timeline=pump_carb_timeline,
        bolus_event_timeline=pump_bolus_timeline
    )

    pump = pump_class(
        time=t0,
        pump_config=pump_config
    )

    sensor = sensor_class(sensor_config=sensor_glucose_history)

    patient_config = PatientConfig(
        basal_schedule=BasalSchedule24hr(
            t0,
            start_times=[setting_start_time],
            values=[BasalRate(0.3, "mg/dL")],
            duration_minutes=[1440.0]
        ),
        carb_ratio_schedule=SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[setting_start_time],
            values=[CarbInsulinRatio(20.0, "g/U")],
            duration_minutes=[1440.0]
        ),
        insulin_sensitivity_schedule=SettingSchedule24Hr(
            t0,
            "ISF",
            start_times=[setting_start_time],
            values=[InsulinSensitivityFactor(150.0, "md/dL / U")],
            duration_minutes=[1440.0]
        ),
        glucose_history=glucose_history,
        carb_event_timeline=patient_carb_timeline,
        bolus_event_timeline=patient_bolus_timeline,
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

