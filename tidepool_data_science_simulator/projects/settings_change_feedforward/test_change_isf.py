__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation, TargetRangeSchedule24hr, SettingSchedule24Hr
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import (
  DATETIME_DEFAULT, get_canonical_risk_patient_config, get_canonical_risk_pump_config,
    get_canonical_sensor_config
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import Bolus, Carb, TargetRange, InsulinSensitivityFactor

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def test_change_isf():
    """
    Make sure Loop can bring a person close to their target range over 8 hours.
    """
    targets = [150, 150]
    isf_values = [150, 15]
    isf_start_times = [datetime.time(0, 0, 0), datetime.time(13, 0, 0)]
    isf_durations = [13*60, 11*60]

    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=150)
    t0, sensor_config = get_canonical_sensor_config(start_value=100)
    t0, controller_config = get_canonical_controller_config()
    t0, pump_config = get_canonical_risk_pump_config()

    bolus_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(1.0, "U")])
    patient_config.bolus_event_timeline = bolus_timeline
    pump_config.bolus_event_timeline = bolus_timeline

    # true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(20.0, "U", 180)])
    # patient_config.carb_event_timeline = true_carb_timeline
    # reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])
    # pump_config.carb_event_timeline = reported_carb_timeline

    # new_target_range_schedule = \
    #     TargetRangeSchedule24hr(
    #         t0,
    #         start_times=[datetime.time(0, 0, 0), datetime.time(20, 0, 0)],
    #         values=[TargetRange(target, target, "mg/dL") for target in targets],
    #         duration_minutes=[1200, 240]
    #     )
    # pump_config.target_range_schedule = new_target_range_schedule

    new_pump_sensitivity_schedule =\
        SettingSchedule24Hr(
        time=t0,
        name="ISF",
        start_times=isf_start_times,
        values=[InsulinSensitivityFactor(value, 'mg/dL/U') for value in isf_values],
        duration_minutes=isf_durations,
    )
    pump_config.insulin_sensitivity_schedule = new_pump_sensitivity_schedule
    
    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)
    # controller = DoNothingController(t0, controller_config)
    controller = LoopController(t0, controller_config)

    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    sim_id = "basic_loop_control"
    sim = Simulation(
        time=t0,
        duration_hrs=8,
        virtual_patient=vp,
        controller=controller,
        sim_id=sim_id
    )

    sim.run()
    sim_results_df = sim.get_results_df()

    # assert abs(target - sim_results_df["bg"].tolist()[-1]) < 10

    plot_sim_results({sim_id: sim_results_df})


# if __name__ == "__main__":

#     test_change_isf()   

test_change_isf()