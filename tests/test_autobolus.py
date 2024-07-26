__author__ = "Mark Jude Connolly"

import datetime

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation, TargetRangeSchedule24hr
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
from tidepool_data_science_simulator.models.measures import Bolus, Carb, TargetRange

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def test_autobolus():
    """
    Make sure Loop can bring a person close to their target range over 24 hours.
    """
    target = 120

    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=250)
    t0, sensor_config = get_canonical_sensor_config(start_value=250)
    t0, controller_config = get_canonical_controller_config()
    t0, pump_config = get_canonical_risk_pump_config()

    # bolus_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(1.0, "U")])
    # patient_config.bolus_event_timeline = bolus_timeline
    # pump_config.bolus_event_timeline = bolus_timeline

    true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 180)])
    patient_config.carb_event_timeline = true_carb_timeline
    reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 180)])
    pump_config.carb_event_timeline = reported_carb_timeline

    new_target_range_schedule = \
        TargetRangeSchedule24hr(
            t0,
            start_times=[datetime.time(0, 0, 0)],
            values=[TargetRange(target, target, "mg/dL")],
            duration_minutes=[1440]
        )
    pump_config.target_range_schedule = new_target_range_schedule

    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)

    controller_no_autobolus = LoopController(t0, controller_config)

    controller_config.controller_settings['minimum_autobolus'] = .1
    controller_config.controller_settings['maximum_autobolus'] = 10
    controller_config.controller_settings['partial_application_factor'] = 0.4
    
    controller_autobolus = LoopController(t0, controller_config)
    

    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    sim_id = "simulation_no_autobolus"
    sim = Simulation(
        time=t0,
        duration_hrs=8,
        virtual_patient=vp,
        controller=controller_no_autobolus,
        sim_id=sim_id
    )

    sim.run()
    no_autobolus_results = sim.get_results_df()

    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)

    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    sim_id = "simulation_autobolus"
    sim = Simulation(
        time=t0,
        duration_hrs=8,
        virtual_patient=vp,
        controller=controller_autobolus,
        sim_id=sim_id
    )

    sim.run()
    autobolus_results = sim.get_results_df()
    # plot_sim_results({sim_id: sim_results_df})

    assert not any(no_autobolus_results.true_bolus[no_autobolus_results.true_bolus  > 0])
    assert any(autobolus_results.true_bolus[autobolus_results.true_bolus > 0])



if __name__ == "__main__":

    test_autobolus()