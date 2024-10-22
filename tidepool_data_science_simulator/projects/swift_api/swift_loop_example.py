__author__ = "Mark Connolly"


import time, datetime
from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import SettingSchedule24Hr, Simulation, TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController, SwiftLoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import (
  DATETIME_DEFAULT, SINGLE_SETTING_DURATION, SINGLE_SETTING_START_TIME, 
  get_canonical_risk_patient_config, get_canonical_risk_pump_config, get_canonical_sensor_config
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import Bolus, Carb, InsulinSensitivityFactor, TargetRange

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

def test_swift_api():
    """
    Extract data from simulator and feed into LoopAlgorithmToPython
    """
    target = 120

    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=100)
    t0, sensor_config = get_canonical_sensor_config(start_value=100)
    t0, controller_config = get_canonical_controller_config()
    t0, pump_config = get_canonical_risk_pump_config()

    controller_config.controller_settings['partial_application_factor'] = 0.9
    
    dt = datetime.time(hour=0, minute=0, second=0)

    true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(20.0, "U", 180)])
    patient_config.carb_event_timeline = true_carb_timeline
    reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(25.0, "U", 240)])
    pump_config.carb_event_timeline = reported_carb_timeline

    insulin_sensitivity_timeline=SettingSchedule24Hr(
        t0,
        "ISF",
        start_times=[dt],
        values=[InsulinSensitivityFactor(40.0, "mg/dL/U")],
        duration_minutes=[SINGLE_SETTING_DURATION * 2]
    )
    pump_config.insulin_sensitivity_schedule = insulin_sensitivity_timeline
    
    insulin_sensitivity_schedule=SettingSchedule24Hr(
        t0,
        "ISF",
        start_times=[SINGLE_SETTING_START_TIME],
        values=[InsulinSensitivityFactor(40.0, "md/dL / U")],
        duration_minutes=[SINGLE_SETTING_DURATION]
    )
    patient_config.insulin_sensitivity_schedule = insulin_sensitivity_schedule

    new_target_range_schedule = \
        TargetRangeSchedule24hr(
            t0,
            start_times=[datetime.time(0, 0, 0)],
            values=[TargetRange(100, 120, "mg/dL")],
            duration_minutes=[1440]
        )
    pump_config.target_range_schedule = new_target_range_schedule
    
    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)

    controller = SwiftLoopController(t0, controller_config)

    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    sim_id = "test_swift_api"
    sim = Simulation(
        time=t0,
        duration_hrs=24,
        virtual_patient=vp,
        controller=controller,
        sim_id=sim_id,
        multiprocess=False
    )
    sim.run()
    
    sim_results_df = sim.get_results_df()

    plot_sim_results({sim_id: sim_results_df})

    return 1




if __name__ == "__main__":
    start_time = time.time()
    test_swift_api()
    print("--- %s seconds ---" % (time.time() - start_time))