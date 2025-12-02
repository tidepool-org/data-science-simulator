__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.makedata.scenario_parser import SINGLE_SETTING_DURATION
from tidepool_data_science_simulator.models.simulation import BasalSchedule24hr, SettingSchedule24Hr, Simulation, TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import (
  DATETIME_DEFAULT, SINGLE_SETTING_START_TIME, get_canonical_risk_patient_config, get_canonical_risk_pump_config,
    get_canonical_sensor_config
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import BasalRate, Bolus, Carb, CarbInsulinRatio, InsulinSensitivityFactor, TargetRange

from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

import matplotlib.pyplot as plt
import numpy as np


def run_simulation(partial_application_factor=None, gradual_transition_threshold=40.0):
    """
    Make sure Loop can bring a person close to their target range over 24 hours.
    """
    target = 120
    
    # Simulation Settings:
    # Starting BG: 180 mg/dL
    # ISF (Insulin Sensitivity Factor): 150.0 mg/dL/U (default from get_canonical_risk_pump_config)
    # CIR (Carb-to-Insulin Ratio): 20.0 g/U (default from get_canonical_risk_pump_config)
    
    starting_bg = 120
    
    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=starting_bg)
    t0, sensor_config = get_canonical_sensor_config(start_value=800)
    t0, controller_config = get_canonical_controller_config()
    t0, pump_config = get_canonical_risk_pump_config()
    
    controller_config.controller_settings['max_basal_rate'] = 0.8 * 3.5 # U/hr
    # bolus_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(1.0, "U")])
    # patient_config.bolus_event_timeline = bolus_timeline
    # pump_config.bolus_event_timeline = bolus_timeline

    true_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(40.0, "U", 240)])
    patient_config.carb_event_timeline = true_carb_timeline
    reported_carb_timeline = CarbTimeline(datetimes=[t0], events=[Carb(40.0, "U", 240)])
    # pump_config.carb_event_timeline = reported_carb_timeline
    
    insulin_sensitivity_schedule=SettingSchedule24Hr(
        t0,
        "ISF",
        start_times=[SINGLE_SETTING_START_TIME],
        values=[InsulinSensitivityFactor(50.0, "md/dL / U")],
        duration_minutes=[SINGLE_SETTING_DURATION]
    )
    patient_config.insulin_sensitivity_schedule = insulin_sensitivity_schedule
    pump_config.insulin_sensitivity_schedule = insulin_sensitivity_schedule

    basal_schedule=BasalSchedule24hr(
        t0,
        start_times=[SINGLE_SETTING_START_TIME],
        values=[BasalRate(0.8, "U/hr")],
        duration_minutes=[SINGLE_SETTING_DURATION]
    )
    patient_config.basal_schedule = basal_schedule
    pump_config.basal_schedule = basal_schedule

    carb_ratio_schedule=SettingSchedule24Hr(
        t0,
        "CIR",
        start_times=[SINGLE_SETTING_START_TIME],
        values=[CarbInsulinRatio(15.0, "g/U")],
        duration_minutes=[SINGLE_SETTING_DURATION]
    )
    patient_config.carb_ratio_schedule = carb_ratio_schedule
    pump_config.carb_ratio_schedule = carb_ratio_schedule   

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

    controller = SwiftLoopController(t0, controller_config)
    if partial_application_factor is not None:
        controller.controller_config.controller_settings['partial_application_factor'] = partial_application_factor
    else:
        # Remove the key to enable temp basal mode
        controller.controller_config.controller_settings.pop('partial_application_factor', None)

    controller.controller_config.controller_settings['gradual_transition_threshold'] = gradual_transition_threshold
    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    if partial_application_factor is not None:
        sim_id = f"autobolus_paf_{partial_application_factor}"
    else:
        sim_id = "tempbasal"
    sim = Simulation(
        time=t0,
        duration_hrs=8,
        virtual_patient=vp,
        controller=controller,
        sim_id=sim_id
    )

    sim.run()
    sim_results_df = sim.get_results_df()
    
    return sim_id, sim_results_df


def test_basic_simulation():
    """
    Run simulations comparing autobolus (PAF=0.4) vs temp basal,
    then plot results including cumulative insulin comparison.
    """
    # Run autobolus simulation
    sim_id_autobolus, results_df_autobolus = run_simulation(partial_application_factor=0.4, gradual_transition_threshold=40.0)
    
    # Run temp basal simulation
    sim_id_tempbasal, results_df_tempbasal = run_simulation(partial_application_factor=None, gradual_transition_threshold=400.0)
    
    # Combine results
    all_results = {
        sim_id_autobolus: results_df_autobolus,
        sim_id_tempbasal: results_df_tempbasal
    }
    
    # Plot comparison with cumulative insulin
    fig, ax = plot_sim_results(all_results, plot_cumulative_insulin=True)
    ax[0].set_ylim((0, 250))  # Adjust BG plot y-limits
    ax[1].set_ylim((0, 4))  # Adjust insulin plot y-limits

    plt.show()

if __name__ == "__main__":
    test_basic_simulation()
