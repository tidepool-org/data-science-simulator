__author__ = "Cameron Summers"

import datetime
import pandas as pd
import numpy as np

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import SettingSchedule24Hr, Simulation, TargetRangeSchedule24hr
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
from tidepool_data_science_simulator.models.measures import Bolus, Carb, InsulinSensitivityFactor, TargetRange

from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


def calculate_time_in_range(bg_values, target_low=70, target_high=180):
    """
    Calculate the percentage of time spent in target range.
    
    Args:
        bg_values: List of blood glucose values
        target_low: Lower bound of target range (mg/dL)
        target_high: Upper bound of target range (mg/dL)
    
    Returns:
        float: Percentage of time in range (0-100)
    """
    bg_array = np.array(bg_values)
    in_range = (bg_array >= target_low) & (bg_array <= target_high)
    return (np.sum(in_range) / len(bg_array)) * 100


def run_simulation_with_insulin_model(insulin_model, sim_duration_hrs=24):
    """
    Run a simulation with specified insulin model.
    
    Args:
        insulin_model: String identifier for insulin model ('novolog', 'fiasp', 'afrezza', 'lyumjev')
        sim_duration_hrs: Duration of simulation in hours
    
    Returns:
        tuple: (sim_results_df, time_in_range_percent)
    """
    target_glucose = 120
    
    # Set up patient, sensor, and pump configurations
    t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=300)
    t0, sensor_config = get_canonical_sensor_config(start_value=300)
    t0, controller_config = get_canonical_controller_config()
    t0, pump_config = get_canonical_risk_pump_config()

    # Create meal scenario with bolus
    meal_time = t0 #+ datetime.timedelta(hours=2)
    bolus_timeline = BolusTimeline(
        datetimes=[meal_time], 
        events=[Bolus(5.0, "U")]
    )
    patient_config.bolus_event_timeline = bolus_timeline
    pump_config.bolus_event_timeline = bolus_timeline

    # Add carbohydrate intake
    true_carb_timeline = CarbTimeline(
        datetimes=[meal_time], 
        events=[Carb(45.0, "g", 180)]
    )
    # patient_config.carb_event_timeline = true_carb_timeline
    
    reported_carb_timeline = CarbTimeline(
        datetimes=[meal_time], 
        events=[Carb(45.0, "g", 180)]
    )
    # pump_config.carb_event_timeline = reported_carb_timeline

    # Set target range
    new_target_range_schedule = TargetRangeSchedule24hr(
        t0,
        start_times=[datetime.time(0, 0, 0)],
        values=[TargetRange(target_glucose, target_glucose, "mg/dL")],
        duration_minutes=[1440]
    )
    pump_config.target_range_schedule = new_target_range_schedule

    # New insulin sensitivity factor
    insulin_sensitivity_schedule=SettingSchedule24Hr(
        t0,
        "ISF",
        start_times=[datetime.time(0, 0, 0)],
        values=[InsulinSensitivityFactor(40.0, "mg/dL/U")],
        duration_minutes=[1440]
    )
    pump_config.insulin_sensitivity_schedule = insulin_sensitivity_schedule
    patient_config.insulin_sensitivity_schedule = insulin_sensitivity_schedule
    
    # Initialize components
    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)
    controller = SwiftLoopController(t0, controller_config)
    
    # Set the insulin model
    controller.controller_config.controller_settings['model'] = insulin_model
    controller.controller_config.controller_settings['partial_application_factor'] = 0.4

    # Create virtual patient
    vp = VirtualPatient(
        time=DATETIME_DEFAULT,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )

    # Run simulation
    sim_id = f"insulin_model_{insulin_model}"
    sim = Simulation(
        time=t0,
        duration_hrs=sim_duration_hrs,
        virtual_patient=vp,
        controller=controller,
        sim_id=sim_id
    )

    sim.run()
    sim_results_df = sim.get_results_df()
    
    # Calculate time in range
    bg_values = sim_results_df["bg"].tolist()
    time_in_range = calculate_time_in_range(bg_values[137:])
    
    return sim_results_df, time_in_range


def test_insulin_model_comparison():
    """
    Compare four different insulin models and their time in range performance.
    Tests: novolog, fiasp, afrezza, and lyumjev
    """
    insulin_models = ['novolog', 'fiasp', 'afrezza', 'lyumjev']
    results = {}
    sim_results_data = {}
    
    print("Running insulin model comparison test...")
    print("=" * 50)
    
    for model in insulin_models:
        print(f"Testing {model}...")
        
        try:
            sim_df, tir = run_simulation_with_insulin_model(model, sim_duration_hrs=8)
            results[model] = {
                'time_in_range': tir,
                'final_bg': sim_df["bg"].tolist()[-1],
                'mean_bg': np.mean(sim_df["bg"].tolist()),
                'bg_std': np.std(sim_df["bg"].tolist())
            }
            sim_results_data[f"{model}_simulation"] = sim_df
            
            print(f"  Time in Range: {tir:.1f}%")
            print(f"  Final BG: {results[model]['final_bg']:.1f} mg/dL")
            print(f"  Mean BG: {results[model]['mean_bg']:.1f} mg/dL")
            print(f"  BG Std Dev: {results[model]['bg_std']:.1f} mg/dL")
            print()
            
        except Exception as e:
            print(f"  Error testing {model}: {e}")
            results[model] = None
            continue
    
    # Create summary comparison
    print("COMPARISON SUMMARY")
    print("=" * 50)
    
    valid_results = {k: v for k, v in results.items() if v is not None}
    
    if valid_results:
        # Sort by time in range (descending)
        sorted_results = sorted(valid_results.items(), 
                              key=lambda x: x[1]['time_in_range'], 
                              reverse=True)
        
        print("Ranking by Time in Range:")
        for i, (model, data) in enumerate(sorted_results, 1):
            print(f"{i}. {model.upper()}: {data['time_in_range']:.1f}% TIR")
        
        print(f"\nBest performing model: {sorted_results[0][0].upper()}")
        print(f"Average TIR across all models: {np.mean([v['time_in_range'] for v in valid_results.values()]):.1f}%")
        
        # Statistical comparison
        tir_values = [v['time_in_range'] for v in valid_results.values()]
        print(f"TIR Standard Deviation: {np.std(tir_values):.1f}%")
        print(f"TIR Range: {max(tir_values) - min(tir_values):.1f}%")
    
    # Uncomment to plot results
    if sim_results_data:
        plot_sim_results(sim_results_data)
    
    # Assert that at least one model achieves reasonable time in range
    if valid_results:
        best_tir = max(v['time_in_range'] for v in valid_results.values())
        assert best_tir > 60, f"Best TIR ({best_tir:.1f}%) should be > 60%"
        print(f"\n✓ Test passed: Best model achieved {best_tir:.1f}% time in range")
    else:
        raise AssertionError("No insulin models completed successfully")
    
    return results


if __name__ == "__main__":
    test_insulin_model_comparison()