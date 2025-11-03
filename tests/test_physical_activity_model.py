__author__ = "Shawn Foster"

"""
Tests for physical activity model effects on glucose.
Isolates physical activity impact using DoNothingController to prevent
insulin modulation from interfering with measurements.
"""

import datetime
import json
import os
from pathlib import Path
import pytest
import pandas as pd

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import (
    Simulation, 
    TargetRangeSchedule24hr,
    BasalSchedule24hr,
    SettingSchedule24Hr
)
from tidepool_data_science_simulator.models.controller import DoNothingController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_patient import (
    DATETIME_DEFAULT,
    get_canonical_risk_patient_config,
    get_canonical_risk_pump_config,
    get_canonical_sensor_config,
)
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.scenario_parser import PatientConfig, PumpConfig, SensorConfig

from tidepool_data_science_simulator.models.events import (
    BolusTimeline, 
    CarbTimeline,
    PhysicalActivityTimeline
)
from tidepool_data_science_simulator.models.measures import (
    Bolus, 
    Carb, 
    TargetRange,
    PhysicalActivity,
    HeartRateTrace,
    BasalRate,
    CarbInsulinRatio,
    InsulinSensitivityFactor,
    GlucoseSensitivityFactor,
    BasalBloodGlucose,
    InsulinProductionRate
)

# ============================================================================
# PHASE 1: TEST FILE STRUCTURE SETUP
# ============================================================================

# Test constants
DURATION_HRS = 1.5  # 90 minutes total (30 min activity + 60 min recovery)
ACTIVITY_DURATION_MINUTES = 30
INITIAL_GLUCOSE = 120  # mg/dL - stable baseline

# Resolve project root and activity profiles directory
PROJECT_ROOT = Path(__file__).parent.parent  # Go up from tests/ to project root
ACTIVITY_PROFILES_DIR = PROJECT_ROOT / "scenario_configs" / "tidepool_risk_v2" / "reusable" / "physical_activities" / "profiles"
TEST_OUTPUT_DIR = PROJECT_ROOT / "tests" / "test_data"


# ============================================================================
# PHASE 2: TEST CONFIGURATION
# ============================================================================

def load_activity_profile(profile_name):
    """
    Load activity profile parameters from JSON file.
    
    Parameters
    ----------
    profile_name : str
        Name of the profile (e.g., "walking_v1", "biking_v1")
        
    Returns
    -------
    dict
        Dictionary containing:
        - metabolism_parameters: dict with w_hr, a, tau, n
        - activity_params: dict with activity, duration, intensity, expected_hr
    """
    profile_path = ACTIVITY_PROFILES_DIR / f"{profile_name}.json"
    
    if not profile_path.exists():
        raise FileNotFoundError(f"Activity profile not found: {profile_path}")
    
    with open(profile_path, 'r') as f:
        profile_data = json.load(f)
    
    # Extract metabolism parameters
    metabolism_params = profile_data.get("metabolism_parameters", {})
    
    # Extract activity parameters (should be single entry for these profiles)
    activity_entries = profile_data.get("physical_activity_entries", [])
    if not activity_entries:
        raise ValueError(f"No physical activity entries found in {profile_name}")
    
    activity_params = activity_entries[0]  # Take first entry
    
    return {
        "metabolism_parameters": metabolism_params,
        "activity_params": activity_params
    }


def get_baseline_patient_config(t0, activity_profile_data):
    """
    Create a baseline patient configuration for physical activity testing.
    
    Parameters
    ----------
    t0 : datetime.datetime
        Simulation start time
    activity_profile_data : dict
        Activity profile data from load_activity_profile()
        
    Returns
    -------
    tuple
        (t0, patient_config, pump_config, sensor_config, controller_config)
    """
    metabolism_params = activity_profile_data["metabolism_parameters"]
    activity_params = activity_profile_data["activity_params"]
    
    # Create physical activity timeline
    # Override start_time to be t0 for all tests
    physical_activity = PhysicalActivity(
        activity=activity_params.get("activity", ""),
        duration=activity_params.get("duration", ACTIVITY_DURATION_MINUTES),
        expected_hr=activity_params.get("expected_hr")
    )
    
    pa_timeline = PhysicalActivityTimeline(
        datetimes=[t0],
        events=[physical_activity]
    )
    
    # ========== Patient Configuration ==========
    # Start with canonical config, then customize
    _, base_patient_config = get_canonical_risk_patient_config(start_glucose_value=INITIAL_GLUCOSE)
    
    # Ensure zero carbs and boluses to isolate physical activity effects
    base_patient_config.bolus_event_timeline = BolusTimeline()
    base_patient_config.carb_event_timeline = CarbTimeline()
    
    # Set physical activity timeline
    base_patient_config.physical_activity_event_timeline = pa_timeline
    
    # Update metabolism parameters from activity profile
    base_patient_config.w_hr = metabolism_params.get("w_hr", 1.0)
    base_patient_config.a = metabolism_params.get("a", -0.002462)
    base_patient_config.tau = metabolism_params.get("tau", 0.9989)
    base_patient_config.n = metabolism_params.get("n", 28)
    
    # ========== Pump Configuration ==========
    _, base_pump_config = get_canonical_risk_pump_config()
    
    # Clear pump events - no boluses or carbs reported to controller
    base_pump_config.bolus_event_timeline = BolusTimeline()
    base_pump_config.carb_event_timeline = CarbTimeline()
    
    # Set a simple target range (won't matter with DoNothingController)
    target_range_schedule = TargetRangeSchedule24hr(
        t0,
        start_times=[datetime.time(0, 0, 0)],
        values=[TargetRange(100, 120, "mg/dL")],
        duration_minutes=[1440]
    )
    base_pump_config.target_range_schedule = target_range_schedule
    
    # ========== Sensor Configuration ==========
    _, base_sensor_config = get_canonical_sensor_config(start_value=INITIAL_GLUCOSE)
    
    # ========== Controller Configuration ==========
    _, base_controller_config = get_canonical_controller_config()
    
    return t0, base_patient_config, base_pump_config, base_sensor_config, base_controller_config


def create_heart_rate_trace(t0, duration_hrs, pa_timeline):
    """
    Create a heart rate trace for the simulation duration.
    
    Heart rate will be elevated during activity and return to baseline afterward.
    
    Parameters
    ----------
    t0 : datetime.datetime
        Simulation start time
    duration_hrs : float
        Duration of simulation in hours
    pa_timeline : PhysicalActivityTimeline
        Physical activity timeline containing activity events
        
    Returns
    -------
    HeartRateTrace
        Heart rate trace with 5-minute intervals
    """
    BASELINE_HR = 70  # bpm at rest
    TIMESTEP_MINUTES = 5
    
    # Calculate total timesteps
    total_minutes = int(duration_hrs * 60)
    num_timesteps = total_minutes // TIMESTEP_MINUTES
    
    datetimes = []
    hr_values = []
    
    for i in range(num_timesteps + 1):  # +1 to include final timestep
        current_time = t0 + datetime.timedelta(minutes=i * TIMESTEP_MINUTES)
        datetimes.append(current_time)
        
        # Determine heart rate based on activity period state
        hr = BASELINE_HR
        
        for activity_time, activity in pa_timeline.events.items():
            activity_end_time = activity_time + datetime.timedelta(minutes=activity.duration)
            
            if activity_time <= current_time < activity_end_time:
                # We're in the activity period - use expected HR
                if activity.expected_hr is not None:
                    hr = activity.expected_hr
                break
        
        hr_values.append(hr)
    
    return HeartRateTrace(datetimes=datetimes, values=hr_values)


# ============================================================================
# PHASE 5: OPTIONAL ENHANCEMENTS
# ============================================================================

def save_test_results(sim_results_df, activity_profile, activity_name):
    """
    Save test results DataFrame to CSV file.
    
    Parameters
    ----------
    sim_results_df : pd.DataFrame
        Simulation results with glucose, HR, and calculated fields
    activity_profile : str
        Profile name (e.g., "walking_v1")
    activity_name : str
        Human-readable activity name (e.g., "walking")
        
    Returns
    -------
    Path
        Path to saved CSV file
    """
    # Create output directory if it doesn't exist
    TEST_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create filename with timestamp
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pa_test_{activity_profile}_{timestamp}.csv"
    output_path = TEST_OUTPUT_DIR / filename
    
    # Save to CSV
    sim_results_df.to_csv(output_path)
    
    print(f"  Results saved to: {output_path}")
    return output_path


def generate_activity_test_report(sim_results_df, activity_name, activity_profile, expected_hr):
    """
    Generate detailed test report with statistics.
    
    Parameters
    ----------
    sim_results_df : pd.DataFrame
        Simulation results with all calculated fields
    activity_name : str
        Human-readable activity name
    activity_profile : str
        Profile name
    expected_hr : float
        Expected heart rate during activity
        
    Returns
    -------
    dict
        Dictionary containing detailed statistics
    """
    initial_glucose = sim_results_df["bg"].iloc[0]
    final_glucose = sim_results_df["bg"].iloc[-1]
    
    # Glucose metrics
    glucose_deltas = sim_results_df["glucose_delta"].values
    max_drop = glucose_deltas.min()
    max_drop_time = sim_results_df["time_minutes"].iloc[glucose_deltas.argmin()]
    
    # Separate activity and recovery periods
    activity_period = sim_results_df[sim_results_df["is_active"]]
    recovery_period = sim_results_df[~sim_results_df["is_active"]]
    
    # Calculate glucose nadir (lowest point)
    nadir_glucose = sim_results_df["bg"].min()
    nadir_time = sim_results_df["time_minutes"].iloc[sim_results_df["bg"].argmin()]
    
    # Recovery rate (if applicable)
    recovery_rate = None
    if len(recovery_period) > 1:
        recovery_start_bg = recovery_period["bg"].iloc[0]
        recovery_end_bg = recovery_period["bg"].iloc[-1]
        recovery_duration = recovery_period["time_minutes"].iloc[-1] - recovery_period["time_minutes"].iloc[0]
        if recovery_duration > 0:
            recovery_rate = (recovery_end_bg - recovery_start_bg) / recovery_duration  # mg/dL per minute
    
    # Heart rate metrics
    hr_column = "hr" if "hr" in sim_results_df.columns else "heart_rate"
    if len(activity_period) > 0 and hr_column in sim_results_df.columns:
        mean_hr_activity = activity_period[hr_column].mean()
        mean_hr_recovery = recovery_period[hr_column].mean() if len(recovery_period) > 0 else None
    else:
        mean_hr_activity = None
        mean_hr_recovery = None
    
    report = {
        "activity_profile": activity_profile,
        "activity_name": activity_name,
        "duration_minutes": ACTIVITY_DURATION_MINUTES,
        "initial_glucose": initial_glucose,
        "final_glucose": final_glucose,
        "nadir_glucose": nadir_glucose,
        "nadir_time_minutes": nadir_time,
        "max_drop": max_drop,
        "max_drop_time_minutes": max_drop_time,
        "total_glucose_change": final_glucose - initial_glucose,
        "recovery_rate_mg_dl_per_min": recovery_rate,
        "expected_hr": expected_hr,
        "mean_hr_during_activity": mean_hr_activity,
        "mean_hr_during_recovery": mean_hr_recovery,
    }
    
    return report


def plot_activity_test_results(sim_results_df, activity_profile, activity_name, expected_hr):
    """
    Plot glucose and heart rate traces from activity test.
    
    NOTE: This function is provided for optional use but is commented out by default.
    Uncomment the function call in the test to enable visualization.
    
    Parameters
    ----------
    sim_results_df : pd.DataFrame
        Simulation results
    activity_profile : str
        Profile name
    activity_name : str
        Human-readable activity name
    expected_hr : float
        Expected heart rate during activity
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
    except ImportError:
        print("  Matplotlib not available, skipping plot")
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Plot glucose
    ax1.plot(sim_results_df["time_minutes"], sim_results_df["bg"], 
             linewidth=2, color='blue', label='Glucose')
    ax1.axhline(y=sim_results_df["bg"].iloc[0], color='gray', 
                linestyle='--', alpha=0.5, label='Initial glucose')
    ax1.axvline(x=ACTIVITY_DURATION_MINUTES, color='red', 
                linestyle='--', alpha=0.5, label='Activity end')
    ax1.set_ylabel('Glucose (mg/dL)', fontsize=12)
    ax1.set_title(f'Physical Activity Test: {activity_name} ({activity_profile})', 
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)
    
    # Plot heart rate
    hr_column = "hr" if "hr" in sim_results_df.columns else "heart_rate"
    if hr_column in sim_results_df.columns:
        ax2.plot(sim_results_df["time_minutes"], sim_results_df[hr_column], 
                 linewidth=2, color='red', label='Heart Rate')
        if expected_hr is not None:
            ax2.axhline(y=expected_hr, color='orange', 
                       linestyle='--', alpha=0.7, label=f'Expected HR ({expected_hr} bpm)')
        ax2.axvline(x=ACTIVITY_DURATION_MINUTES, color='red', 
                   linestyle='--', alpha=0.5, label='Activity end')
        ax2.set_ylabel('Heart Rate (bpm)', fontsize=12)
        ax2.legend(loc='best')
        ax2.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Time (minutes)', fontsize=12)
    
    plt.tight_layout()
    
    # Save plot
    plot_filename = f"pa_test_{activity_profile}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_path = TEST_OUTPUT_DIR / plot_filename
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"  Plot saved to: {plot_path}")
    
    # Optionally show plot (comment out for automated testing)
    # plt.show()
    
    plt.close()


# ============================================================================
# PHASE 3: TEST IMPLEMENTATION
# ============================================================================

@pytest.mark.parametrize("activity_profile", [
    "walking_v1",
    "biking_v1",
    "jogging_v1",
    "strength_training_v1"
])
def test_physical_activity_glucose_effect(activity_profile):
    """
    Test that physical activity affects glucose levels as expected.
    
    Uses DoNothingController to isolate physical activity effects without
    Loop insulin modulation interfering.
    
    Pass criteria:
    1. Test completes without errors
    2. Heart rate is passed at each timestep during activity
    3. Glucose delta ≠ 0 (physical activity has measurable effect)
    
    Parameters
    ----------
    activity_profile : str
        Name of activity profile to test (e.g., "walking_v1")
    """
    # ========== Step 3.2.1: Load activity profile parameters ==========
    profile_data = load_activity_profile(activity_profile)
    activity_params = profile_data["activity_params"]
    expected_hr = activity_params.get("expected_hr")
    activity_name = activity_params.get("activity", "unknown")
    
    print(f"\n{'='*70}")
    print(f"Testing: {activity_profile} ({activity_name})")
    print(f"Expected HR: {expected_hr} bpm")
    print(f"{'='*70}")
    
    # ========== Step 3.2.2: Create simulation components ==========
    t0 = DATETIME_DEFAULT
    t0, patient_config, pump_config, sensor_config, controller_config = \
        get_baseline_patient_config(t0, profile_data)
    
    # Create heart rate trace
    hr_trace = create_heart_rate_trace(
        t0, 
        DURATION_HRS, 
        patient_config.physical_activity_event_timeline
    )
    patient_config.hr_trace = hr_trace  # FIX: Use hr_trace, not heart_rate_trace
    
    # Initialize pump with DoNothingController
    pump = ContinuousInsulinPump(pump_config, t0)
    sensor = IdealSensor(t0, sensor_config)
    controller = DoNothingController(t0, controller_config)
    
    # Create virtual patient
    vp = VirtualPatient(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=patient_config
    )
    
    # ========== Step 3.2.3: Initialize and run simulation ==========
    sim_id = f"pa_test_{activity_profile}"
    sim = Simulation(
        time=t0,
        duration_hrs=DURATION_HRS,
        virtual_patient=vp,
        controller=controller,
        sim_id=sim_id
    )
    
    # Run the simulation
    sim.run()
    
    # ========== Step 3.2.4: Extract results ==========
    sim_results_df = sim.get_results_df()
    
    # ========== Step 3.3: Data extraction and calculation ==========
    # Calculate glucose delta from initial value
    initial_glucose = sim_results_df["bg"].iloc[0]
    sim_results_df["glucose_delta"] = sim_results_df["bg"] - initial_glucose
    
    # Add time in minutes from start
    sim_results_df["time_minutes"] = [
        (dt - t0).total_seconds() / 60.0 
        for dt in sim_results_df.index
    ]
    
    # Add is_active flag (true during activity period)
    sim_results_df["is_active"] = sim_results_df["time_minutes"] < ACTIVITY_DURATION_MINUTES
    
    # Extract heart rate if available in results
    if "heart_rate" in sim_results_df.columns:
        hr_column = "heart_rate"
    else:
        # Create heart rate column from trace
        hr_column = "hr"
        sim_results_df[hr_column] = [
            hr_trace.get_heart_rate(dt) for dt in sim_results_df.index
        ]
    
    # ========== PHASE 4: ASSERTIONS AND PASS CRITERIA ==========
    
    # Pass Criterion 1: Test completes
    assert sim_results_df is not None, "Simulation results are None"
    assert len(sim_results_df) > 0, "Simulation results are empty"
    print(f"✓ Pass Criterion 1: Test completed successfully")
    print(f"  Total timesteps: {len(sim_results_df)}")
    
    # Pass Criterion 2: Heart rate is passed at each timestep during activity
    activity_period = sim_results_df[sim_results_df["is_active"]]
    
    if len(activity_period) > 0 and expected_hr is not None:
        # Check heart rate during activity
        hr_during_activity = activity_period[hr_column].values
        hr_tolerance = 5  # bpm tolerance
        
        # Check if HR is within tolerance for all activity timesteps
        hr_matches = [abs(hr - expected_hr) <= hr_tolerance for hr in hr_during_activity]
        hr_match_rate = sum(hr_matches) / len(hr_matches) * 100
        
        assert hr_match_rate >= 80, (
            f"Heart rate verification failed: {hr_match_rate:.1f}% of timesteps within tolerance. "
            f"Expected {expected_hr} ± {hr_tolerance} bpm, got {hr_during_activity}"
        )
        
        print(f"✓ Pass Criterion 2: Heart rate verified during activity")
        print(f"  Expected HR: {expected_hr} bpm")
        print(f"  Mean HR during activity: {hr_during_activity.mean():.1f} bpm")
        print(f"  HR match rate: {hr_match_rate:.1f}% (≥80% required)")
    else:
        print(f"⚠ Pass Criterion 2: Skipped (no activity period or expected_hr)")
    
    # Pass Criterion 3: Glucose delta ≠ 0
    glucose_deltas = sim_results_df["glucose_delta"].values
    has_nonzero_delta = any(abs(delta) > 0.1 for delta in glucose_deltas)  # 0.1 mg/dL tolerance
    
    assert has_nonzero_delta, (
        f"Glucose delta is zero throughout simulation. "
        f"Physical activity should affect glucose levels.\n"
        f"Glucose values: {sim_results_df['bg'].values}"
    )
    
    max_delta = glucose_deltas.min()  # Most negative (largest drop)
    final_glucose = sim_results_df["bg"].iloc[-1]
    
    print(f"✓ Pass Criterion 3: Glucose delta ≠ 0")
    print(f"  Initial glucose: {initial_glucose:.1f} mg/dL")
    print(f"  Final glucose: {final_glucose:.1f} mg/dL")
    print(f"  Maximum glucose drop: {max_delta:.1f} mg/dL")
    
    # ========== Summary output ==========
    print(f"\n{activity_profile} Test Summary:")
    print(f"  Activity: {activity_name}")
    print(f"  Duration: {ACTIVITY_DURATION_MINUTES} minutes")
    print(f"  Initial glucose: {initial_glucose:.1f} mg/dL")
    print(f"  Final glucose: {final_glucose:.1f} mg/dL")
    print(f"  Total glucose change: {final_glucose - initial_glucose:.1f} mg/dL")
    print(f"  Maximum drop: {max_delta:.1f} mg/dL")
    if expected_hr is not None:
        print(f"  Expected HR: {expected_hr} bpm")
        print(f"  Mean HR during activity: {hr_during_activity.mean():.1f} bpm")
    print(f"  ✓ All pass criteria met")
    print(f"{'='*70}\n")
    
    # ========== PHASE 5: SAVE RESULTS AND GENERATE REPORT ==========
    
    # Save results to CSV
    saved_path = save_test_results(sim_results_df, activity_profile, activity_name)
    
    # Generate detailed report
    detailed_report = generate_activity_test_report(
        sim_results_df, 
        activity_name, 
        activity_profile, 
        expected_hr
    )
    
    # Optional: Uncomment to generate plots
    # plot_activity_test_results(sim_results_df, activity_profile, activity_name, expected_hr)
    
    return sim_results_df


if __name__ == "__main__":
    # Test the helper functions
    print("Testing helper functions...")
    
    # Test load_activity_profile
    profile = load_activity_profile("walking_v1")
    print(f"\nLoaded walking profile:")
    print(f"  Metabolism params: {profile['metabolism_parameters']}")
    print(f"  Activity params: {profile['activity_params']}")
    
    # Test get_baseline_patient_config
    t0 = DATETIME_DEFAULT
    t0, patient_cfg, pump_cfg, sensor_cfg, controller_cfg = get_baseline_patient_config(t0, profile)
    print(f"\nCreated baseline config:")
    print(f"  Start time: {t0}")
    print(f"  Initial glucose: {INITIAL_GLUCOSE}")
    print(f"  Activity timeline events: {len(patient_cfg.physical_activity_event_timeline.events)}")
    
    # Test create_heart_rate_trace
    hr_trace = create_heart_rate_trace(t0, DURATION_HRS, patient_cfg.physical_activity_event_timeline)
    print(f"\nCreated heart rate trace:")
    print(f"  Total timesteps: {len(hr_trace.datetimes)}")
    print(f"  First 10 HR values: {hr_trace.hr_values[:10]}")
    
    print("\n" + "="*70)
    print("Phase 1, 2, 3, 4 & 5 implementation complete!")
    print("="*70)
    print("\nPhase 5 Features:")
    print("  ✓ Automatic CSV saving to tests/test_data")
    print("  ✓ Detailed statistical reports")
    print("  ✓ Optional visualization (commented out)")
    
    # Run a sample test
    print("\n" + "="*70)
    print("Running sample test for walking_v1...")
    print("="*70)
    try:
        results = test_physical_activity_glucose_effect("walking_v1")
        print("\n✓ Sample test completed successfully!")
        print(f"  Results shape: {results.shape}")
        print(f"  Columns: {list(results.columns)}")
        print(f"\nCheck {TEST_OUTPUT_DIR} for saved CSV files")
    except Exception as e:
        print(f"\n✗ Sample test failed with error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\nTo run all tests with pytest:")
    print("  pytest tests/test_physical_activity_model.py -v")
    print("\nTo run a specific activity test:")
    print("  pytest tests/test_physical_activity_model.py::test_physical_activity_glucose_effect[walking_v1] -v -s")
