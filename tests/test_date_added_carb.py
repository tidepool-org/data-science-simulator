#!/usr/bin/env python3
"""
Test script for carb date_added functionality.

This script demonstrates how the new date_added field works for modeling
scenarios where users log meals at different times than consumption.
"""

import datetime
import sys
import os

# Add the parent directory to path so we can import simulator modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tidepool_data_science_simulator.models.measures import Carb
from tidepool_data_science_simulator.models.events import CarbTimeline


def test_basic_carb_creation():
    """Test creating Carb objects with and without date_added"""
    print("=" * 80)
    print("TEST 1: Basic Carb Creation")
    print("=" * 80)

    # Create a carb without date_added (traditional behavior)
    carb1 = Carb(value=60, units="g", duration_minutes=180)
    print(f"✓ Created carb without date_added: {carb1.value}g")
    print(f"  date_added = {carb1.date_added} (None = logged at consumption time)")

    # Create a carb with date_added
    consumption_time = datetime.datetime(2024, 1, 15, 12, 0, 0)
    logged_time = datetime.datetime(2024, 1, 15, 11, 50, 0)  # Pre-logged 10 min early

    carb2 = Carb(value=60, units="g", duration_minutes=180, date_added=logged_time)
    print(f"\n✓ Created carb with date_added: {carb2.value}g")
    print(f"  date_added = {carb2.date_added} (logged 10 min before consumption)")

    print("\n✅ PASSED: Carb objects created successfully\n")


def test_carb_timeline_filtering():
    """Test that CarbTimeline.get_loop_inputs properly filters based on date_added"""
    print("=" * 80)
    print("TEST 2: CarbTimeline Filtering")
    print("=" * 80)

    # Set up times
    base_time = datetime.datetime(2024, 1, 15, 12, 0, 0)

    # Scenario: Three meals with different logging patterns
    carb_times = [
        base_time,  # Meal 1: Logged immediately (12:00)
        base_time + datetime.timedelta(hours=3),  # Meal 2: Pre-logged (15:00, logged at 14:50)
        base_time + datetime.timedelta(hours=6),  # Meal 3: Delayed logging (18:00, logged at 19:00)
    ]

    carbs = [
        Carb(60, "g", 180, date_added=None),  # Meal 1: immediate
        Carb(45, "g", 180, date_added=carb_times[1] - datetime.timedelta(minutes=10)),  # Meal 2: pre-logged
        Carb(80, "g", 180, date_added=carb_times[2] + datetime.timedelta(hours=1)),  # Meal 3: delayed
    ]

    # Create timeline and add events with proper input_time
    timeline = CarbTimeline()
    for carb_time, carb in zip(carb_times, carbs):
        input_time = carb.date_added if carb.date_added is not None else carb_time
        timeline.add_event(carb_time, carb, input_time=input_time)

    # Test at different simulation times
    test_times = [
        ("12:00", base_time, [60], "Only Meal 1 (logged immediately)"),
        ("14:50", base_time + datetime.timedelta(hours=2, minutes=50), [60, 45], "Meals 1 & 2 (Meal 2 just logged)"),
        ("15:00", base_time + datetime.timedelta(hours=3), [60, 45], "Meals 1 & 2 (Meal 2 consumed now)"),
        ("18:00", base_time + datetime.timedelta(hours=6), [60, 45], "Meals 1 & 2 only (Meal 3 not logged yet)"),
        ("19:00", base_time + datetime.timedelta(hours=7), [60, 45, 80], "All meals (Meal 3 finally logged)"),
    ]

    all_passed = True
    for time_label, sim_time, expected_values, description in test_times:
        carb_values, carb_start_times, carb_durations = timeline.get_loop_inputs(sim_time, num_hours_history=24)

        print(f"\nAt {time_label} ({description}):")
        print(f"  Expected carbs visible: {expected_values}")
        print(f"  Actually visible: {carb_values}")

        if carb_values == expected_values:
            print(f"  ✓ CORRECT")
        else:
            print(f"  ✗ FAILED - MISMATCH!")
            all_passed = False

    if not all_passed:
        print("\n❌ FAILED: CarbTimeline filtering has errors\n")
        return False
        
    print("\n✅ PASSED: CarbTimeline filtering works correctly\n")
    return True


def test_use_case_scenarios():
    """Demonstrate realistic use case scenarios"""
    print("=" * 80)
    print("TEST 3: Real-World Use Case Scenarios")
    print("=" * 80)

    base_time = datetime.datetime(2024, 1, 15, 12, 0, 0)

    # Scenario 1: Pre-logging (announcement)
    print("\nScenario 1: Pre-logging (meal announcement)")
    print("-" * 40)
    consumption_time = base_time
    logged_time = base_time - datetime.timedelta(minutes=10)
    carb = Carb(60, "g", 180, date_added=logged_time)
    print(f"  User logs 60g at {logged_time.strftime('%H:%M')}")
    print(f"  Actual consumption at {consumption_time.strftime('%H:%M')}")
    print(f"  → Controller sees entry 10 min before consumption")
    print(f"  → Can dose proactively for the meal")

    # Scenario 2: Delayed logging (forgot to log)
    print("\nScenario 2: Delayed logging (forgot to log)")
    print("-" * 40)
    consumption_time = base_time
    logged_time = base_time + datetime.timedelta(hours=1)
    carb = Carb(60, "g", 180, date_added=logged_time)
    print(f"  User eats 60g at {consumption_time.strftime('%H:%M')}")
    print(f"  Forgets to log until {logged_time.strftime('%H:%M')}")
    print(f"  → Carbs already absorbing for 1 hour before controller knows")
    print(f"  → Controller must play catch-up")

    # Scenario 3: Normal case (immediate logging)
    print("\nScenario 3: Normal case (immediate logging)")
    print("-" * 40)
    consumption_time = base_time
    carb = Carb(60, "g", 180, date_added=None)  # or date_added=consumption_time
    print(f"  User logs 60g at consumption time {consumption_time.strftime('%H:%M')}")
    print(f"  → Traditional behavior (no delay)")

    print("\n✅ PASSED: Use case scenarios demonstrated\n")
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 80)
    print("CARB DATE_ADDED FUNCTIONALITY TESTS")
    print("=" * 80 + "\n")

    try:
        # Track test results
        all_tests_passed = True
        
        # Run tests
        test_basic_carb_creation()

        if not test_carb_timeline_filtering():
            print("\n❌ TEST 2 FAILED - Stopping test suite\n")
            all_tests_passed = False
            return 1

        if not test_use_case_scenarios():
            print("\n❌ TEST 3 FAILED\n")
            all_tests_passed = False
            return 1

        if all_tests_passed:
            print("=" * 80)
            print("✅ ALL TESTS PASSED!")
            print("=" * 80)
            print("\nThe date_added functionality is working correctly.")
            print("\nTo use in JSON configs:")
            print("  - Add 'date_added' field to carb_entries in pump config")
            print("  - Leave it out of patient_model config (uses actual consumption time)")
            print("  - Format: same as start_time (MM/DD/YYYY HH:MM:SS)")
            print("\n")
            return 0
        else:
            print("\n❌ TEST SUITE FAILED - See errors above\n")
            return 1

    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())