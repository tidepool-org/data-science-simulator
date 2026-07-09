__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.models.measures import TempBasal, BasalRate
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump, OmnipodMissingPulses, Omnipod
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient


def test_continous_insulin_pump():

    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump

    # Test initialization expectations
    assert pump.basal_insulin_delivered_last_update == 0
    pump.init()
    assert pump.basal_insulin_delivered_last_update > 0
    assert pump.basal_insulin_delivered_last_update == pump.get_delivered_basal_insulin_since_update(update_interval_minutes=5)
    assert pump.get_delivered_basal_insulin_since_update(update_interval_minutes=5) < pump.get_delivered_basal_insulin_since_update(update_interval_minutes=6)

    assert pump.get_basal_rate() == BasalRate(0.3, "U/hr")
    assert pump.get_basal_rate() != BasalRate(0.3, "U/min")
    assert pump.get_basal_rate() != BasalRate(0.0, "U/hr")

    # Check validity of temp basals
    valid_temp_basal = TempBasal(t0, 0.0, 30, "U/hr")
    is_valid, message = pump.is_valid_temp_basal(valid_temp_basal)
    assert is_valid

    invalid_temp_basal = TempBasal(t0, -0.5, 30, "U/hr")
    is_valid, message = pump.is_valid_temp_basal(invalid_temp_basal)
    assert not is_valid

    invalid_temp_basal = TempBasal(t0, 0.5, 10, "U/h")
    is_valid, message = pump.is_valid_temp_basal(invalid_temp_basal)
    assert not is_valid

    invalid_temp_basal = TempBasal(t0, 0.3, 45, "U/h")
    is_valid, message = pump.is_valid_temp_basal(invalid_temp_basal)
    assert not is_valid

    # Actually set a temp basal to 0.0 and check state
    pump.set_temp_basal(valid_temp_basal)
    assert pump.has_active_temp_basal()

    assert pump.get_basal_rate() == TempBasal(t0, 0.0, 30, "U/hr")
    assert pump.get_basal_rate() != TempBasal(t0, 0.0, 30, "U/min")
    assert pump.get_basal_rate() != TempBasal(t0, 0.1, 30, "U/hr")
    assert pump.get_basal_rate() != TempBasal(t0, 0.0, 35, "U/hr")
    assert pump.get_basal_rate() != TempBasal(t0 + datetime.timedelta(minutes=5), 0.0, 30, "U/hr")

    # Update pump through 30 minutes of time and check state
    update_time_delta = datetime.timedelta(minutes=5)
    for _ in range(5):
        pump.update(pump.time + update_time_delta)
        assert pump.has_active_temp_basal()
        assert pump.basal_insulin_delivered_last_update == 0

    # Update to expected end of temp basal
    pump.update(pump.time + update_time_delta)
    assert not pump.has_active_temp_basal()

    # Set a higher temp basal
    temp_basal = TempBasal(pump.time, 0.6, 30, "U/hr")
    pump.set_temp_basal(temp_basal)

    # Update through 30 minutes and check state
    for _ in range(5):
        pump.update(pump.time + update_time_delta)
        assert pump.has_active_temp_basal()
        assert pump.basal_insulin_delivered_last_update == pump.get_delivered_basal_insulin_since_update()

    pump.update(pump.time + update_time_delta)
    assert not pump.has_active_temp_basal()

    # Overlapping temp basals
    # Set first temp basal
    temp_basal1 = TempBasal(pump.time, 0.6, 30, "U/hr")
    pump.set_temp_basal(temp_basal1)
    pump.update(pump.time + update_time_delta)
    assert pump.basal_insulin_delivered_last_update == 0.6 / 12

    # Set next temp basal
    temp_basal2 = TempBasal(pump.time, 0.2, 30, "U/hr")
    pump.set_temp_basal(temp_basal2)
    assert pump.get_basal_rate() == temp_basal2
    pump.update(pump.time + update_time_delta)
    assert pump.get_basal_rate() == temp_basal2
    assert pump.basal_insulin_delivered_last_update == 0.2 / 12

    # Run 2nd temp basal to expiration
    for _ in range(5):
        pump.update(pump.time + update_time_delta)
    assert not pump.has_active_temp_basal()


def test_omnipod():

    t0, vp = get_canonical_risk_patient(pump_class=Omnipod)
    pump = vp.pump
    assert pump.get_basal_rate() == BasalRate(0.3, "U/hr")

    # Test initialization prior to any updating, ie t0 setup behavior in simulation
    assert pump.basal_insulin_delivered_last_update == 0
    pump.init()
    assert pump.basal_insulin_delivered_last_update == 0  # No pulses for 0.3 rate in 5 minutes

    temp_basal = TempBasal(pump.time, 0.6, 30, "U/hr")
    pump.set_temp_basal(temp_basal)
    pump.init()
    assert pump.basal_insulin_delivered_last_update == 0.05


def test_omnipod_missing_pulses():

    t0, vp = get_canonical_risk_patient(pump_class=OmnipodMissingPulses)
    pump = vp.pump
    assert pump.get_basal_rate() == BasalRate(0.3, "U/hr")

    # Check delivered basal insulin over 1 hr with no expected missing pulses
    delivered_basal_insulin = 0
    update_time_delta = datetime.timedelta(minutes=5)
    for _ in range(12):
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert delivered_basal_insulin == 0.3  # all delivered

    # Check delivered insulin over 1 hr that will give no insulin
    delivered_basal_insulin = 0
    for _ in range(12):
        temp_basal = TempBasal(pump.time, 0.3, 30, "U/h")
        pump.set_temp_basal(temp_basal)
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert  delivered_basal_insulin == 0  # none delivered

    # Check delivered insulin over 1 hr at the boundary of 0.6
    delivered_basal_insulin = 0
    for _ in range(12):
        temp_basal = TempBasal(pump.time, 0.6, 30, "U/h")
        pump.set_temp_basal(temp_basal)
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert delivered_basal_insulin == 0.6  # all delivered

    # Check delivered insulin over 1 hr just across the boundary
    delivered_basal_insulin = 0
    for _ in range(12):
        temp_basal = TempBasal(pump.time, 0.7, 30, "U/h")
        pump.set_temp_basal(temp_basal)
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert delivered_basal_insulin == 0.6  # most delivered

    # Check delivered insulin over 1 hr further across the boundary
    delivered_basal_insulin = 0
    for _ in range(12):
        temp_basal = TempBasal(pump.time, 0.9, 30, "U/h")
        pump.set_temp_basal(temp_basal)
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert delivered_basal_insulin == 0.6  # most delivered

    # Check delivered insulin over 1 hr just across the boundary
    delivered_basal_insulin = 0
    for _ in range(12):
        temp_basal = TempBasal(pump.time, 1.1, 30, "U/h")
        pump.set_temp_basal(temp_basal)
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert delivered_basal_insulin == 0.6  # roughly half delivered


    # Check delivered insulin over 1 hr at the next boundary
    delivered_basal_insulin = 0
    for _ in range(12):
        temp_basal = TempBasal(pump.time, 1.2, 30, "U/h")
        pump.set_temp_basal(temp_basal)
        pump.update(pump.time + update_time_delta)
        delivered_basal_insulin += pump.basal_insulin_delivered_last_update

    assert delivered_basal_insulin == 1.2  # most delivered


def test_scheduled_basal_recording():
    """
    Test that scheduled basal deliveries are recorded in timeline.
    
    This test verifies that when no temp basal is active, the pump records
    scheduled basal delivery as basal events in the temp_basal_event_timeline.
    """
    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump
    
    # Initialize pump - this should record initial basal delivery
    pump.init()
    
    # Check that timeline has at least one event after init
    initial_event_count = len(pump.temp_basal_event_timeline.events)
    assert initial_event_count >= 1, "Init should record initial basal delivery"
    
    # Run updates for 30 minutes (6 updates) with no temp basal
    update_time_delta = datetime.timedelta(minutes=5)
    for i in range(6):
        pump.update(pump.time + update_time_delta)
    
    # Check that timeline has more events
    final_event_count = len(pump.temp_basal_event_timeline.events)
    assert final_event_count > initial_event_count, "Updates should add scheduled basal events"
    assert final_event_count >= 7, "Should have at least 7 events (1 init + 6 updates)"
    
    # Verify that the basal events have correct properties
    for event_time, basal_event in pump.temp_basal_event_timeline.events.items():
        # Should be marked as inactive (historical)
        assert not basal_event.active, "Scheduled basal events should be marked inactive"
        
        # Should have delivered_units set
        assert basal_event.delivered_units is not None, "Should have delivered_units"
        assert basal_event.delivered_units >= 0, "Delivered units should be non-negative"
        
        # Should match the scheduled basal rate
        scheduled_rate = pump.pump_config.basal_schedule.get_state().value
        expected_delivery = scheduled_rate * (5 / 60.0)  # 5 minutes worth
        assert abs(basal_event.delivered_units - expected_delivery) < 0.001, \
            f"Delivered units should match expected: {basal_event.delivered_units} vs {expected_delivery}"


def test_scheduled_basal_not_recorded_during_temp_basal():
    """
    Test that scheduled basal is NOT recorded when a temp basal is active.
    
    This verifies that we don't double-record insulin delivery.
    """
    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump
    pump.init()
    
    # Record event count after init
    event_count_after_init = len(pump.temp_basal_event_timeline.events)
    
    # Set a temp basal
    temp_basal = TempBasal(pump.time, 0.6, 30, "U/hr")
    pump.set_temp_basal(temp_basal)
    
    # The temp basal itself should be in the timeline
    event_count_after_temp = len(pump.temp_basal_event_timeline.events)
    assert event_count_after_temp == event_count_after_init + 1, \
        "Setting temp basal should add one event"
    
    # Run updates while temp basal is active
    update_time_delta = datetime.timedelta(minutes=5)
    for i in range(3):  # 15 minutes
        pump.update(pump.time + update_time_delta)
    
    # Event count should not increase during temp basal
    # (only the temp basal event itself, no scheduled basal events)
    event_count_during_temp = len(pump.temp_basal_event_timeline.events)
    assert event_count_during_temp == event_count_after_temp, \
        "Should not add scheduled basal events while temp basal is active"


def test_populate_historical_basal():
    """
    Test populating historical basal doses.
    
    This test verifies that populate_historical_basal_doses correctly
    backfills the pump timeline with historical scheduled basal deliveries.
    
    The method populates the full num_hours_history (8 hours by default)
    to represent the basal insulin that was in the patient's system,
    not just the time since simulation start. This is necessary for
    accurate IOB calculations when Loop is activated.
    """
    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump
    pump.init()
    
    # Advance time by 2 hours (simulating pre-Loop period)
    hours_to_advance = 2
    update_time_delta = datetime.timedelta(minutes=5)
    for _ in range(hours_to_advance * 12):  # 12 updates per hour
        pump.update(pump.time + update_time_delta)
    
    current_time = pump.time
    
    # Clear the timeline to simulate fresh Loop activation
    initial_event_count = len(pump.temp_basal_event_timeline.events)
    pump.temp_basal_event_timeline.events.clear()
    
    # Populate 8 hours of history
    # Even though only 2 hours of simulation have elapsed, this will populate
    # the full 8 hours to represent basal that was in the patient's system
    # before simulation started (necessary for accurate IOB calculation)
    pump.populate_historical_basal_doses(
        current_time=current_time,
        num_hours_history=8
    )
    
    # Should have populated approximately 8 hours worth of 5-minute intervals
    # 8 hours = 96 intervals of 5 minutes
    populated_event_count = len(pump.temp_basal_event_timeline.events)
    assert populated_event_count > 0, "Should populate some historical events"
    
    # Allow some tolerance for edge cases
    expected_count = 96  # 8 hours * 12 intervals per hour
    assert populated_event_count >= expected_count - 2, \
        f"Should populate approximately 8 hours. Got {populated_event_count}, expected ~{expected_count}"
    assert populated_event_count <= expected_count + 2, \
        f"Should not exceed 8 hours significantly. Got {populated_event_count}, expected ~{expected_count}"
    
    # Verify all events are in the past
    for event_time, basal_event in pump.temp_basal_event_timeline.events.items():
        assert event_time <= current_time, "All historical events should be in the past"
        assert not basal_event.active, "Historical events should be inactive"
        assert basal_event.delivered_units is not None, "Should have delivered_units"


def test_populate_historical_basal_full_8_hours():
    """
    Test populating 8 hours of historical basal when sufficient time has elapsed.
    """
    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump
    pump.init()
    
    # Advance time by 10 hours (more than the 8-hour lookback)
    hours_to_advance = 10
    update_time_delta = datetime.timedelta(minutes=5)
    for _ in range(hours_to_advance * 12):
        pump.update(pump.time + update_time_delta)
    
    current_time = pump.time
    
    # Clear the timeline
    pump.temp_basal_event_timeline.events.clear()
    
    # Populate 8 hours of history
    pump.populate_historical_basal_doses(
        current_time=current_time,
        num_hours_history=8
    )
    
    # Should have approximately 8 hours worth of 5-minute intervals
    # 8 hours = 96 intervals of 5 minutes
    populated_event_count = len(pump.temp_basal_event_timeline.events)
    
    # Allow some tolerance for edge cases
    expected_count = 96
    assert populated_event_count >= expected_count - 2, \
        f"Should populate approximately 8 hours of history. Got {populated_event_count}, expected ~{expected_count}"
    assert populated_event_count <= expected_count + 2, \
        f"Should not exceed 8 hours significantly. Got {populated_event_count}, expected ~{expected_count}"
    
    # Verify the time range of events
    event_times = list(pump.temp_basal_event_timeline.events.keys())
    earliest_event = min(event_times)
    latest_event = max(event_times)
    
    time_span = current_time - earliest_event
    # Should span approximately 8 hours
    assert time_span >= datetime.timedelta(hours=7, minutes=55), \
        f"Historical events should span ~8 hours. Got {time_span}"
    assert time_span <= datetime.timedelta(hours=8, minutes=5), \
        f"Historical events should not exceed 8 hours much. Got {time_span}"


