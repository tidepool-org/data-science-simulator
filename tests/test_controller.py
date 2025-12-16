__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump


def test_loop_activation_populates_history():
    """
    Test that Loop activation triggers historical basal population.
    
    This test verifies that when LoopController.get_loop_recommendations is called
    for the first time, it populates the pump's historical dose timeline.
    """
    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump
    pump.init()
    
    # Advance simulation by 2 hours before activating Loop
    update_time_delta = datetime.timedelta(minutes=5)
    for _ in range(24):  # 2 hours
        pump.update(pump.time + update_time_delta)
    
    # Clear the pump timeline to simulate fresh state
    # (In reality, pump would have recorded ongoing basal, but we're testing
    # that populate_historical_basal_doses gets called)
    initial_event_count = len(pump.temp_basal_event_timeline.events)
    pump.temp_basal_event_timeline.events.clear()
    
    # Create controller
    t0, controller_config = get_canonical_controller_config(t0)
    controller = LoopController(
        time=t0,
        controller_config=controller_config
    )
    
    # Verify pump_history_initialized starts as False
    assert not controller.pump_history_initialized, \
        "pump_history_initialized should start as False"
    
    # First call to get_loop_recommendations should populate pump history
    controller.time = pump.time
    try:
        recommendations = controller.get_loop_recommendations(
            time=pump.time,
            virtual_patient=vp
        )
    except Exception as e:
        # Loop algorithm may fail without proper glucose data, but we're testing
        # that populate_historical_basal_doses was called
        pass
    
    # Verify pump_history_initialized is now True
    assert controller.pump_history_initialized, \
        "pump_history_initialized should be True after first call"
    
    # Verify pump timeline was populated
    populated_event_count = len(pump.temp_basal_event_timeline.events)
    assert populated_event_count > 0, \
        "Pump timeline should be populated after Loop activation"


def test_loop_activation_only_populates_once():
    """
    Test that pump history population only happens once.
    
    Verifies that subsequent calls to get_loop_recommendations do not
    re-populate the pump history.
    """
    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    pump = vp.pump
    pump.init()
    
    # Advance simulation by 2 hours
    update_time_delta = datetime.timedelta(minutes=5)
    for _ in range(24):
        pump.update(pump.time + update_time_delta)
    
    # Clear timeline
    pump.temp_basal_event_timeline.events.clear()
    
    # Create controller
    t0, controller_config = get_canonical_controller_config(t0)
    controller = LoopController(
        time=t0,
        controller_config=controller_config
    )
    
    # First call
    controller.time = pump.time
    try:
        controller.get_loop_recommendations(time=pump.time, virtual_patient=vp)
    except Exception:
        pass
    
    event_count_after_first_call = len(pump.temp_basal_event_timeline.events)
    
    # Second call
    try:
        controller.get_loop_recommendations(time=pump.time, virtual_patient=vp)
    except Exception:
        pass
    
    event_count_after_second_call = len(pump.temp_basal_event_timeline.events)
    
    # Event count should not change on second call
    assert event_count_after_first_call == event_count_after_second_call, \
        "Pump history should only be populated once"
    
    # Flag should remain True
    assert controller.pump_history_initialized, \
        "pump_history_initialized should remain True"
