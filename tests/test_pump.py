__author__ = "Cameron Summers"

import datetime

from tidepool_data_science_simulator.models.measures import TempBasal, BasalRate
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump, OmnipodMissingPulses
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










