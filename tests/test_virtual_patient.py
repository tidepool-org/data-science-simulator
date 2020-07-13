__author__ = "Cameron Summers"

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.models.events import ActionTimeline, BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import BasalRate, InsulinSensitivityFactor

def test_patient():
    t0, vp = get_canonical_risk_patient()

    # Test initialization expectations
    vp.init()
    # assert vp.iob_current == vp.iob_init[0]
    assert len(vp.bg_prediction) != 0
    assert len(vp.iob_prediction) != 0

    assert isinstance(vp.bolus_event_timeline, BolusTimeline)
    assert isinstance(vp.carb_event_timeline, CarbTimeline)
    assert isinstance(vp.action_timeline, ActionTimeline)

    # Check state validity
    vp_state = vp.get_state()
    assert vp.bg_current == 110
    assert vp_state.sbr == BasalRate(0.3, "mg/dL")
    assert vp_state.isf == InsulinSensitivityFactor(150.0, "mg/dL/U")

    # TODO: Check patient updating
