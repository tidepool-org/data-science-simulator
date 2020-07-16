__author__ = "Eden Grown-Haeberli"

from datetime import timedelta

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_patient import DATETIME_DEFAULT
from tidepool_data_science_simulator.models.controller import LoopController


def test_simulation():

    t0, sim = get_canonical_simulation(controller_class=LoopController)

    # Initialization checks
    assert sim.time == DATETIME_DEFAULT
    assert sim.duration_hrs == 8
    assert len(sim.simulation_results) == 1

    # Run checks
    sim.run(early_stop_datetime=t0+timedelta(minutes=30))
    assert sim.virtual_patient.bolus_event_timeline == sim.virtual_patient.pump.bolus_event_timeline
    assert sim.virtual_patient.bolus_event_timeline != sim.controller.carb_event_timeline
    assert sim.virtual_patient.carb_event_timeline == sim.virtual_patient.pump.carb_event_timeline
    assert sim.virtual_patient.carb_event_timeline != sim.controller.carb_event_timeline
    assert sim.virtual_patient.pump.temp_basal_event_timeline == sim.controller.temp_basal_event_timeline

