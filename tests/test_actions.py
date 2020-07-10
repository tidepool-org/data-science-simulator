__author__ = "Eden Grown-Haeberli"

import datetime

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData


def test_virtual_patient_delete():

    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    action_time = t0 + datetime.timedelta(minutes=30)
    vp.action_timeline = ActionTimeline()
    vp.action_timeline.add_event(action_time, VirtualPatientDeleteLoopData("Deleted Insulin History"))

    _, controller_config = get_canonical_controller_config()

    controller = LoopController(
        time=t0,
        controller_config=controller_config
    )

    simulation = Simulation(
            time=t0,
            duration_hrs=6.0,
            virtual_patient=vp,
            controller=controller
    )

    simulation.run(early_stop_datetime=action_time)

    assert vp.pump.temp_basal_event_timeline.is_empty_timeline()
    assert vp.pump.bolus_event_timeline.is_empty_timeline()
