__author__ = "Eden Grown-Haeberli"

from datetime import timedelta

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient, \
    get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData, VirtualPatientRemovePump


#TODO: Make these rely on patient configs rather than patients?
def test_virtual_patient_delete():

    t0, vp = get_canonical_risk_patient(pump_class=ContinuousInsulinPump)
    action_time = t0 + timedelta(minutes=30)
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


def test_remove_pump():

    t0, patient_config = get_canonical_risk_patient_config()
    t0, controller_config = get_canonical_controller_config()

    patient_config.recommendation_accept_prob = 0.0

    remove_pump_time = t0 + timedelta(minutes=90)
    user_remove_pump_action = VirtualPatientRemovePump("Remove Pump")
    patient_config.action_timeline.add_event(remove_pump_time, user_remove_pump_action)

    _, patient = get_canonical_risk_patient(t0, patient_config=patient_config)

    controller = LoopController(
        time=t0,
        controller_config=controller_config
    )

    simulation = Simulation(
        time=t0,
        duration_hrs=6.0,
        virtual_patient=patient,
        controller=controller
    )

    simulation.run(early_stop_datetime=remove_pump_time)

    assert patient.pump is None


def test_add_pump():
    pass

def test_add_carbs():
    pass