__author__ = "Eden Grown-Haeberli"

from datetime import timedelta

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient, \
    get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.loop_risk_analysis_TLR315 import VirtualPatientCarbBolusAccept, LoopBolusRecMalfunctionDelay
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData, VirtualPatientRemovePump
from tidepool_data_science_simulator.models.measures import Carb, Bolus

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


def test_carb_delay():
    t0, patient_config = get_canonical_risk_patient_config(accept_prob=1.0)
    t0, pump_config = get_canonical_risk_pump_config()
    t0, controller_config = get_canonical_controller_config()
    patient_config.min_bolus_rec_threshold = 0.5

    carb = Carb(20, "g", 180)
    reported_carb_time = t0 + timedelta(minutes=30)
    pump_config.carb_event_timeline.add_event(reported_carb_time, carb, input_time=reported_carb_time)

    # User eats Carb at a different time than was reported.
    actual_carb_time = reported_carb_time + timedelta(minutes=30)
    patient_config.carb_event_timeline.add_event(actual_carb_time, carb)

    t0, sim = get_canonical_simulation(
        t0=t0,
        patient_config=patient_config,
        patient_class=VirtualPatientCarbBolusAccept,
        pump_config=pump_config,
        controller_class=LoopController,
        multiprocess=True,
        duration_hrs=8,
    )

    sim.run()

    assert sim.virtual_patient.carb_event_timeline.get_event(actual_carb_time) == carb
    assert sim.virtual_patient.pump.carb_event_timeline.get_event(reported_carb_time) == carb


def test_bolus_delay():
    t0, patient_config = get_canonical_risk_patient_config(accept_prob=1.0)
    t0, pump_config = get_canonical_risk_pump_config()
    t0, controller_config = get_canonical_controller_config()

    assert patient_config.recommendation_accept_prob == 1.0
    patient_config.min_bolus_rec_threshold = 0.5

    carb = Carb(20, "g", 180)
    carb_time = t0

    pump_config.carb_event_timeline.add_event(carb_time, carb)
    patient_config.carb_event_timeline.add_event(carb_time, carb)

    controller_config.bolus_rec_delay_minutes = 30

    t0, sim = get_canonical_simulation(
        t0=t0,
        patient_config=patient_config,
        pump_config=pump_config,
        patient_class=VirtualPatientCarbBolusAccept,
        controller_class=LoopBolusRecMalfunctionDelay,
        controller_config=controller_config,
        multiprocess=True,
        duration_hrs=8,
    )

    bolus_time = carb_time + timedelta(minutes=35)
    sim.run(early_stop_datetime=bolus_time)

    assert sim.virtual_patient.bolus_event_timeline.get_event(bolus_time) == Bolus(0.95, "U")
    assert sim.virtual_patient.pump.bolus_event_timeline.get_event(carb_time+timedelta(minutes=5)) == Bolus(0.95, "U")
