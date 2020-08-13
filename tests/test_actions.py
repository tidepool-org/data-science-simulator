__author__ = "Eden Grown-Haeberli"

from datetime import timedelta

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel


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


def test_virtual_patient_delete_with_scenario_file():

    scenario_csv_filepath = "tests/data/Scenario-0-simulation-template - inputs.tsv"
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    controller = LoopController(
        time=t0,
        controller_config=sim_parser.get_controller_config(),
    )
    controller.num_hours_history = 8  # Force 8 hours to look for historical boluses

    pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())
    sensor = IdealSensor(time=t0, sensor_config=sim_parser.get_sensor_config())

    vp = VirtualPatient(
        time=t0,
        pump=pump,
        sensor=sensor,
        metabolism_model=SimpleMetabolismModel,
        patient_config=sim_parser.get_patient_config(),
    )

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
            duration_hrs=8.0,
            virtual_patient=vp,
            controller=controller
    )

    before_action_time = action_time - timedelta(minutes=5)
    simulation.run(early_stop_datetime=before_action_time)

    assert len(vp.pump.temp_basal_event_timeline.get_recent_event_times(action_time, num_hours_history=0.5)) != 0
    assert len(vp.pump.bolus_event_timeline.get_recent_event_times(action_time, num_hours_history=0.5)) != 0

    simulation.run(early_stop_datetime=action_time)

    assert len(vp.pump.temp_basal_event_timeline.get_recent_event_times(action_time, num_hours_history=0.5)) == 0
    assert len(vp.pump.bolus_event_timeline.get_recent_event_times(action_time, num_hours_history=0.5)) == 0

    after_action_time = action_time + timedelta(minutes=30)
    simulation.run(early_stop_datetime=after_action_time)

    # We've gone 30 minutes past the delete event to make sure the
    assert len(vp.pump.temp_basal_event_timeline.get_recent_event_times(action_time, num_hours_history=0.5)) == 0
    assert len(vp.pump.bolus_event_timeline.get_recent_event_times(action_time, num_hours_history=0.5)) == 0

    assert len(vp.pump.temp_basal_event_timeline.get_recent_event_times(after_action_time, num_hours_history=0.5)) != 0
    assert len(vp.pump.bolus_event_timeline.get_recent_event_times(after_action_time, num_hours_history=0.5)) != 0
