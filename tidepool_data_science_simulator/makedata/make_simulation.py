__author__ = "Cameron Summers"

from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.models.controller import DoNothingController
from tidepool_data_science_simulator.models.simulation import Simulation

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV


def get_canonical_simulation(t0=None,
                             patient_class=VirtualPatient,
                             pump_class=ContinuousInsulinPump,
                             sensor_class=IdealSensor,
                             controller_class=DoNothingController,
                             multiprocess=False,
                             duration_hrs=8,
                             include_initial_events=False
                             ):
    t0, vp = get_canonical_risk_patient(t0, patient_class, pump_class, sensor_class, include_initial_events)
    t0, controller = get_canonical_controller(t0, controller_class)

    sim = Simulation(
        time=t0,
        duration_hrs=duration_hrs,
        virtual_patient=vp,
        controller=controller,
        multiprocess=multiprocess
    )

    return t0, sim
