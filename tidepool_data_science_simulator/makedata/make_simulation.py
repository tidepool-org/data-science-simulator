__author__ = "Cameron Summers"

from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.models.controller import DoNothingController
from tidepool_data_science_simulator.models.simulation import Simulation

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient, DATETIME_DEFAULT
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV


def get_canonical_simulation(t0=DATETIME_DEFAULT,
                             patient_class=VirtualPatient,
                             patient_config=None,
                             pump_class=ContinuousInsulinPump,
                             pump_config=None,
                             sensor_class=IdealSensor,
                             sensor_config=None,
                             controller_class=DoNothingController,
                             controller_config=None,
                             multiprocess=False,
                             duration_hrs=8,
                             ):
    t0, vp = get_canonical_risk_patient(
        t0,
        patient_class=patient_class,
        patient_config=patient_config,
        pump_class=pump_class,
        pump_config=pump_config,
        sensor_class=sensor_class,
        sensor_config=sensor_config
    )
    t0, controller = get_canonical_controller(
        t0,
        controller_class=controller_class,
        controller_config=controller_config
    )

    sim = Simulation(
        time=t0,
        duration_hrs=duration_hrs,
        virtual_patient=vp,
        controller=controller,
        multiprocess=multiprocess,
        sim_id="Canonical Simulation"
    )

    return t0, sim
