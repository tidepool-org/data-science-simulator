__author__ = "Cameron Summers"

import os
import subprocess

import numpy as np
import pandas as pd

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump, Omnipod, OmnipodMissingPulses
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient_config, get_canonical_risk_pump_config

from tidepool_data_science_simulator.models.measures import Bolus, Carb
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline

REGRESSION_COMMIT = "35cbbb4"


def build_test_sims_ORIG():

    controllers = [
        DoNothingController,
        LoopController,
    ]

    sims = {}
    for controller in controllers:

        t0, patient_config = get_canonical_risk_patient_config()
        patient_config.bolus_event_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
        patient_config.carb_event_timeline = CarbTimeline([t0], [Carb(20.0, "g", 180)])

        t0, pump_config = get_canonical_risk_pump_config(t0)
        pump_config.bolus_event_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
        pump_config.carb_event_timeline = CarbTimeline([t0], [Carb(40.0, "g", 180)])

        t0, sim = get_canonical_simulation(
            patient_config=patient_config,
            sensor_class=IdealSensor,
            pump_config=pump_config,
            pump_class=ContinuousInsulinPump,
            controller_class=controller,
            duration_hrs=8,
        )
        sim_id = controller.get_classname()
        sims[sim_id] = sim

    return sims


def build_test_sims():

    controllers = [
        DoNothingController,
        LoopController,
        LoopControllerDisconnector
    ]

    sensors = [
        IdealSensor,
        NoisySensor
    ]

    virtual_patients = [
        VirtualPatient,
        VirtualPatientModel
    ]

    pumps = [
        ContinuousInsulinPump,
        Omnipod,
        OmnipodMissingPulses
    ]


def test_regression():
    """
    Test the output of the canonical simulation to saved values from a standard commit.
    """
    sims = build_test_sims_ORIG()

    result_compare_dir = "tests/data/regression/commit-{}/".format(REGRESSION_COMMIT)

    for sim_id, sim in sims.items():
        sim.run()
        results_df = sim.get_results_df()

        result_df_compare = pd.read_csv(os.path.join(result_compare_dir, "{}.csv".format(sim_id)),
                                        index_col="time")

        for col in list(result_df_compare.columns):
            assert np.sum(np.abs(results_df[col] - result_df_compare[col])) < 1e-6


def make_regression():
    """
    Make a regression test with the current version of the code.

    This should match exactly the output of the test_regression for the same commit.
    """

    current_commit = subprocess.check_output(["git", "describe", "--always"]).strip().decode("utf-8")
    this_dir = os.path.dirname(os.path.abspath(__file__))

    sims = build_test_sims_ORIG()

    save_dir = os.path.join(this_dir, "data/regression/commit-{}/".format(current_commit))
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    else:
        raise Exception("Save dir exists: {}".format(save_dir))

    for sim_id, sim in sims.items():
        sim.run()
        results_df = sim.get_results_df()

        save_path = os.path.join(save_dir, "{}.csv".format(sim_id))
        results_df.to_csv(save_path)


if __name__ == "__main__":

    make_regression()
