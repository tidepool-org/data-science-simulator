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
from tidepool_data_science_simulator.makedata.make_patient import (
    get_canonical_risk_patient_config, get_canonical_risk_pump_config,
    get_variable_risk_patient_config, get_pump_config_from_patient
)

from tidepool_data_science_simulator.models.measures import Bolus, Carb
from tidepool_data_science_simulator.models.events import CarbTimeline, BolusTimeline

REGRESSION_COMMIT = "12c5839"


def build_test_sims_controllers():
    """
    Build a simple sim for controller classes for regression testing.

    Returns
    -------
    dict:
        Sim id to simulation map
    """
    controllers = [
        DoNothingController,
        LoopController,
        LoopControllerDisconnector
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


def build_test_sims_pumps():
    """
    Build a simple sim for pump classes for regression testing.

    Returns
    -------
    dict:
        Sim id to simulation map
    """
    pumps = [
        ContinuousInsulinPump,
        Omnipod,
        OmnipodMissingPulses
    ]

    sims = {}
    for pump in pumps:
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
            pump_class=pump,
            controller_class=LoopController,
            duration_hrs=8,
        )
        sim_id = pump.get_classname()
        sims[sim_id] = sim

    return sims


def build_test_sims_sensors():
    """
    Build a simple sim for sensor classes for regression testing.

    Returns
    -------
    dict:
        Sim id to simulation map
    """

    sensor_classes = [
        IdealSensor,
        NoisySensor
    ]

    sims = {}
    for sensor_class in sensor_classes:
        t0, patient_config = get_canonical_risk_patient_config()
        patient_config.bolus_event_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
        patient_config.carb_event_timeline = CarbTimeline([t0], [Carb(20.0, "g", 180)])

        t0, pump_config = get_canonical_risk_pump_config(t0)
        pump_config.bolus_event_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
        pump_config.carb_event_timeline = CarbTimeline([t0], [Carb(40.0, "g", 180)])

        t0, sim = get_canonical_simulation(
            patient_config=patient_config,
            sensor_class=sensor_class,
            pump_config=pump_config,
            pump_class=ContinuousInsulinPump,
            controller_class=LoopController,
            duration_hrs=8,
        )
        sim_id = sensor_class.get_classname()
        sims[sim_id] = sim

    return sims


def build_test_sims_patients():
    """
    Build a simple sim for patient classes for regression testing.

    Returns
    -------
    dict:
        Sim id to simulation map
    """

    virtual_patient_classes = [
        VirtualPatientModel
    ]

    sims = {}
    for patient_class in virtual_patient_classes:

        random_state = np.random.RandomState(0)

        t0, patient_config = get_variable_risk_patient_config(random_state)
        patient_config.bolus_event_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
        patient_config.carb_event_timeline = CarbTimeline([t0], [Carb(20.0, "g", 180)])

        patient_config.recommendation_accept_prob = 0.9
        patient_config.min_bolus_rec_threshold = 0.5
        patient_config.correct_bolus_bg_threshold = 180  # no impact
        patient_config.correct_bolus_delay_minutes = 30  # no impact
        patient_config.correct_carb_bg_threshold = 80
        patient_config.correct_carb_delay_minutes = 15
        patient_config.carb_count_noise_percentage = 0.15
        patient_config.report_bolus_probability = 1.0  # no impact
        patient_config.report_carb_probability = 0.95

        patient_config.prebolus_minutes_choices = [0]
        patient_config.carb_reported_minutes_choices = [0]

        t0, pump_config = get_pump_config_from_patient(random_state, patient_config, risk_level=1, t0=t0)
        pump_config.bolus_event_timeline = BolusTimeline([t0], [Bolus(2.0, "U")])
        pump_config.carb_event_timeline = CarbTimeline([t0], [Carb(40.0, "g", 180)])

        t0, sim = get_canonical_simulation(
            patient_config=patient_config,
            patient_class=patient_class,
            sensor_class=IdealSensor,
            pump_config=pump_config,
            pump_class=ContinuousInsulinPump,
            controller_class=LoopController,
            duration_hrs=8,
        )
        sim_id = patient_class.get_classname()
        sims[sim_id] = sim

    return sims


def build_all_test_sims():
    """
    Collect all the sims for regression testing.

    Returns
    -------
    dict:
        Sim id to simulation map
    """

    all_sims = {}

    controller_sims = build_test_sims_controllers()
    pump_sims = build_test_sims_pumps()
    sensor_sims = build_test_sims_sensors()
    patient_sims = build_test_sims_patients()

    all_sims.update(controller_sims)
    all_sims.update(pump_sims)
    all_sims.update(sensor_sims)
    all_sims.update(patient_sims)

    return all_sims


def test_regression():
    """
    Test the output of the canonical simulation to saved values from a standard commit.
    """
    sims = build_all_test_sims()

    result_compare_dir = "tests/test_data/regression/commit-{}/".format(REGRESSION_COMMIT)

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

    sims = build_all_test_sims()

    save_dir = os.path.join(this_dir, "test_data/regression/commit-{}/".format(current_commit))
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
