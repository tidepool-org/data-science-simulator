__author__ = "Cameron Summers"

"""
https://tidepool.atlassian.net/browse/LOOP-1423

This code runs the analysis for capping the rate of change computation in the
Pyloopkit algorithm.

Update: Oct 1, 2020

Analysis results pointed to a value of 4.0 mg/dL / minute for the rate cap.
"""

import os
import logging

import numpy as np


print("***************** Warning: Overriding Pyloopkit with Local Copy ***************************")
import sys

# NOTE This uses Pyloopkit commit: https://github.com/tidepool-org/PyLoopKit/commit/edb52bd1e733f0f1d5591a2c36d917395f20ee44
local_pyloopkit_path = "/Users/csummers/dev/PyLoopKit/"
if not os.path.isdir(local_pyloopkit_path):
    local_pyloopkit_path = "/mnt/cameronsummers/dev/PyLoopKit/"
assert os.path.isdir(local_pyloopkit_path)
sys.path.insert(0, local_pyloopkit_path)

# Setup Logging
logger = logging.getLogger(__name__)

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import ActionTimeline
from tidepool_data_science_simulator.models.controller import LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, \
    get_pump_config_from_patient, get_variable_risk_patient_config
from tidepool_data_science_simulator.utils import get_sim_results_save_dir


from numpy.random import RandomState

SEED = 1234567890
ONCE_PER_WEEK_PROB = 1.0 / (12 * 24 * 7)  # 12 readings/hr * hr/day * day/wk = readings/wk
NO_RATE_CAP_VALUE = 1e12


def get_new_random_state(seed=SEED):
    return RandomState(seed)


def build_rate_cap_sims(test_run=True):

    logger.debug("Random Seed: {}".format(SEED))

    num_patients = 75
    rate_caps = list(np.arange(1.0, 11, 1))
    duration_hrs = 4 * 7 * 24

    if test_run:
        num_patients = 1
        rate_caps = [3.0]
        duration_hrs = 8

    logger.debug("Running {} patients. {} rate caps. {} hours".format(num_patients, len(rate_caps), duration_hrs))

    # Setup Patients
    patient_random_state = get_new_random_state()  # Single instance generates different patients
    sims = {}
    for i in range(num_patients):

        # Setup patient config
        t0, patient_config = get_variable_risk_patient_config(patient_random_state)

        patient_config.recommendation_accept_prob = patient_random_state.uniform(0.8, 0.99)
        patient_config.min_bolus_rec_threshold = patient_random_state.uniform(0.4, 0.6)
        patient_config.correct_bolus_bg_threshold = patient_random_state.uniform(140, 190)  # no impact
        patient_config.correct_bolus_delay_minutes = patient_random_state.uniform(20, 40)  # no impact
        patient_config.correct_carb_bg_threshold = patient_random_state.uniform(70, 90)
        patient_config.correct_carb_delay_minutes = patient_random_state.uniform(5, 15)
        patient_config.carb_count_noise_percentage = patient_random_state.uniform(0.1, 0.25)
        patient_config.report_bolus_probability = patient_random_state.uniform(1.0, 1.0)  # no impact
        patient_config.report_carb_probability = patient_random_state.uniform(0.95, 1.0)

        patient_config.prebolus_minutes_choices = [0]
        patient_config.carb_reported_minutes_choices = [0]

        t0, pump_config = get_pump_config_from_patient(patient_random_state, patient_config=patient_config,
                                                       risk_level=0.0)

        # Setup sensor config
        t0, baseline_sensor_config = get_canonical_sensor_config()
        baseline_sensor_config.std_dev = 1.0
        baseline_sensor_config.spurious_prob = 0.0
        baseline_sensor_config.spurious_outage_prob = 0.0
        baseline_sensor_config.time_delta_crunch_prob = 0.0
        baseline_sensor_config.name = "Clean"

        t0, noisy_sensor_config = get_canonical_sensor_config()
        noisy_sensor_config.std_dev = 1.0
        noisy_sensor_config.spurious_prob = 3.5 * ONCE_PER_WEEK_PROB  # spurious events
        noisy_sensor_config.spurious_outage_prob = 0.8  # data outage
        noisy_sensor_config.time_delta_crunch_prob = 3.5 * ONCE_PER_WEEK_PROB  # small time delta
        noisy_sensor_config.bg_spurious_error_delta_mgdl_range = [60, 150]
        noisy_sensor_config.not_working_time_minutes_range = [10, 45]
        noisy_sensor_config.cgm_offset_minutes_range = [2, 4.99]
        noisy_sensor_config.name = "Noisy"

        loop_connect_prob = patient_random_state.uniform(0.8, 0.99)

        # ===== Setup Baseline Simulations =====
        for sensor_config in [
            baseline_sensor_config,
            noisy_sensor_config
        ]:
            sim_random_state = get_new_random_state(seed=i)

            sensor = NoisySensor(time=t0, sensor_config=sensor_config, random_state=sim_random_state)
            pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

            vp = VirtualPatientModel(
                time=t0,
                pump=pump,
                sensor=sensor,
                metabolism_model=SimpleMetabolismModel,
                patient_config=patient_config,
                random_state=sim_random_state,
                id=i
            )

            t0, controller_config = get_canonical_controller_config()
            controller_config.controller_settings["max_physiologic_slope"] = NO_RATE_CAP_VALUE
            controller = LoopControllerDisconnector(time=t0,
                                                    controller_config=controller_config,
                                                    connect_prob=loop_connect_prob,
                                                    random_state=sim_random_state)
            controller.name = "PyloopKit_BG_Change_Max={}".format(NO_RATE_CAP_VALUE)

            # Setup Sims
            sim_id = "{}_{}_{}".format(vp.name, controller.name, sensor_config.name)
            sim = Simulation(
                time=t0,
                duration_hrs=duration_hrs,  # 4 weeks
                virtual_patient=vp,
                sim_id=sim_id,
                controller=controller,
                multiprocess=True,
                random_state=sim_random_state
            )
            sims[sim_id] = sim

        # ===== Setup Risk Mitigation Controllers =====
        for max_rate in rate_caps:
            sim_random_state = get_new_random_state(seed=i)

            sensor = NoisySensor(time=t0, sensor_config=noisy_sensor_config, random_state=sim_random_state)
            pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

            vp = VirtualPatientModel(
                time=t0,
                pump=pump,
                sensor=sensor,
                metabolism_model=SimpleMetabolismModel,
                patient_config=patient_config,
                random_state=sim_random_state,
                id=i
            )

            t0, controller_config = get_canonical_controller_config()
            controller_config.controller_settings["max_physiologic_slope"] = max_rate
            controller = LoopControllerDisconnector(time=t0,
                                                    controller_config=controller_config,
                                                    connect_prob=loop_connect_prob,
                                                    random_state=sim_random_state)
            controller.name = "PyloopKit_BG_Change_Max={}".format(max_rate)

            # Setup Sims
            sim_id = "{}_{}_{}".format(vp.name, controller.name, noisy_sensor_config.name)
            sim = Simulation(
                time=t0,
                duration_hrs=duration_hrs,  # 4 weeks
                virtual_patient=vp,
                sim_id=sim_id,
                controller=controller,
                multiprocess=True,
                random_state=sim_random_state
            )
            sims[sim_id] = sim

    return sims


if __name__ == "__main__":

    test_run = True

    results_dir = "./"
    save_results = False
    if not test_run:
        results_dir = get_sim_results_save_dir()
        save_results = True

    sims = build_rate_cap_sims(test_run=test_run)
    run_simulations(sims,
                    save_dir=results_dir,
                    save_results=save_results,
                    num_procs=10)

