__author__ = "Cameron Summers"

"""
https://tidepool.atlassian.net/browse/LOOP-1732

Simulations for exploring impact of mismatched biological and Loop settings. Specifically,
the Tidepool guardrails are to be examined for their impact on safety.
"""

import os
import logging
import datetime

import numpy as np

# Setup Logging
logger = logging.getLogger(__name__)

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.models.simulation import Simulation, TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.controller import LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.measures import TargetRange

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, \
    get_pump_config_from_patient, get_variable_risk_patient_config, get_pump_config_from_aace_settings
from tidepool_data_science_simulator.makedata.make_icgm_patients import (
    get_patients_by_age, get_icgm_patient_config
)
from tidepool_data_science_simulator.utils import get_sim_results_save_dir

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


from numpy.random import RandomState

SEED = 1234567890


def get_new_random_state(seed=SEED):
    return RandomState(seed)


def build_gsl_tr_lower_sims(test_run=True):
    """
    Create sims for comparing guardrail boudnaries of Glucose Safety Limi and
    Target Range Lower Bound.
    """

    logger.debug("Random Seed: {}".format(SEED))

    sim_duration_hrs = 4 * 7 * 24
    icgm_patients = get_patients_by_age(min_age=14, max_age=1e12)

    pump_settings_grid = [
        {
            "target_range_lower": max(gsl, target),  # A guardrail specification
            "target_range_upper": target,
            "glucose_safety_limit": gsl
        }
        for gsl in [50, 55, 60, 65, 70, 75, 80, 85]
        for target in [87 + i*5 for i in range(10)]
    ]

    if test_run:
        sim_duration_hrs = 48
        pump_settings_grid = [
            {
                "target_range_lower": max(gsl, t_lower),
                "target_range_upper": t_upper,
                "glucose_safety_limit": gsl
            }
            for gsl in [70]
            for t_lower, t_upper in [(90, 90), (80, 100)]#, (110, 110), (100, 120)]
        ]
        icgm_patients = icgm_patients[0:1]

    logger.debug("Running {} patients. {} hours".format(len(icgm_patients), sim_duration_hrs))

    # Setup Patients
    patient_random_state = get_new_random_state()  # Single instance generates different patients
    sims = {}

    for vp_idx, icgm_patient_obj in enumerate(icgm_patients):

        # Setup patient config
        t0, patient_config = get_icgm_patient_config(icgm_patient_obj, patient_random_state)

        patient_config.recommendation_accept_prob = 0.8  # patient_random_state.uniform(0.8, 0.99)
        patient_config.min_bolus_rec_threshold = patient_random_state.uniform(0.4, 0.6)
        patient_config.correct_bolus_bg_threshold = patient_random_state.uniform(140, 190)  # no impact
        patient_config.correct_bolus_delay_minutes = patient_random_state.uniform(20, 40)  # no impact
        patient_config.correct_carb_bg_threshold = patient_random_state.uniform(70, 90)
        patient_config.correct_carb_delay_minutes = patient_random_state.uniform(5, 15)
        patient_config.carb_count_noise_percentage = patient_random_state.uniform(0.2, 0.25)
        patient_config.report_bolus_probability = patient_random_state.uniform(1.0, 1.0)  # no impact
        patient_config.report_carb_probability = patient_random_state.uniform(0.95, 1.0)

        patient_config.prebolus_minutes_choices = [0]
        patient_config.carb_reported_minutes_choices = [0]

        # t0, pump_config = get_pump_config_from_patient(patient_random_state,
        #                                                patient_config=patient_config,
        #                                                risk_level=3.0,
        #                                                t0=t0)

        logger.debug("Patient Age: {}".format(icgm_patient_obj.age))
        t0, pump_config = get_pump_config_from_aace_settings(patient_random_state,
                                                             patient_weight=icgm_patient_obj.sample_weight_kg_by_age(random_state=patient_random_state),
                                                             patient_tdd=icgm_patient_obj.sample_total_daily_dose_by_age(random_state=patient_random_state),
                                                             risk_level=0,
                                                             t0=t0
                                                             )
        # Setup sensor config
        t0, sensor_config = get_canonical_sensor_config()
        sensor_config.std_dev = 2.0
        sensor_config.spurious_prob = 0.0
        sensor_config.spurious_outage_prob = 0.0
        sensor_config.time_delta_crunch_prob = 0.0
        sensor_config.name = "Clean"

        loop_connect_prob = patient_random_state.uniform(0.8, 0.95)

        for i, params in enumerate(pump_settings_grid):
            sim_random_state = get_new_random_state(seed=vp_idx)

            new_target_range_schedule = \
                TargetRangeSchedule24hr(
                    t0,
                    start_times=[datetime.time(0, 0, 0)],
                    values=[TargetRange(params["target_range_lower"], params["target_range_upper"], "mg/dL")],
                    duration_minutes=[1440]
                )
            pump_config.target_range_schedule = new_target_range_schedule

            sensor = NoisySensor(time=t0, sensor_config=sensor_config, random_state=sim_random_state)
            pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)

            vp = VirtualPatientModel(
                time=t0,
                pump=pump,
                sensor=sensor,
                metabolism_model=SimpleMetabolismModel,
                patient_config=patient_config,
                random_state=sim_random_state,
                id=icgm_patient_obj.get_patient_id()
            )

            t0, controller_config = get_canonical_controller_config()
            controller_config.controller_settings["suspend_threshold"] = params["glucose_safety_limit"]
            controller_config.controller_settings["max_basal_rate"] = icgm_patient_obj.get_basal_rate() * 4

            controller = LoopControllerDisconnector(time=t0,
                                                    controller_config=controller_config,
                                                    connect_prob=loop_connect_prob,
                                                    random_state=sim_random_state)

            # Setup Sims
            sim_id = "{}_tr={}_gsl={}".format(vp.name, params["target_range_upper"], params["glucose_safety_limit"])
            sim = Simulation(
                time=t0,
                duration_hrs=sim_duration_hrs,  # 4 weeks
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
    save_results = True

    results_dir = "."
    if save_results:
        results_dir = get_sim_results_save_dir("guardrails_gsl_tr")

    sims = build_gsl_tr_lower_sims(test_run=test_run)

    logger.info("Starting to run {} sims.".format(len(sims)))
    all_results = run_simulations(sims,
                    save_dir=results_dir,
                    save_results=save_results,
                    num_procs=10)

    plot_sim_results(all_results)

