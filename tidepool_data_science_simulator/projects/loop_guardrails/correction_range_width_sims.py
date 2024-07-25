__author__ = "Shawn Foster"

"""
Simulations for exploring impact of mismatched biological and Loop settings. Specifically,
the correction range width is to be examined for its impact on safety when used with settings
that differ from physiologically true.
"""
import os
import numpy as np
import datetime
import logging

# Setup Logging
logger = logging.getLogger(__name__)

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.models.simulation import Simulation, BasalSchedule24hr, TargetRangeSchedule24hr, SettingSchedule24Hr
from tidepool_data_science_simulator.models.controller import LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.measures import TargetRange, BasalRate, CarbInsulinRatio, \
    InsulinSensitivityFactor

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, \
    get_pump_config_from_patient, compute_aace_settings_tmp, get_variable_risk_patient_config, \
    get_pump_config_from_aace_settings
from tidepool_data_science_simulator.makedata.make_icgm_patients import (
    get_patients_by_age, get_icgm_patient_config, get_icgm_patient_settings_objects
)
from tidepool_data_science_simulator.utils import get_sim_results_save_dir

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results

from numpy.random import RandomState

SEED = 1234567890


def get_new_random_state(seed=SEED):
    return RandomState(seed)


def adjust_pump_settings(pump_config, lower_deviation, upper_deviation, random_state):
    print(f"Lower deviation: {lower_deviation}, Upper deviation: {upper_deviation}")

    # Helper function to adjust a single value
    def adjust_value(value, random_state):
        adjustment = random_state.uniform(lower_deviation, upper_deviation)
        adjusted_value = value * adjustment
        print(f"Adjusting value {value} with factor {adjustment}, result: {adjusted_value}")
        return adjusted_value

    # Adjust basal rate
    original_basal_schedule = pump_config.basal_schedule
    adjusted_schedule = {}
    for (start_time, end_time), basal_rate in original_basal_schedule.schedule.items():
        adjusted_value = adjust_value(basal_rate.value, random_state)
        adjusted_schedule[(start_time, end_time)] = BasalRate(adjusted_value, basal_rate.units)

    pump_config.basal_schedule = BasalSchedule24hr(
        time=pump_config.basal_schedule.time,
        start_times=[start for start, _ in adjusted_schedule.keys()],
        values=list(adjusted_schedule.values()),
        duration_minutes=[original_basal_schedule.schedule_durations[k] for k in adjusted_schedule.keys()]
    )

    # Adjust carb ratio
    original_carb_ratio_schedule = pump_config.carb_ratio_schedule
    adjusted_schedule = {}
    for (start_time, end_time), carb_ratio in original_carb_ratio_schedule.schedule.items():
        adjusted_value = adjust_value(carb_ratio.value, random_state)
        adjusted_schedule[(start_time, end_time)] = CarbInsulinRatio(adjusted_value, carb_ratio.units)

    pump_config.carb_ratio_schedule = SettingSchedule24Hr(
        time=original_carb_ratio_schedule.time,
        name=original_carb_ratio_schedule.name,
        start_times=[start for start, _ in adjusted_schedule.keys()],
        values=list(adjusted_schedule.values()),
        duration_minutes=[original_carb_ratio_schedule.schedule_durations[k] for k in adjusted_schedule.keys()]
    )

    # Adjust insulin sensitivity factor
    original_isf_schedule = pump_config.insulin_sensitivity_schedule
    adjusted_schedule = {}
    for (start_time, end_time), isf in original_isf_schedule.schedule.items():
        adjusted_value = adjust_value(isf.value, random_state)
        adjusted_schedule[(start_time, end_time)] = InsulinSensitivityFactor(adjusted_value, isf.units)

    pump_config.insulin_sensitivity_schedule = SettingSchedule24Hr(
        time=original_isf_schedule.time,
        name=original_isf_schedule.name,
        start_times=[start for start, _ in adjusted_schedule.keys()],
        values=list(adjusted_schedule.values()),
        duration_minutes=[original_isf_schedule.schedule_durations[k] for k in adjusted_schedule.keys()]
    )

    # Add debugging print statements
    print("Final adjusted pump settings:")
    print(f"Basal rates: {pump_config.basal_schedule.get_state()}")
    print(f"Carb ratios: {pump_config.carb_ratio_schedule.get_state()}")
    print(f"ISF: {pump_config.insulin_sensitivity_schedule.get_state()}")

    return pump_config


def build_corr_width(test_run=True):
    logger.debug("Random Seed: {}".format(SEED))

    sim_duration_hrs = 4 * 7 * 24 if not test_run else 48
    icgm_patients = get_patients_by_age(min_age=14, max_age=1e12)
    if test_run:
        icgm_patients = icgm_patients[:1]

    # Setup Patients
    patient_random_state = get_new_random_state()  # Single instance generates different patients

    sims = {}

    targets = [87, 100, 120, 140, 160, 180]
    widths = [0, 10, 20, 30]
    settings_deviations = [0, 10, 20, 30]

    for vp_idx, icgm_patient_obj in enumerate(icgm_patients):
        t0, patient_config = get_icgm_patient_config(icgm_patient_obj, patient_random_state)

        # Setup patient config
        patient_config.recommendation_accept_prob = 0.8
        patient_config.min_bolus_rec_threshold = patient_random_state.uniform(0.4, 0.6)
        patient_config.correct_bolus_bg_threshold = patient_random_state.uniform(180, 400)
        patient_config.correct_bolus_delay_minutes = patient_random_state.uniform(30, 60)
        patient_config.correct_carb_bg_threshold = patient_random_state.uniform(0, 80)
        patient_config.correct_carb_delay_minutes = patient_random_state.uniform(5, 15)
        patient_config.carb_count_noise_percentage = patient_random_state.uniform(0.12, 0.40)
        patient_config.report_bolus_probability = 1.0
        patient_config.report_carb_probability = patient_random_state.uniform(0.9, 1.0)
        patient_config.prebolus_minutes_choices = [0]
        patient_config.carb_reported_minutes_choices = [0]

        logger.debug("Patient Age: {}".format(icgm_patient_obj.age))
        t0, pump_config = get_pump_config_from_aace_settings(patient_random_state,
                                                             patient_weight=icgm_patient_obj.sample_weight_kg_by_age(
                                                                 random_state=patient_random_state),
                                                             patient_tdd=icgm_patient_obj.sample_total_daily_dose_by_age(
                                                                 random_state=patient_random_state),
                                                             risk_level=0,
                                                             t0=t0)
        logger.debug(f"Number of patients after age filtering: {len(icgm_patients)}")

        # Setup sensor config
        t0, sensor_config = get_canonical_sensor_config()
        sensor_config.std_dev = 0.0
        sensor_config.spurious_prob = 0.0
        sensor_config.spurious_outage_prob = 0.0
        sensor_config.time_delta_crunch_prob = 0.0
        sensor_config.name = "Clean"

        loop_connect_prob = patient_random_state.uniform(0.85, 1.0)

        for target in targets:
            for width in widths:
                for settings_deviation in settings_deviations:
                    sim_random_state = get_new_random_state(seed=vp_idx)

                    # Create a new pump_config for each iteration
                    t0, pump_config = get_pump_config_from_aace_settings(
                        patient_random_state,
                        patient_weight=icgm_patient_obj.sample_weight_kg_by_age(random_state=patient_random_state),
                        patient_tdd=icgm_patient_obj.sample_total_daily_dose_by_age(random_state=patient_random_state),
                        risk_level=0,
                        t0=t0
                    )

                    # Calculate the actual deviation range
                    lower_deviation = 1 - (settings_deviation / 100)
                    upper_deviation = 1 + (settings_deviation / 100)

                    # Adjust pump settings based on the deviation
                    pump_config = adjust_pump_settings(pump_config, lower_deviation, upper_deviation, sim_random_state)

                    # Set up target range
                    new_target_range_schedule = TargetRangeSchedule24hr(
                        t0,
                        start_times=[datetime.time(0, 0, 0)],
                        values=[TargetRange(target - width / 2, target + width / 2, "mg/dL")],
                        duration_minutes=[1440]
                    )
                    pump_config = adjust_pump_settings(pump_config, lower_deviation, upper_deviation, sim_random_state)

                    # Consistency check
                    if settings_deviation == 0:
                        assert abs(pump_config.insulin_sensitivity_schedule.get_state().value -
                                   patient_config.insulin_sensitivity_schedule.get_state().value) < 1e-6, \
                            "Pump and patient ISF don't match when they should"

                    sensor = IdealSensor(time=t0, sensor_config=sensor_config)
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
                    controller_config.controller_settings["max_basal_rate"] = icgm_patient_obj.get_basal_rate() * 4

                    controller = LoopControllerDisconnector(
                        time=t0,
                        controller_config=controller_config,
                        connect_prob=loop_connect_prob,
                        random_state=sim_random_state
                    )

                    # Setup sims
                    sim_id = f"{vp.name}_target={target}_width={width}_deviation={settings_deviation}"
                    sim = Simulation(
                        time=t0,
                        duration_hrs=sim_duration_hrs,
                        virtual_patient=vp,
                        sim_id=sim_id,
                        controller=controller,
                        multiprocess=True,
                        random_state=sim_random_state
                    )
                    sims[sim_id] = sim
                    logger.debug(f"Created simulation: {sim_id}")

        logger.info(f"Total simulations created: {len(sims)}")

        return sims


if __name__ == "__main__":
    test_run = True
    save_results = True

    results_dir = "."
    if save_results:
        results_dir = get_sim_results_save_dir("correction_width_analysis")

    sims = build_corr_width(test_run=test_run)

    if not sims:  # Add a check to ensure sims is not empty
        logger.error("No simulations were created. Check the build_corr_width function.")
    else:
        logger.info(f"Starting to run {len(sims)} sims.")
        all_results, metrics = run_simulations(sims,
                                               save_dir=results_dir,
                                               save_results=save_results,
                                               num_procs=10)

        # If all_results is not a dictionary, transform it into one
        if not isinstance(all_results, dict):
            all_results_dict = {sim.sim_id: result for sim, result in zip(sims.values(), all_results)}
        else:
            all_results_dict = all_results

        plot_sim_results(all_results_dict)