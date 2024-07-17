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
from tidepool_data_science_simulator.models.simulation import Simulation, TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.controller import LoopController, LoopControllerDisconnector
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.models.measures import TargetRange

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, \
    get_pump_config_from_patient, compute_aace_settings_tmp, get_variable_risk_patient_config, get_pump_config_from_aace_settings, BasalRate, CarbInsulinRatio, InsulinSensitivityFactor
from tidepool_data_science_simulator.makedata.make_icgm_patients import (
    get_patients_by_age, get_icgm_patient_config, get_icgm_patient_settings_objects
)
from tidepool_data_science_simulator.utils import get_sim_results_save_dir

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


from numpy.random import RandomState

SEED = 1234567890


def get_new_random_state(seed=SEED):
    return RandomState(seed)


def build_corr_width(test_run=False):
    """Create sims for analyzing various correction range widths
    with varying levels of deviation from true in settings"""

    logger.debug("Random Seed: {}".format(SEED))

    sim_duration_hrs = 4 * 7 * 24 if not test_run else 48
    icgm_patients = get_patients_by_age(min_age=14, max_age=1e12)
    if test_run:
        icgm_patients = icgm_patients[:1]

# Setup Patients
    patient_random_state = get_new_random_state() #Single instance generates different patients
    sims = {}

    targets = [87, 100, 120, 140, 160, 180]
    widths = [0, 10, 20, 30]
    settings_deviations = [0, 10, 20]

    for vp_idx, icgm_patient_obj in enumerate(icgm_patients):
        t0, patient_config = get_icgm_patient_config(icgm_patient_obj, patient_random_state)

        # Setup patient config
        patient_config.recommendation_accept_prob = 0.8
        patient_config.min_bolus_rec_threshold = patient_random_state.uniform(0.4, 0.6)
        patient_config.correct_bolus_bg_threshold = patient_random_state.uniform(140, 190)
        patient_config.correct_bolus_delay_minutes = patient_random_state.uniform(20, 40)
        patient_config.correct_carb_bg_threshold = patient_random_state.uniform(70, 90)
        patient_config.correct_carb_delay_minutes = patient_random_state.uniform(5, 15)
        patient_config.carb_count_noise_percentage = patient_random_state.uniform(0.2, 0.25)
        patient_config.report_bolus_probability = 1.0
        patient_config.report_carb_probability = patient_random_state.uniform(0.95, 1.0)
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
        sensor_config.std_dev = 2.0
        sensor_config.spurious_prob = 0.0
        sensor_config.spurious_outage_prob = 0.0
        sensor_config.time_delta_crunch_prob = 0.0
        sensor_config.name = "Clean"

        loop_connect_prob = patient_random_state.uniform(0.8, 0.95)

        for target in targets:
            for width in widths:
                for settings_deviation in settings_deviations:
                    sim_random_state = get_new_random_state(seed=vp_idx)

                    # Calculate the actual deviation range
                    lower_deviation = 1 - (settings_deviation / 100)
                    upper_deviation = 1 + (settings_deviation / 100)

                    # Adjust pump settings based on the deviation
                    adjusted_pump_config = adjust_pump_settings(pump_config, lower_deviation, upper_deviation)

                    # Set up target range
                    new_target_range_schedule = TargetRangeSchedule24hr(
                        t0,
                        start_times=[datetime.time(0, 0, 0)],
                        values=[TargetRange(target - width / 2, target + width / 2, "mg/dL")],
                        duration_minutes=[1440]
                    )
                    adjusted_pump_config.target_range_schedule = new_target_range_schedule

                    sensor = NoisySensor(time=t0, sensor_config=sensor_config, random_state=sim_random_state)
                    pump = ContinuousInsulinPump(time=t0, pump_config=adjusted_pump_config)

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

                    #Setup sims
                    sim_id = f"{vp.name}_target={target}_width={width}_deviation={settings_deviation}"
                    sim = Simulation(
                        time=t0,
                        duration_hrs=sim_duration_hrs, #4 weeks
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

def adjust_pump_settings(pump_config, lower_deviation, upper_deviation):
    # debugging assistance
    print("BasalSchedule24hr methods:", dir(pump_config.basal_schedule))
    print("Carb Ratio Schedule methods:", dir(pump_config.carb_ratio_schedule))
    print("Insulin Sensitivity Schedule methods:", dir(pump_config.insulin_sensitivity_schedule))

    # Adjust basal rate
    basal_state = pump_config.basal_schedule.get_state()
    if isinstance(basal_state, list) and basal_state and isinstance(basal_state[0], BasalRate):
        adjusted_basal_rates = [
            BasalRate(rate.value * np.random.uniform(lower_deviation, upper_deviation), rate.units)
            for rate in basal_state
        ]
        pump_config.basal_schedule.set_state(adjusted_basal_rates)
    else:
        print(f"Unexpected basal state type: {type(basal_state)}")

    # Adjust carb ratio
    carb_ratio_state = pump_config.carb_ratio_schedule.get_state()
    if isinstance(carb_ratio_state, list) and carb_ratio_state and isinstance(carb_ratio_state[0], CarbInsulinRatio):
        adjusted_carb_ratios = [
            CarbInsulinRatio(ratio.value * np.random.uniform(lower_deviation, upper_deviation), ratio.units)
            for ratio in carb_ratio_state
        ]
        pump_config.carb_ratio_schedule.set_state(adjusted_carb_ratios)
    else:
        print(f"Unexpected carb ratio state type: {type(carb_ratio_state)}")

    # Adjust insulin sensitivity factor
    isf_state = pump_config.insulin_sensitivity_schedule.get_state()
    if isinstance(isf_state, list) and isf_state and isinstance(isf_state[0], InsulinSensitivityFactor):
        adjusted_isfs = [
            InsulinSensitivityFactor(isf.value * np.random.uniform(lower_deviation, upper_deviation), isf.units)
            for isf in isf_state
        ]
        pump_config.insulin_sensitivity_schedule.set_state(adjusted_isfs)
    else:
        print(f"Unexpected insulin sensitivity factor state type: {type(isf_state)}")

    return pump_config


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