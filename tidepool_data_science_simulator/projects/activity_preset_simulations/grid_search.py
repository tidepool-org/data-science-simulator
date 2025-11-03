import sys

sys.path.append("..")
sys.path.append("../../")
sys.path.append("../../..")

import copy
import os
import re
import json
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import gc

from tidepool_data_science_simulator.models.sensor import NoisySensor, IdealSensor
from tidepool_data_science_simulator.models.patient import VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.controller import (
    DoNothingController,
    LoopController,
)
from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController
from tidepool_data_science_simulator.evaluation.inspect_results import load_results
from tidepool_data_science_simulator.makedata.make_simulation import (
    get_canonical_simulation,
)
from tidepool_data_science_simulator.makedata.make_patient import (
    get_canonical_risk_pump_config,
    get_canonical_virtual_patient_model_config,
    get_canonical_sensor_config,
    DATETIME_DEFAULT,
)
from datetime import datetime, timedelta
from param_values import metabolism_model_params

from tidepool_data_science_simulator.run import run_simulations

from tidepool_data_science_simulator.models.events import (
    CarbTimeline,
    BolusTimeline,
    PhysicalActivityTimeline,
)
from tidepool_data_science_simulator.models.simulation import TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.measures import (
    Carb,
    Bolus,
    PhysicalActivity,
    InsulinSensitivityFactor,
    BasalRate,
    CarbInsulinRatio,
    TargetRange,
)

import datetime
from pandas import Timestamp

import logging

logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.ERROR)

import matplotlib.dates as mdates

formatter = mdates.DateFormatter("%H:%M")
cmap = plt.cm.plasma_r

np.random.seed(42)


def get_noise(value, percentage=0.25, sample=True):
    if sample:
        return np.random.uniform(-percentage, percentage) * value
    else:
        noise_factor = np.random.choice([1 + percentage, 1 - percentage])
        return value * noise_factor


def build_metabolic_sensitivity_sims(
    start_glucose_value=110,
    basal_rate=0.3,
    cir=20.0,
    isf=150.0,
    controller=SwiftLoopController,
    add_noise=False,
    sample_noise=False,
    use_target=False,
    carb_timeline=None,
    pump_carb_timeline=None,
    bolus_timeline=None,
    duration_hrs=1,
    pa_timeline=None,
    heart_rate_trace=None,
    w_hr=1.0,
    w_ins=1.0,
    exercise_preset_p=1.0,
    a=0,
    n=0,
    tau=0,
    new_target_min=100,
    new_target_max=120,
):
    """
    Parameters:
    start_glucose_value: glucose value at the start of the simulation
    basal_rate, cir, isf: human metabolism model EGP, CIR, and ISF
    controller: SwiftLoopController or DoNothingController
    add_noise: noise between controller and human metabolism model parameters
    sample_noise: whether to uniformly sample noise
    use_target: whether to use raised controller target during activity
    target_range_min, target_range_max: controller target range
    carb_timeline, bolus_timeline: patient carb and bolus timeline
    pump_carb_timeline: pump carb timeline
    duration_hrs: simulation number of hours
    pa_timeline: physical activity timeline
    heart_rate_trace: patient heart rate trace
    exercise_preset_p: activity preset
    """

    # check to make sure parameters are being passed
    assert not a == 0
    assert not n == 0
    assert not tau == 0

    t0, patient_config = get_canonical_virtual_patient_model_config(
        start_glucose_value=start_glucose_value,
        basal_rate=basal_rate,
        cir=cir,
        isf=isf,
        carb_timeline=carb_timeline,
        bolus_timeline=bolus_timeline,
        pa_timeline=pa_timeline,
        sim_length=duration_hrs,
        heart_rate_trace=heart_rate_trace,
    )
    patient_config.recommendation_accept_prob = 1.0  # Accept the bolus
    patient_config.w_hr = w_hr  # KT: add w_hr
    patient_config.a = a  # MD: add metabolism model parameters
    patient_config.n = n
    patient_config.tau = tau

    t0, sensor_config = get_canonical_sensor_config(t0, start_value=start_glucose_value)
    sensor_config.std_dev = 1.0  # not sure what this is for

    if add_noise:
        pump_basal_rate = basal_rate + get_noise(basal_rate, 0.25, sample_noise)
        pump_cir = cir + get_noise(cir, 0.25, sample_noise)
        pump_isf = isf + get_noise(isf, 0.25, sample_noise)
    else:
        pump_basal_rate, pump_cir, pump_isf = basal_rate, cir, isf

    t0, pump_config = get_canonical_risk_pump_config(
        t0,
        basal_rate=pump_basal_rate,
        cir=pump_cir,
        isf=pump_isf,
        carb_timeline=pump_carb_timeline,
        bolus_timeline=bolus_timeline,
        pa_timeline=pa_timeline,
    )
    # preset overrides
    # for basal
    basal_p_factor = (
        -1 + exercise_preset_p
    )  # note: funky math because of how override function works
    pump_config.basal_schedule.set_override(basal_p_factor)

    # for isf
    isf_p_factor = -1 + 1 / (max(exercise_preset_p, 0.00000001))
    pump_config.insulin_sensitivity_schedule.set_override(isf_p_factor)

    # for cir
    cir_p_factor = -1 + 1 / (max(exercise_preset_p, 0.00000001))
    pump_config.carb_ratio_schedule.set_override(cir_p_factor)

    # activity duration
    for dt in pa_timeline.events:  # hack --> will work because there's only 1 pa
        pa = pa_timeline.events[dt]
    activity_duration = pa.duration

    # set the target
    target_end_time = time_plus(
        datetime.time(13, 00), timedelta(minutes=activity_duration)
    )  # note --> this will change if we decide to start the activity at a different time than 1 hour after sim start
    target_duration = 60 * 1 + activity_duration
    remaining_time = 720 - target_duration

    if use_target:
        target_range_schedule = TargetRangeSchedule24hr(
            t0,
            start_times=[
                datetime.time(hour=0, minute=0, second=0),
                datetime.time(hour=12, minute=0, second=0),
                target_end_time,
            ],
            values=[
                TargetRange(100, 120, "mg/dL"),
                TargetRange(new_target_min, new_target_max, "mg/dL"),
                TargetRange(100, 120, "mg/dL"),
            ],
            duration_minutes=[720, target_duration, remaining_time],
        )
        pump_config.target_range_schedule = target_range_schedule
    else:
        target_range_schedule = TargetRangeSchedule24hr(
            t0,
            start_times=[
                datetime.time(hour=0, minute=0, second=0),
                datetime.time(hour=12, minute=0, second=0),
                target_end_time,
            ],
            values=[
                TargetRange(100, 120, "mg/dL"),
                TargetRange(100, 120, "mg/dL"),
                TargetRange(100, 120, "mg/dL"),
            ],
            duration_minutes=[720, target_duration, remaining_time],
        )
        pump_config.target_range_schedule = target_range_schedule

    # set override to patient isf for 60 minutes
    isf_change_factor = -1 + w_ins
    basal_change_factor = -1 + 1 / (w_ins)
    patient_config.insulin_sensitivity_schedule.schedule_durations = {
        (datetime.time(0, 0), datetime.time(13, 0)): 720 + 60,
        (datetime.time(13, 0), target_end_time): target_duration - 60,
        (target_end_time, datetime.time(23, 59, 59)): remaining_time + 60,
    }

    patient_config.insulin_sensitivity_schedule.schedule = {
        (datetime.time(0, 0), datetime.time(13, 0)): InsulinSensitivityFactor(
            isf, "mg/dL/U"
        ),
        (datetime.time(13, 0), target_end_time): InsulinSensitivityFactor(
            isf * (1.0 + isf_change_factor), "mg/dL/U"
        ),
        (target_end_time, datetime.time(23, 59, 59)): InsulinSensitivityFactor(
            isf, "mg/dL/U"
        ),
    }

    patient_config.basal_schedule.schedule_durations = {
        (datetime.time(0, 0), datetime.time(13, 0)): 720 + 60,
        (datetime.time(13, 0), target_end_time): target_duration - 60,
        (target_end_time, datetime.time(23, 59, 59)): remaining_time + 60,
    }
    patient_config.basal_schedule.schedule = {
        (datetime.time(0, 0), datetime.time(13, 0)): BasalRate(basal_rate, "mg/dL"),
        (datetime.time(13, 0), target_end_time): BasalRate(
            basal_rate * (1.0 + basal_change_factor), "mg/dL"
        ),
        (target_end_time, datetime.time(23, 59, 59)): BasalRate(basal_rate, "mg/dL"),
    }

    t0, sim = get_canonical_simulation(
        patient_config=patient_config,
        patient_class=VirtualPatientModel,
        sensor_config=sensor_config,
        # sensor_class=NoisySensor,
        sensor_class=IdealSensor,
        pump_config=pump_config,
        pump_class=ContinuousInsulinPump,
        controller_class=controller,
        multiprocess=True,
        duration_hrs=duration_hrs,
    )
    # sim.controller.controller_config.controller_settings['partial_application_factor'] *= preset
    return sim


# https://stackoverflow.com/a/31817722
def time_plus(time, timedelta):
    start = datetime.datetime(
        2000, 1, 1, hour=time.hour, minute=time.minute, second=time.second
    )
    end = start + timedelta
    return end.time()


if __name__ == "__main__":
    code_start_time = time.time()

    save_root_base = f"./04_07_long_simulations"
    if not os.path.isdir(save_root_base):
        os.makedirs(save_root_base)

    # Grid search parameters
    sim_conditions = ["preset+target"]
    starting_glucoses = np.arange(
        70, 270, 20
    ).tolist()  # from 80 to 160, with an interval of 20
    netIoBs = [0, 1, 2, 3]
    pa_durations = [30, 60]
    activities = ["walking", "biking", "jogging", "strength training"]
    presets = (
        np.arange(1, 21, 1) / 10.0
    ).tolist()  # from 0.1 to 2.0, using the division to avoid floating point error
    targets = [
        (100, 120),
        (120, 140),
        (140, 160),
        (150, 170),
        (160, 180),
    ]
    num_virtual_patients = 10
    noise_conditions = ["nonoise", "samplenoise", "fullnoise"]

    # activity names for PA model parameters
    activity_name_map = {
        "Walking, Dog Walking": "walking",
        "Biking (Indoor or Outdoor)": "biking",
        "Jogging": "jogging",
        "Strength Training, Weight Lifting": "strength training",
    }

    # VP info
    vp_info_path = "./tidepool_helmsley_preset_virtual_patients_oct_2024.csv"
    vp_info = pd.read_csv(vp_info_path)
    vp_info = vp_info[vp_info["age"] >= 18]  # select adult virtual patients
    vp_info = vp_info.head(num_virtual_patients)
    print(f"vp info: {vp_info}")

    # metabolism model parameters
    w_ins_vals = metabolism_model_params["w_ins_vals"]
    w_hr_vals = metabolism_model_params["w_hr_vals"]
    a_vals = metabolism_model_params["a_vals"]
    n_vals = metabolism_model_params["n_vals"]
    tau_vals = metabolism_model_params["tau_vals"]
    hr_vals = metabolism_model_params["hr_vals"]

    # default parameters
    t0 = DATETIME_DEFAULT
    sim_activity_starttime = t0 + timedelta(
        hours=1
    )  # start activity 1 hour after start of simulation

    for sim_condition in sim_conditions:

        # update parameters
        use_controller = False
        use_preset = False
        use_target = False

        if sim_condition == "preset+target":
            use_controller = True
            use_target = True
            use_preset = True
        else:
            raise Exception(
                "This grid search only considers the case when both percentage and target are activated!"
            )

        for noise_condition in noise_conditions:

            # update parameters
            add_noise = False
            sample_noise = False

            if noise_condition == "samplenoise":
                add_noise = True
                sample_noise = True
            elif noise_condition == "fullnoise":
                add_noise = True

            # update experiment name
            if use_controller:
                experiment_name = "controller"
            else:
                experiment_name = "nocontroller"

            if not add_noise:
                experiment_name += "_nonoise"
            elif sample_noise:
                experiment_name += "_samplednoise"
            else:
                experiment_name += "_fullnoise"

            if use_preset:
                experiment_name += "_withpreset"
            else:
                experiment_name += "_withoutpreset"

            if use_target:
                experiment_name += "_withtarget"
            else:
                experiment_name += "_withouttarget"

            print(f"Experiment name: {experiment_name}")

            save_root = os.path.join(save_root_base, experiment_name)
            if not os.path.isdir(save_root):
                os.makedirs(save_root)

            for activity in activities:
                # pa model parameters
                w_ins = w_ins_vals[activity]
                w_hr = w_hr_vals[activity]
                a = a_vals[activity]
                n = n_vals[activity]
                tau = tau_vals[activity]
                hr = hr_vals[activity]

                for starting_glucose in starting_glucoses:
                    for netIoB in netIoBs:
                        for pa_duration in pa_durations:
                            for preset in presets:
                                start_time = time.time()
                                sims = {}
                                for target in targets:
                                    for vp_index, vp in vp_info.iterrows():

                                        isf = vp["isf"]
                                        cir = vp["cir"]
                                        egp = vp["basal_rate"]

                                        if not use_preset:
                                            preset = 1.0
                                        if use_controller:
                                            controller = SwiftLoopController
                                        else:
                                            controller = DoNothingController

                                        sim_name = f"{activity}_{starting_glucose}_{netIoB}_{pa_duration}_{preset}_{target}_{vp_index}"
                                        savefilename = os.path.join(
                                            save_root, f"{sim_name}.tsv"
                                        )

                                        if os.path.isfile(savefilename):
                                            print(
                                                f"FOUND FILE FOR {sim_name}.. continuing"
                                            )
                                            continue

                                        print(f"CREATING {sim_name}")

                                        bolus_timeline = BolusTimeline(
                                            [t0], [Bolus(netIoB, "U")]
                                        )
                                        carb_timeline = CarbTimeline()
                                        pump_carb_timeline = CarbTimeline()

                                        pa_timeline = PhysicalActivityTimeline(
                                            datetimes=[sim_activity_starttime],
                                            events=[
                                                PhysicalActivity(
                                                    activity=activity,
                                                    duration=pa_duration,
                                                )
                                            ],
                                        )
                                        hrs = [hr] * (pa_duration * 60 // 10)
                                        duration_hrs = math.ceil(
                                            (pa_duration + 4 * 60) / 60
                                        )
                                        sim = build_metabolic_sensitivity_sims(
                                            start_glucose_value=starting_glucose,
                                            basal_rate=egp,
                                            cir=cir,
                                            isf=isf,
                                            controller=controller,
                                            add_noise=add_noise,
                                            sample_noise=sample_noise,
                                            use_target=use_target,
                                            carb_timeline=carb_timeline,
                                            pump_carb_timeline=pump_carb_timeline,
                                            bolus_timeline=bolus_timeline,
                                            duration_hrs=duration_hrs,
                                            pa_timeline=pa_timeline,
                                            heart_rate_trace=hrs,
                                            w_hr=w_hr,
                                            w_ins=w_ins,
                                            exercise_preset_p=preset,
                                            a=a,
                                            n=n,
                                            tau=tau,
                                            new_target_min=target[0],
                                            new_target_max=target[1],
                                        )
                                        sims[sim_name] = sim
                                        gc.collect()

                                # The code below is put at this loop, because we don't want to initialize too many simulations before running any.

                                gc.collect()
                                print(f"RUNNING {len(sims)} sims")
                                save_dir = save_root
                                if not os.path.isdir(save_dir):
                                    os.makedirs(save_dir)

                                batch_size = 60
                                sims_items = list(sims.items())
                                sim_batches = [
                                    dict(sims_items[i : i + batch_size])
                                    for i in range(0, len(sims), batch_size)
                                ]

                                for batch_num, sim_batch in enumerate(sim_batches):
                                    print(f"BATCH {batch_num} of {len(sim_batches)}")
                                    all_results, summary_results_df = run_simulations(
                                        sim_batch,
                                        save_dir=save_dir,
                                        save_results=True,
                                        num_procs=20,
                                        name=f"{batch_num}",
                                    )

                                    for sim_id, results_df in all_results.items():
                                        hr_trace = sims[
                                            sim_id
                                        ].virtual_patient.patient_config.hr_trace
                                        hr_series = pd.Series(
                                            data=hr_trace.hr_values,
                                            index=hr_trace.datetimes,
                                        )
                                        filtered_hrs_series = hr_series.loc[
                                            hr_series.index.isin(results_df.index)
                                        ]
                                        results_df["hrs"] = filtered_hrs_series.reindex(
                                            results_df.index
                                        )
                                        results_df_dict = {sim_id: results_df}

                                end_time = time.time()
                                elapsed_time = end_time - start_time
                                elapsed_hours = elapsed_time / 3600

                                print(
                                    f"COMPLETED {len(sims)} SIMS IN {elapsed_hours:.2f} HOURS ({elapsed_time:.2f} SECONDS)"
                                )
