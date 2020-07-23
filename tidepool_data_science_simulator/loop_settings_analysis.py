__author__ = "Cameron Summers"

import os
import numpy as np
import datetime

import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_metrics.glucose.glucose import lbgi_risk_score
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index

from tidepool_data_science_simulator.models.simulation import Simulation, BasalSchedule24hr
from tidepool_data_science_simulator.models.measures import BasalRate
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_sensor_config, get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.patient.virtual_patient import VirtualPatientModelCarbBolusAccept
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import NoisySensor
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


@timing
def eval_settings():
    """
    """
    np.random.seed(1234)

    param_grid = [
        {
            "suspend_threshold": round(st, 2),
        }
        for st in np.arange(115, 120, 5)
    ]

    sims = {}
    sim_params = {}
    for pgrid in param_grid:
        patient_seed = np.random.randint(1e8, 9e8)

        sim_id = "st {}".format(pgrid["suspend_threshold"])
        print("Running: {}".format(sim_id))

        t0, patient_config = get_canonical_risk_patient_config()
        patient_config.min_bolus_rec_threshold = 0.5
        patient_config.recommendation_accept_prob = 1.0
        patient_config.report_carb_probability = 0.8

        t0, pump_config = get_canonical_risk_pump_config()
        pump_config.basal_schedule = BasalSchedule24hr(t0,
                                                          [datetime.time(hour=0, minute=0, second=0)],
                                                          [BasalRate(0.4, "U/hr")],
                                                          [1440])
        pump_config.insulin_sensitivity_schedule = SettingSchedule24Hr(
            t0,
            "CIR",
            start_times=[datetime.time(hour=0, minute=0, second=0)],
            values=[InsulinSensitivityFactor(20.0, "g/U")],
            duration_minutes=[1440]
        )

        t0, controller_config = get_canonical_controller_config()
        t0, sensor_config = get_canonical_sensor_config()
        controller_config.controller_settings["suspend_threshold"] = pgrid["suspend_threshold"]

        controller = LoopController(time=t0, controller_config=controller_config)
        pump = ContinuousInsulinPump(time=t0, pump_config=pump_config)
        sensor = NoisySensor(time=t0, sensor_config=sensor_config)

        vp = VirtualPatientModelCarbBolusAccept(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
            remember_meal_bolus_prob=1.0,
            correct_bolus_bg_threshold=180,
            correct_bolus_delay_minutes=30,
            correct_carb_bg_threshold=80,
            correct_carb_delay_minutes=10,
            carb_count_noise_percentage=0.5,
            id=0,
        )

        sim = Simulation(
            time=t0,
            duration_hrs=8.0,
            virtual_patient=vp,
            controller=controller,
            multiprocess=True
        )
        sim.seed = patient_seed

        sims[sim_id] = sim
        sim_params[sim_id] = pgrid
        sim.start()

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

    # Gather results and get dka risk
    summary_results_df = []
    for sim_id, results_df in all_results.items():

        lbgi, hbgi, brgi = blood_glucose_risk_index(results_df['bg'])

        row = {
            "lbgi": lbgi,
            "suspend_threshold": sim_params[sim_id]["suspend_threshold"],
        }
        print(row)
        summary_results_df.append(row)

    summary_results_df = pd.DataFrame(summary_results_df)

    print("plotting...")
    sns.boxplot(y=summary_results_df["lbgi"])

    plot_sim_results(all_results)

    plt.show()


if __name__ == "__main__":

    eval_settings()
