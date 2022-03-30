__author__ = "Cameron Summers"

import logging
import datetime
import numpy as np

import matplotlib.pyplot as plt
import pandas as pd

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation, TargetRangeSchedule24hr
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import (
  DATETIME_DEFAULT, get_canonical_risk_patient_config, get_canonical_risk_pump_config,
    get_canonical_sensor_config, SINGLE_SETTING_DURATION, SINGLE_SETTING_START_TIME
)

from tidepool_data_science_simulator.models.simulation import (
    SettingSchedule24Hr, BasalSchedule24hr, TargetRangeSchedule24hr
)
from tidepool_data_science_simulator.models.measures import (
    InsulinSensitivityFactor, CarbInsulinRatio, BasalRate, TargetRange, GlucoseTrace, Bolus, Carb
)

from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.measures import Bolus, Carb, TargetRange

from tidepool_data_science_simulator.run import run_simulations
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results


logger = logging.getLogger(__name__)


def generate_bolus_lbgi_sims(pgrid):

    sims = {}
    sim_id_params_map = {}

    for params in pgrid:
        basal = params["basal_rate"]
        start_bg = params["start_bg"]
        bolus_amt = params["bolus_amount"]
        t0, patient_config = get_canonical_risk_patient_config(start_glucose_value=start_bg)

        basal_schedule = BasalSchedule24hr(
            t0,
            start_times=[SINGLE_SETTING_START_TIME],
            values=[BasalRate(basal, "mg/dL")],
            duration_minutes=[SINGLE_SETTING_DURATION]
        )

        patient_config.basal_schedule = basal_schedule

        t0, sensor_config = get_canonical_sensor_config(start_value=start_bg)
        t0, controller_config = get_canonical_controller_config()
        t0, pump_config = get_canonical_risk_pump_config()

        bolus_timeline = BolusTimeline(datetimes=[t0], events=[Bolus(bolus_amt, "U")])
        patient_config.bolus_event_timeline = bolus_timeline
        pump_config.bolus_event_timeline = bolus_timeline

        pump = ContinuousInsulinPump(pump_config, t0)
        sensor = IdealSensor(t0, sensor_config)
        # controller = DoNothingController(t0, controller_config)
        controller = LoopController(t0, controller_config)

        vp = VirtualPatient(
            time=DATETIME_DEFAULT,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config
        )

        sim_id = "start_bg={}_bolus={}_basal={}".format(start_bg, bolus_amt, basal)
        sim = Simulation(
            time=t0,
            duration_hrs=8,
            virtual_patient=vp,
            controller=controller,
            sim_id=sim_id,
            multiprocess=True
        )

        sims[sim_id] = sim
        sim_id_params_map[sim_id] = params

    return sims, sim_id_params_map


if __name__ == "__main__":

    bolus_amounts = np.arange(0, 2, 0.2)
    starting_bgs = [100]
    basals = [0.3]

    pgrid = [
        {
            "bolus_amount": bolus_amt,
            "start_bg": start_bg,
            "basal_rate": basal
        }
        for basal in basals
        for start_bg in starting_bgs
        for bolus_amt in bolus_amounts
    ]
    sim_batch, sim_id_params_map = generate_bolus_lbgi_sims(pgrid)


    full_results, summary_results_df = run_simulations(
        sim_batch,
        save_dir=".",
        save_results=False,
        num_procs=8
    )

    data = []
    for sim_id, params in sim_id_params_map.items():
        params.update({
            "lbgi":summary_results_df[summary_results_df.index == sim_id]["lbgi"]
        })
        data.append(params)
    df = pd.DataFrame(data)

    for basal in basals:
        isf_data = df[df["basal_rate"] == basal]
        plt.plot(isf_data["bolus_amount"], isf_data["lbgi"], label="basal={}".format(basal))

    # plt.ylim(0, 15)
    plt.legend()
    plt.show()
    plot_sim_results(full_results)