__author__ = "Cameron Summers"

__author__ = "Cameron Summers"

import os
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.models.measures import TempBasal, BasalRate

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation, BasalSchedule24hr
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results_missing_insulin
from tidepool_data_science_simulator.utils import timing

from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score


def analyze_omnipod_missing_pulses():
    current_time, patient = get_canonical_risk_patient(pump_class=OmnipodMissingPulses)

    delivered_insulin = []
    undelivered_insulin = []

    temp_basal_scenario = [0.7, 0.65] * 12  # This scenario gives no insulin

    for temp_basal_value in temp_basal_scenario:
        next_time = current_time + datetime.timedelta(minutes=5)

        patient.pump.update(next_time)
        temp_basal = TempBasal(next_time, temp_basal_value, duration_minutes=30, units="U/hr")
        patient.pump.set_temp_basal(temp_basal)

        delivered_insulin.append(patient.pump.basal_insulin_delivered_last_update)
        undelivered_insulin.append(patient.pump.basal_undelivered_insulin_since_last_update)

    total_delivered_insulin = np.sum(delivered_insulin)
    total_undelivered_insulin = np.sum(undelivered_insulin)
    total_expected_insulin = total_delivered_insulin + total_undelivered_insulin
    print("Total Delivered Insulin  {:>4} ({:.0f}%)".format(total_delivered_insulin,
                                                            total_delivered_insulin / total_expected_insulin * 100.0)),
    print("Total Undelivered Insulin {:>4} ({:.0f}%)".format(total_undelivered_insulin,
                                                             total_undelivered_insulin / total_expected_insulin * 100.0))

    plt.title("Omnipod Missing Insulin Pulses")
    plt.plot(delivered_insulin, label="delivered")
    plt.plot(undelivered_insulin, label="undelivered")
    plt.plot(temp_basal_scenario, label="Temp Basal Values")
    plt.legend()
    plt.show()


@timing
def analyze_omnipod_missing_pulses_wLoop(scenario_csv_filepath, dry_run):
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    param_grid = [
        {
            "loop_max_basal_rate": round(sbr * xer, 2),
            "patient_basal_rate": round(sbr, 2),
            "pump_basal_rate": round(sbr, 2)
        }
        for sbr in np.arange(0.05, 0.75, 0.05)
        for xer in [1.5, 2, 3, 4, 5, 6] #range(2, 20)
        # for sbr in [0.2]
        # for xer in [3]
    ]
    sim_num_hours = 24

    if dry_run:
        sim_num_hours = 2
        param_grid = param_grid[:1]

    sims = {}
    sim_params = {}
    for pgrid in param_grid:
        np.random.seed(1234)
        sim_id = "SBR {pump_basal_rate} VPBR {patient_basal_rate} MBR {loop_max_basal_rate}".format(**pgrid)
        print("Running: {}".format(sim_id))

        patient_config = sim_parser.get_patient_config()
        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file

        patient_config.basal_schedule = BasalSchedule24hr(t0,
                                                            "basal_rate",
                                                            [datetime.time(hour=0, minute=0, second=0)],
                                                            [BasalRate(pgrid['patient_basal_rate'], "U/hr")],
                                                            [1440])

        pump_config = sim_parser.get_pump_config()
        pump_config.basal_schedule = BasalSchedule24hr(t0,
                                                         "basal_rate",
                                                         [datetime.time(hour=0, minute=0, second=0)],
                                                         [BasalRate(pgrid['pump_basal_rate'], "U/hr")],
                                                         [1440])

        controller_config = sim_parser.get_controller_config()
        controller_config.controller_settings["max_basal_rate"] = pgrid["loop_max_basal_rate"]
        controller = LoopController(
            time=t0,
            loop_config=controller_config,
            simulation_config=sim_parser.get_simulation_config(),
        )

        pump = OmnipodMissingPulses(time=t0, pump_config=pump_config)

        # Facilitates frequent temp basals
        sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
        )

        simulation = Simulation(
            time=t0,
            duration_hrs=sim_num_hours,
            simulation_config=sim_parser.get_simulation_config(),
            virtual_patient=vp,
            controller=controller,
            multiprocess=True
        )
        sims[sim_id] = simulation
        sim_params[sim_id] = pgrid
        simulation.start()

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

    summary_results_df = []
    for sim_id, results_df in all_results.items():
        dkai = dka_index(results_df['iob'], sim_params[sim_id]["patient_basal_rate"])
        dkars = dka_risk_score(dkai)

        print("dkai", dkai)
        row = {
            "dka_index": dkai,
            "dka_risk_score": dkars,
            "loop_max_basal_rate": sim_params[sim_id]["loop_max_basal_rate"],
            "sbr": sim_params[sim_id]["patient_basal_rate"]
        }
        summary_results_df.append(row)

    summary_results_df = pd.DataFrame(summary_results_df)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    if 1:
        summary_results_pivot_df = summary_results_df.pivot(index='sbr', columns='loop_max_basal_rate', values='dka_index')
        sns.heatmap(summary_results_pivot_df, ax=ax1)
        ax1.set_title("DKAI for Canonical Patient with Low Basal Settings")

        plt.figure()
        summary_results_pivot_df = summary_results_df.pivot(index='sbr', columns='loop_max_basal_rate', values='dka_risk_score')
        sns.heatmap(summary_results_pivot_df, ax=ax2)
        ax2.set_title("DKA Risk Score for Canonical Patient with Low Basal Settings")

        plt.show()
    else:
        plot_sim_results_missing_insulin(all_results)


if __name__ == "__main__":
    scenarios_folder_path = "../data/raw/"
    scenario_csv_filepath = os.path.join(
        scenarios_folder_path, "Scenario-0-simulation-template - inputs - OmnipodMissingPulses.tsv"
    )
    analyze_omnipod_missing_pulses_wLoop(scenario_csv_filepath, False)

    # analyze_omnipod_missing_pulses()
