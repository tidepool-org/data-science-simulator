__author__ = "Cameron Summers"

import os
import numpy as np
import matplotlib.pyplot as plt

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient, VirtualPatientModel
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import get_equivalent_isf


def analyze_settings(scenario_csv_filepath, param_grid):
    """
    Look at resulting bgs from settings that are correct/incorrect for analysis.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file

    param_grid: list of dicts
        Parameters to vary
    """
    sim_parser = ScenarioParserCSV(scenario_csv_filepath)

    # FIXME: Warning, Hack! For near term presentation. Don't do this. Need to refactor parser.
    sims = {}
    for pgrid in param_grid:
        br = pgrid["basal_rate"]
        isf = pgrid["isf"]
        target = 110  # shortcut, this is value in scenario
        starting_glucose = 250
        sim_parser.tmp_dict["glucose_values"] = [starting_glucose] * len(
            sim_parser.tmp_dict["glucose_values"]
        )
        sim_parser.tmp_dict["actual_blood_glucose"] = [starting_glucose] * len(
            sim_parser.tmp_dict["actual_blood_glucose"]
        )
        sim_parser.tmp_dict["basal_rate_values"][0] = br
        sim_parser.tmp_dict["sensitivity_ratio_values"][0] = isf
        sim_parser.tmp_dict["dose_values"][0] = (starting_glucose - target) / isf
        print("Correction Bolus: ", sim_parser.tmp_dict["dose_values"][0])

        t0 = sim_parser.get_simulation_start_time()

        controller = LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config(),
        )
        pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())
        sensor = IdealSensor(sensor_config=sim_parser.get_sensor_config())
        # sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

        print("Length of param grid: {}".format(len(param_grid)))

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=sim_parser.get_patient_config(),
        )

        sim_id = "{} BR: {} ISF: {}".format(vp.name, pgrid["basal_rate"], pgrid["isf"])
        print("Running: {}".format(sim_id))

        simulation = Simulation(
            time=t0,
            duration_hrs=18.0,
            virtual_patient=vp,
            controller=controller,
            multiprocess=True,
        )
        sims[sim_id] = simulation
        simulation.start()

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

    plot_sim_results(all_results)


def plot_auc_basal_isf():
    """
    Visualize complementary ISF/Basal Rates
    """

    egp = 0.5

    num_hours = 8
    fig2, ax2 = plt.subplots(1, 1)
    cumulative_delta_bgs = [100]
    # cumulative_delta_bgs = [100, 120, 130, 140, 150]

    for cum_delta_bg in cumulative_delta_bgs:
        brs = np.arange(0.0, 1.0, 0.1)
        isfs = []

        for br in brs:

            ss_br = br * 2.1

            isf = cum_delta_bg / (ss_br + 1.0)
            isfs.append(isf)

            insulin_model = PalermInsulinModel(isf=isf, cir=0)

            t_min, bg_delta, bg, iob = insulin_model.run(num_hours, 1.0, five_min=True)

            ss_br_bgd = [ss_br * isf / len(t_min) for _ in t_min]
            isf_bgd = -bg_delta + ss_br_bgd
            egp_bdg = [egp * isf / len(t_min) for _ in t_min]

            if 1:
                fig, ax = plt.subplots(1, 1)
                ax.plot(t_min, ss_br_bgd, label="SBR")
                ax.plot(t_min, egp_bdg, label="EGP")
                ax.plot(t_min, isf_bgd, label="ISF")

                ax.set_ylim(0, 3.5)
                plt.fill_between(t_min, egp_bdg, isf_bgd, color="grey", alpha=0.5)

                ax.legend()
                # plt.show()

        ax2.plot(brs, isfs)
    plt.show()


if __name__ == "__main__":

    scenarios_folder_path = "../data/raw/"

    scenario_csv_filepath = os.path.join(
        scenarios_folder_path,
        "Scenario-0-simulation-template - inputs - SettingsDemo.tsv",
    )

    #plot_auc_basal_isf()

    # Explore grid
    param_grid = [
        {"basal_rate": br, "isf": isf}
        for br in [0.2, 0.3, 0.4]
        for isf in [140, 150, 160]
    ]

    # Explore equivalent combos of br and isf
    brs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    param_grid = [
        {"basal_rate": br, "isf": isf}
        for br, isf in zip(brs, get_equivalent_isf(250 - 110, brs))
    ]

    analyze_settings(scenario_csv_filepath, param_grid)
