__author__ = "Cameron Summers"

"""
This file to for running risk analysis of Tidepool Loop when a
user deletes their insulin history from the Loop app.

Tidepool Loop Risk Card: https://tidepool.atlassian.net/browse/TLR-337
"""

import os
from datetime import timedelta

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing
from tidepool_data_science_metrics.insulin.insulin import dka_index
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, lbgi_risk_score, percent_values_lt_40

import numpy as np
import pandas as pd

@timing
def risk_analysis_tlr337_user_delete_data(scenario_csv_filepath):
    """
    Compare loop running with an action and without that action.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """

    sim_parser = ScenarioParserCSV(scenario_csv_filepath)
    t0 = sim_parser.get_simulation_start_time()

    comparison_controllers = [
        DoNothingController(
            time=t0,
            controller_config=sim_parser.get_controller_config()
        ),
        LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config()
        )
    ]

    comparison_patient_action_configs = [
        [],
        [
            {
                "t0_delay_minutes": 0,
                "action": VirtualPatientDeleteLoopData("Deleted Insulin History")
            }
        ]
    ]

    configurations = [
        {
            "controller": controller,
            "actions": actions,
        }
        for controller in comparison_controllers
        for actions in comparison_patient_action_configs
    ]

    all_results = {}
    for i, config in enumerate(configurations):

        sim_id = i
        actions = config["actions"]
        controller = config["controller"]

        print("Running: {}".format(sim_id))

        pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())
        sensor = IdealSensor(time=t0, sensor_config=sim_parser.get_sensor_config())

        patient_config = sim_parser.get_patient_config()
        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file
        patient_config.action_timeline = ActionTimeline()

        for action_config in actions:
            delay_minutes = action_config["t0_delay_minutes"]
            action = action_config["action"]
            patient_config.action_timeline.add_event(t0 + timedelta(minutes=delay_minutes), action)

        vp = VirtualPatient(
            time=t0,
            pump=pump,
            sensor=sensor,
            metabolism_model=SimpleMetabolismModel,
            patient_config=patient_config,
        )

        simulation = Simulation(
            time=t0,
            duration_hrs=8.0,
            virtual_patient=vp,
            controller=controller,
        )

        simulation.run()

        results_df = simulation.get_results_df()
        all_results[sim_id] = results_df

    dkais = {}
    lgbis = {}
    for sim_id, results_df in all_results.items():
        # TODO: Separate out into it's own function
        risk_assessment_array = np.zeros((5, len(all_results)))
        ids = []
        for items, results_tuple in enumerate(all_results.items()):
            ids.append(results_tuple[0])
            dkai = dka_index(results_tuple[1]['iob'], results_tuple[1]['sbr'])
            risk_assessment_array[0][items] = dkai
            risk_assessment_array[1][items] = dka_risk_score(dkai)
            lbgi, _, _ = blood_glucose_risk_index(results_tuple[1]['bg'])
            risk_assessment_array[2][items] = lbgi
            risk_assessment_array[3][items] = lbgi_risk_score(lbgi)
            risk_assessment_array[4][items] = percent_values_lt_40(results_tuple[1]['bg'])
        risk_assessment = pd.DataFrame(risk_assessment_array, columns=ids, index=['dkai', 'dkai_risk_score', 'lbgi',
                                                                                  'lbgi_risk_score',
                                                                                  'percent_less_than_40'])
    print(file_name)
    print(risk_assessment)
    plot_sim_results(all_results, save=False)



if __name__ == "__main__":

    scenarios_folder_path = "/Users/shawnfoster/Documents/py4e/data-science-simulator/data/raw/fda_risk_scenarios"
    scenario_file_names = os.listdir(scenarios_folder_path)

    for file_name in scenario_file_names[0:]:
        scenario_csv_filepath = os.path.join(
            scenarios_folder_path, file_name
        )
        risk_analysis_tlr337_user_delete_data(scenario_csv_filepath)
