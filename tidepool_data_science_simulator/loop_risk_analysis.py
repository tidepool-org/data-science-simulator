__author__ = "Cameron Summers"

import os
import datetime

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel

from tidepool_data_science_simulator.models.simulation import Simulation
from tidepool_data_science_simulator.models.events import ActionTimeline, VirtualPatientDeleteLoopData
from tidepool_data_science_simulator.models.controller import DoNothingController, LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses, Omnipod, ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing
from tidepool_data_science_metrics.insulin.insulin import dka_index
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, lbgi_risk_score, percent_values_gt_180, percent_values_lt_40

import numpy as np
import pandas as pd

@timing
def compare_loop_to_pump_only(scenario_csv_filepath):
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

    controllers = [
        DoNothingController(
            time=t0, controller_config=sim_parser.get_controller_config()
        ),
        LoopController(
            time=t0,
            controller_config=sim_parser.get_controller_config(),
        ),
    ]

    all_results = {}
    for controller in controllers:
        sim_id = controller.name
        print("Running: {}".format(sim_id))

        # pump = OmnipodMissingPulses(time=t0, pump_config=sim_parser.get_pump_config())
        # pump = Omnipod(time=t0, pump_config=sim_parser.get_pump_config())
        pump = ContinuousInsulinPump(time=t0, pump_config=sim_parser.get_pump_config())

        sensor = IdealSensor(time=t0, sensor_config=sim_parser.get_sensor_config())
        # sensor = NoisySensor(sensor_config=sim_parser.get_sensor_config())

        patient_config = sim_parser.get_patient_config()
        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file
        patient_config.action_timeline = ActionTimeline()

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

    scenarios_folder_path = "../data/raw/fda_risk_scenarios"
    scenario_file_names = sorted(os.listdir(scenarios_folder_path))

    for file_name in scenario_file_names:
        if file_name.startswith('Scenario'):
            scenario_csv_filepath = os.path.join(
                scenarios_folder_path, file_name
            )
            compare_loop_to_pump_only(scenario_csv_filepath)

