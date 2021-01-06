__author__ = "Cameron Summers"

"""
This file to for running risk analysis of Tidepool Loop when a
there is a time difference when the patient eats carbs vs when it
was reported to Loop.

Tidepool Loop Risk Card: https://tidepool.atlassian.net/browse/TLR-342
"""

from datetime import timedelta

from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatientCarbBolusAccept
from tidepool_data_science_simulator.models.measures import Bolus, Carb
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing
from tidepool_data_science_metrics.insulin.insulin import dka_index
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, lbgi_risk_score, percent_values_lt_40

import numpy as np
import pandas as pd

@timing
def risk_analysis_tlr342_bolus_report_time_difference():
    """
    Compare loop running with an action and without that action.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    carb_value_candidates = [25.0]
    delay_time_minutes_candidates = [-60]

    param_grid = [
        {
            "carb_value": carb_value,
            "delay_time_minutes": delay_time_minutes
        }
        for carb_value in carb_value_candidates
        for delay_time_minutes in delay_time_minutes_candidates
    ]

    sims = dict()
    for params in param_grid:
        carb_value = params["carb_value"]
        delay_time_minutes = params["delay_time_minutes"]

        sim_id = "tlr315_delay_{}_carb_{}".format(delay_time_minutes, carb_value)
        print("Running: {}".format(sim_id))

        sim_num_hours = 8

        t0, patient_config = get_canonical_risk_patient_config()
        t0, pump_config = get_canonical_risk_pump_config()

        patient_config.recommendation_accept_prob = 1.0  # NOTE. Using Loop bolus recommendation
        patient_config.min_bolus_rec_threshold = 0.5

        # User reports Carb
        carb = Carb(25, "g", 180)
        reported_carb_time = t0 + timedelta(minutes=5)
        pump_config.carb_event_timeline.add_event(reported_carb_time, carb, input_time=reported_carb_time)

        # User eats Carb at a different time than was reported.
        actual_carb_time = reported_carb_time + timedelta(minutes=delay_time_minutes)
        patient_config.carb_event_timeline.add_event(actual_carb_time, carb)

        t0, sim = get_canonical_simulation(
            t0=t0,
            patient_config=patient_config,
            patient_class=VirtualPatientCarbBolusAccept,
            pump_config=pump_config,
            controller_class=LoopController,
            multiprocess=True,
            duration_hrs=sim_num_hours,
        )

        sims[sim_id] = sim
        sim.start()

    all_results = {id: sim.queue.get() for id, sim in sims.items()}
    [sim.join() for id, sim in sims.items()]

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

    plot_sim_results(all_results, save=False)


if __name__ == "__main__":

    risk_analysis_tlr342_bolus_report_time_difference()
