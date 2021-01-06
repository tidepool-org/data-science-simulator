__author__ = "Cameron Summers"

"""
This file to for running risk analysis of Tidepool Loop when a
there is a time difference when the patient boluses vs when it
was reported to Loop.

Tidepool Loop Risk Card: https://tidepool.atlassian.net/browse/TLR-315
"""

from datetime import timedelta

import numpy as np
import pandas as pd
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, lbgi_risk_score, \
    percent_values_gt_180
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score

from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient_config, \
    get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.measures import Carb
from tidepool_data_science_simulator.models.patient import VirtualPatientCarbBolusAccept
from tidepool_data_science_simulator.utils import timing
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_metrics.insulin.insulin import dka_index, dka_risk_score
from tidepool_data_science_metrics.glucose.glucose import blood_glucose_risk_index, lbgi_risk_score, percent_values_gt_180, percent_values_lt_40


class LoopBolusRecMalfunctionDelay(LoopController):
    """
    A Loop controller that exhibits a malfunction delay in the delivery of a
    recommended bolus. It thinks the bolus was delivered at a particular
    time but the patient gets it at a later time.
    """

    def __init__(self, time, controller_config):
        super().__init__(time, controller_config)

    def set_bolus_recommendation_event(self, virtual_patient, bolus):
        """
                Add the accepted bolus event to the virtual patient's timeline to
                be applied at the next update.

                Parameters
                ----------
                virtual_patient
                bolus
                """
        delay_minutes = self.controller_config.bolus_rec_delay_minutes
        reported_time = self.time + timedelta(minutes=0)
        delivered_time = reported_time + timedelta(minutes=delay_minutes)

        # Add to patient timeline
        virtual_patient.bolus_event_timeline.add_event(delivered_time, bolus)

        # Log in pump, which Loop will read at update
        virtual_patient.pump.bolus_event_timeline.add_event(reported_time, bolus)


@timing
def risk_analysis_tlr315_bolus_report_time_difference():
    """
    Compare loop running with an action and without that action.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """

    delay_time_minutes_candidates = [60, 90, 120, 180, 270, 360] #, 60, 90]

    param_grid = [
        {
            "delay_time_minutes": delay_time_minutes
        }
        for delay_time_minutes in delay_time_minutes_candidates
    ]

    sims = dict()
    for params in param_grid:

        delay_time_minutes = params["delay_time_minutes"]

        sim_id = "tlr315_delay_{}".format(delay_time_minutes)
        print("Running: {}".format(sim_id))

        sim_num_hours = 24

        t0, patient_config = get_canonical_risk_patient_config()
        t0, pump_config = get_canonical_risk_pump_config()
        t0, controller_config = get_canonical_controller_config()

        patient_config.recommendation_accept_prob = 1.0  # Note: Important use here
        patient_config.min_bolus_rec_threshold = 0.5

        carb = Carb(25, "g", 180)
        carb_time = t0
        pump_config.carb_event_timeline.add_event(carb_time, carb)
        patient_config.carb_event_timeline.add_event(carb_time, carb)

        controller_config.bolus_rec_delay_minutes = delay_time_minutes

        t0, sim = get_canonical_simulation(
            t0=t0,
            patient_config=patient_config,
            pump_config=pump_config,
            patient_class=VirtualPatientCarbBolusAccept,
            controller_class=LoopBolusRecMalfunctionDelay,
            controller_config=controller_config,
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
        risk_assessment_array = np.zeros((6, len(all_results)))
        ids = []
        for items, results_tuple in enumerate(all_results.items()):
            ids.append(results_tuple[0])
            dkai = dka_index(results_tuple[1]['iob'], results_tuple[1]['sbr'])
            risk_assessment_array[0][items] = dkai
            risk_assessment_array[1][items] = dka_risk_score(dkai)
            lbgi, _, _ = blood_glucose_risk_index(results_tuple[1]['bg'])
            risk_assessment_array[2][items] = lbgi
            risk_assessment_array[3][items] = lbgi_risk_score(lbgi)
            risk_assessment_array[4][items] = percent_values_gt_180(results_tuple[1]['bg'])
            risk_assessment_array[5][items] = percent_values_lt_40(results_tuple[1]['bg'])
        risk_assessment = pd.DataFrame(risk_assessment_array, columns=ids, index=['dkai', 'dkai_risk_score', 'lbgi',
                                                                                  'lbgi_risk_score',
                                                                                  'percent_greater_than_180', 'percent_less_than_40'])

    print(risk_assessment)
    plot_sim_results(all_results, save=False)



if __name__ == "__main__":

    risk_analysis_tlr315_bolus_report_time_difference()
