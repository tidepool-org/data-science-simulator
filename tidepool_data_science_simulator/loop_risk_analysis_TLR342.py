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


@timing
def risk_analysis_tlr342_bolus_report_time_difference():
    """
    Compare loop running with an action and without that action.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    carb_value_candidates = [20.0]
    delay_time_minutes_candidates = [30]#, 60, 90]

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

        sim_num_hours = 24

        t0, patient_config = get_canonical_risk_patient_config(accept_prob=1.0)
        t0, pump_config = get_canonical_risk_pump_config()

        #patient_config.recommendation_accept_prob = 1.0  # NOTE. Using Loop bolus recommendation
        patient_config.min_bolus_rec_threshold = 0.5

        # User reports Carb
        carb = Carb(20, "g", 180)
        reported_carb_time = t0 + timedelta(minutes=30)
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

    plot_sim_results(all_results, save=False)


if __name__ == "__main__":

    risk_analysis_tlr342_bolus_report_time_difference()
