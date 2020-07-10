__author__ = "Cameron Summers"

"""
This file to for running risk analysis of Tidepool Loop when a
there is a time difference when the patient boluses vs when it
was reported to Loop.

Tidepool Loop Risk Card: https://tidepool.atlassian.net/browse/TLR-315
"""

from datetime import timedelta

from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.measures import Bolus, Carb
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


@timing
def risk_analysis_tlr315_bolus_report_time_difference():
    """
    Compare loop running with an action and without that action.

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """
    bolus_value_candidates = [1.0]#, 2.0]
    delay_time_minutes_candidates = [0, 30]#, 60, 90]

    param_grid = [
        {
            "bolus_value": bolus_value,
            "delay_time_minutes": delay_time_minutes
        }
        for bolus_value in bolus_value_candidates
        for delay_time_minutes in delay_time_minutes_candidates
    ]

    sims = dict()
    for params in param_grid:
        bolus_value = params["bolus_value"]
        delay_time_minutes = params["delay_time_minutes"]

        sim_id = "tlr315_delay_{}_bolus_{}".format(delay_time_minutes, bolus_value)
        print("Running: {}".format(sim_id))

        sim_num_hours = 24

        t0, patient_config = get_canonical_risk_patient_config()
        t0, pump_config = get_canonical_risk_pump_config()

        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file

        bolus = Bolus(bolus_value, "U")
        carb = Carb(20, "g", 180)

        # Reported Bolus
        reported_bolus_time = t0 + timedelta(minutes=30)
        pump_config.bolus_event_timeline.add_event(reported_bolus_time, bolus)
        pump_config.carb_event_timeline.add_event(reported_bolus_time, carb)

        # Delivered Bolus
        delivered_bolus_time = reported_bolus_time + timedelta(minutes=delay_time_minutes)
        patient_config.bolus_event_timeline.add_event(delivered_bolus_time, bolus)
        patient_config.carb_event_timeline.add_event(reported_bolus_time, carb)

        t0, sim = get_canonical_simulation(
            t0=t0,
            patient_config=patient_config,
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

    risk_analysis_tlr315_bolus_report_time_difference()
