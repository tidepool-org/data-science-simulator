__author__ = "Cameron Summers"

"""
This file to for running risk analysis of Tidepool Loop when a
there is a time difference when the patient boluses vs when it
was reported to Loop.

Tidepool Loop Risk Card: https://tidepool.atlassian.net/browse/TLR-315
"""

from datetime import timedelta

from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.patient import VirtualPatientCarbBolusAccept
from tidepool_data_science_simulator.models.measures import Bolus, Carb
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing


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
        reported_time = self.time + timedelta(minutes=5)
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

    delay_time_minutes_candidates = [30]#, 60, 90]

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

        t0, patient_config = get_canonical_risk_patient_config(accept_prob=1.0)
        t0, pump_config = get_canonical_risk_pump_config()
        t0, controller_config = get_canonical_controller_config()

        #patient_config.recommendation_accept_prob = 1.0  # Note: Important use here
        patient_config.min_bolus_rec_threshold = 0.5

        carb = Carb(20, "g", 180)
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

    plot_sim_results(all_results, save=False)


if __name__ == "__main__":

    risk_analysis_tlr315_bolus_report_time_difference()
