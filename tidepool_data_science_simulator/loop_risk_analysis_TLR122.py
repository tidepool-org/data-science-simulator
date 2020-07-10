__author__ = "Cameron Summers"

"""
This file to for running risk analysis of Tidepool Loop when there is a
gap between pump sessions.

Tidepool Loop Risk Card: https://tidepool.atlassian.net/browse/TLR-122
"""

from datetime import timedelta

from tidepool_data_science_simulator.models.controller import LoopController
from tidepool_data_science_simulator.models.measures import ManualBolus
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient_config, get_canonical_risk_pump_config
from tidepool_data_science_simulator.makedata.make_simulation import get_canonical_simulation
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing

from tidepool_data_science_simulator.models.events import VirtualPatientAttachPump, VirtualPatientRemovePump


@timing
def risk_analysis_tlr122_pump_session_gap():
    """
    Compare two controllers for a given scenario file:
        1. No controller, ie no insulin modulation except for pump schedule
        2. Loop controller

    Parameters
    ----------
    scenario_csv_filepath: str
        Path to the scenario file
    """

    all_results = {}

    pump_break_duration_hrs = [i for i in range(5, 7)]

    for break_duration_hrs in pump_break_duration_hrs:

        sim_id = "tlr122_duration_{}".format(break_duration_hrs)
        print("Running: {}".format(sim_id))

        sim_num_hours = 24

        t0, patient_config = get_canonical_risk_patient_config()
        t0, pump_config = get_canonical_risk_pump_config()

        patient_config.recommendation_accept_prob = 0.0  # TODO: put in scenario file

        # Remove Pump Event
        remove_pump_time = t0 + timedelta(minutes=120)
        user_remove_pump_action = VirtualPatientRemovePump("Remove Pump")
        patient_config.action_timeline.add_event(remove_pump_time, user_remove_pump_action)

        # Add Pump Event
        attach_pump_time = remove_pump_time + timedelta(hours=break_duration_hrs)
        user_attach_pump_action = VirtualPatientAttachPump("Attach Pump", ContinuousInsulinPump, pump_config)
        patient_config.action_timeline.add_event(attach_pump_time, user_attach_pump_action)

        # Manual Bolus Event
        bolus_time = remove_pump_time + timedelta(hours=break_duration_hrs / 2)
        manual_bolus = ManualBolus(1.0, "U")
        patient_config.bolus_event_timeline.add_event(bolus_time, manual_bolus)

        t0, sim = get_canonical_simulation(
            t0=t0,
            patient_config=patient_config,
            controller_class=LoopController,
            multiprocess=True,
            duration_hrs=sim_num_hours,
        )

        sim.run()
        results_df = sim.get_results_df()
        all_results[sim_id] = results_df

    plot_sim_results(all_results, save=False)


if __name__ == "__main__":

    risk_analysis_tlr122_pump_session_gap()
