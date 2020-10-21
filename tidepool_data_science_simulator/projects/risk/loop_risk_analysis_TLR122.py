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
from tidepool_data_science_simulator.makedata.make_controller import get_canonical_controller_config
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing

from tidepool_data_science_simulator.models.events import VirtualPatientAttachPump, VirtualPatientRemovePump

from tidepool_data_science_simulator.run import run_simulations

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

    param_grid = [
        {
            "pump_session_gap_hrs": pump_session_gap_hrs,
            "do_report_manual_bolus": do_report_manual_bolus,
            "manual_bolus_value": manual_bolus_value
        }
        for pump_session_gap_hrs in [6]
        for do_report_manual_bolus in [True, False]
        for manual_bolus_value in [1.0]
    ]

    sims = dict()
    for pgrid in param_grid:
        pump_session_gap_hrs = pgrid["pump_session_gap_hrs"]
        do_report_manual_bolus = pgrid["do_report_manual_bolus"]
        manual_bolus_value = pgrid["manual_bolus_value"]

        sim_id = "tlr122_duration_{}_reported_{}".format(pump_session_gap_hrs, do_report_manual_bolus)
        print("Running: {}".format(sim_id))

        sim_num_hours = 24

        t0, patient_config = get_canonical_risk_patient_config()
        t0, pump_config = get_canonical_risk_pump_config()
        t0, controller_config = get_canonical_controller_config()

        patient_config.recommendation_accept_prob = 0.0

        # Remove Pump Event
        remove_pump_time = t0 + timedelta(minutes=120)
        user_remove_pump_action = VirtualPatientRemovePump("Remove Pump")
        patient_config.action_timeline.add_event(remove_pump_time, user_remove_pump_action)

        # Add Pump Event
        attach_pump_time = remove_pump_time + timedelta(hours=pump_session_gap_hrs)
        user_attach_pump_action = VirtualPatientAttachPump("Attach Pump", ContinuousInsulinPump, pump_config)
        patient_config.action_timeline.add_event(attach_pump_time, user_attach_pump_action)

        # Manual Bolus Event
        bolus_time = remove_pump_time + timedelta(hours=pump_session_gap_hrs / 2)
        manual_bolus = ManualBolus(manual_bolus_value, "U")
        patient_config.bolus_event_timeline.add_event(bolus_time, manual_bolus)

        # User reports manual bolus event
        if do_report_manual_bolus:
            controller_config.bolus_event_timeline.add_event(bolus_time, manual_bolus)

        t0, sim = get_canonical_simulation(
            t0=t0,
            patient_config=patient_config,
            controller_class=LoopController,
            controller_config=controller_config,
            multiprocess=True,
            duration_hrs=sim_num_hours,
        )

        sims[sim_id] = sim

    return sims


if __name__ == "__main__":

    sims = risk_analysis_tlr122_pump_session_gap()
    all_results = run_simulations(sims, results_dir="./")
    plot_sim_results(all_results, save=False)

