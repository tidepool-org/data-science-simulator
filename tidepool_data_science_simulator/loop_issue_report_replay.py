__author__ = "Cameron Summers"

from tidepool_data_science_simulator.makedata.issue_report_sim_parser import LoopIssueReportSimParser
from tidepool_data_science_simulator.models.simulation import Simulation

from tidepool_data_science_simulator.models.patient.real_patient import RealPatient
from tidepool_data_science_simulator.models.controller import LoopController, DoNothingController
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.models.sensor import RealSensor
from tidepool_data_science_simulator.models.pump import RealPump


def replay_loop_issue_report(issue_report_path):

    all_results = dict()
    sim_id = "0"
    sim_parser = LoopIssueReportSimParser(issue_report_path)

    t0 = sim_parser.get_simulation_start_time()
    patient_config = sim_parser.get_patient_config()
    pump_config = sim_parser.get_pump_config()
    sensor_config = sim_parser.get_sensor_config()
    controller_config = sim_parser.get_controller_config()

    real_pump = RealPump(t0, pump_config=pump_config)
    real_sensor = RealSensor(t0, sensor_config=sensor_config)
    real_patient = RealPatient(t0,
                               pump=real_pump,
                               sensor=real_sensor,
                               patient_config=patient_config)

    # controller = DoNothingController(t0, controller_config=controller_config)
    controller = LoopController(t0, controller_config=controller_config)

    simulation = Simulation(
        time=t0,
        duration_hrs=24.0,
        virtual_patient=real_patient,
        controller=controller,
    )

    simulation.run()

    results_df = simulation.get_results_df()
    all_results[sim_id] = results_df

    plot_sim_results(all_results, save=False)


def simulate_with_issue_report_seed(issue_report_path):

    sim_parser = LoopIssueReportSimParser(issue_report_path)

    patient_config = sim_parser.get_patient_config()
    pump_config = sim_parser.get_pump_config()
    sensor_config = sim_parser.get_sensor_config()


if __name__ == "__main__":

    issue_report_path = "/Users/csummers/Downloads/PHI-Loop Issue Report files for Tidepool - 2020-05-29/LOOP-1221/Loop Report 2019-10-06 16_37_29-07_00.md"
    replay_loop_issue_report(issue_report_path)
