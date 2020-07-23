__author__ = "Cameron Summers"


__author__ = "Cameron Summers"

import json
import os

from loop.issue_report import parser
from pyloopkit import pyloop_parser

from tidepool_data_science_simulator.makedata.scenario_parser import ScenarioParserCSV


class LoopIssueReportSimParser(ScenarioParserCSV):
    """
    Parser that transforms a Loop issue report into simulation configuration.
    """
    def __init__(self, loop_issue_report_path, sim_time_at_start=True):

        self.report_path = loop_issue_report_path

        lr = parser.LoopReport()
        directory, filename = os.path.split(loop_issue_report_path)
        parsed_issue_report_dict = lr.parse_by_file(directory, filename)
        with open('issue_report.json', 'w') as file:
            file.write(json.dumps(parsed_issue_report_dict))
        pyloopkit_inputs = pyloop_parser.transform_report_to_inputs('issue_report.json')

        # Make a list of units to match the scenario format
        pyloopkit_inputs["basal_rate_units"] = [pyloopkit_inputs["basal_rate_units"] for _ in pyloopkit_inputs["basal_rate_values"]]
        pyloopkit_inputs["dose_value_units"] = [pyloopkit_inputs["dose_value_units"] for _ in pyloopkit_inputs["dose_values"]]
        pyloopkit_inputs["carb_value_units"] = [pyloopkit_inputs["carb_value_units"] for _ in pyloopkit_inputs["carb_values"]]
        pyloopkit_inputs["sensitivity_ratio_value_units"] = [pyloopkit_inputs["sensitivity_ratio_value_units"] for _ in pyloopkit_inputs["sensitivity_ratio_values"]]
        pyloopkit_inputs["carb_ratio_value_units"] = [pyloopkit_inputs["carb_ratio_value_units"] for _ in pyloopkit_inputs["carb_ratio_values"]]
        pyloopkit_inputs["target_range_value_units"] = [pyloopkit_inputs["target_range_value_units"] for _ in pyloopkit_inputs["target_range_minimum_values"]]
        self.loop_inputs_dict = pyloopkit_inputs

        self.start_time = self.loop_inputs_dict["time_to_calculate_at"]
        if sim_time_at_start:
            self.start_time = self.loop_inputs_dict["glucose_dates"][0]  # beginning of issue report

        self.transform_patient(self.start_time)
        self.transform_pump(self.start_time)
        self.transform_sensor()

    def transform_patient(self, time):
        """
        Loop issue report gives no true patient information.
        """
        self.patient_basal_schedule = None
        self.patient_carb_ratio_schedule = None
        self.patient_insulin_sensitivity_schedule = None
        self.patient_target_range_schedule = None
        self.patient_carb_events = None
        self.patient_bolus_events = None
        self.patient_glucose_history = None

    def get_simulation_start_time(self):

        return self.start_time
