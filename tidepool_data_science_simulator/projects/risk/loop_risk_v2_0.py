__author__ = "Cameron Summers"

"""
This file builds and runs the Tidepool Loop Risk simulations. Scenarios in JSON format
specify how to build the simulations. 
"""

import os
import datetime
import pandas as pd
import subprocess
import json

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.utils import timing, PROJECT_ROOT_DIR, DATA_DIR
from tidepool_data_science_simulator.run import run_simulations


THIS_DIR = os.path.abspath(__file__)
TIDEPOOL_RISK_SCENARIOS_DIR = os.path.join(PROJECT_ROOT_DIR, "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/")

RESULTS_SAVE_DIR = os.path.join(DATA_DIR, "results/tidepool_loop_risk_v2_0")


@timing
def build_risk_sim_generator(scenario_json_filepath, override_config_save_dir=None):
    """
    Build a generator of suites of related simulations for processing.
    """
    risk_dirs = [risk_dir for risk_dir in os.listdir(TIDEPOOL_RISK_SCENARIOS_DIR) if "TLR-" in risk_dir]
    for risk_dir_name in risk_dirs:

        # for u!se in filtering to just one risk. If wanting to run all of them, comment out lines 35-37
        if ("TLR-682") not in risk_dir_name:
            continue
        print("!!!"+risk_dir_name)

        risk_dir_path = os.path.join(scenario_json_filepath, risk_dir_name)
        scenario_json_filenamess = [filename for filename in os.listdir(risk_dir_path) if ".json" in filename]

        for scenario_json_name in scenario_json_filenamess:
            # for use in filtering to just one file in a folder. If wanting to run all files, comment out lines 44-46
            #if "stress" not in scenario_json_name:
            #    continue
            # print("!!!"+scenario_json_name)
            scenario_json_path = os.path.join(risk_dir_path, scenario_json_name)
            parser = ScenarioParserV2(path_to_json_config=scenario_json_path)
            print(scenario_json_path)
            sim_suite = parser.get_sims(override_json_save_dir=override_config_save_dir)
            yield risk_dir_name, scenario_json_name, sim_suite


def create_save_dir():

    timestamp = get_timestamp()
    run_save_dir = os.path.join(RESULTS_SAVE_DIR, "Risk_Run_{}".format(timestamp))
    os.mkdir(run_save_dir)
    return run_save_dir


def get_timestamp():

    return datetime.datetime.now().isoformat()


if __name__ == "__main__":

    # Create place to save results
    run_save_dir = create_save_dir()

    # Build the scenarios
    sim_suite_generator = build_risk_sim_generator(TIDEPOOL_RISK_SCENARIOS_DIR, override_config_save_dir=run_save_dir)

    # Run the scenarios
    all_risk_results = []
    risk_run_metadata = {}
    for risk_name, scenario_json_name, sim_suite in sim_suite_generator:

        risk_result_dirpath = os.path.join(run_save_dir, risk_name)
        if not os.path.exists(risk_result_dirpath):
          os.mkdir(risk_result_dirpath)

        full_results_dict, summary_results_df = run_simulations(sim_suite,
                                                                save_dir=risk_result_dirpath,
                                                                save_results=True,
                                                                num_procs=4)
        summary_results_df["scenario_name"] = scenario_json_name
        summary_results_df["risk_name"] = risk_name

        # Save figure
        figure_filepath = os.path.join(risk_result_dirpath, "{}_{}_{}.png".format(risk_name, scenario_json_name, get_timestamp()))
        plot_sim_results(full_results_dict, save=True, save_path=figure_filepath)

        all_risk_results.append(summary_results_df)

    all_risk_results_df = pd.concat(all_risk_results)

    # Add high-level metadata
    # simulator_git_commit = subprocess.check_output(["git", "describe"]).strip()
    # risk_run_metadata["simulator_git_commit"] = simulator_git_commit
    risk_run_metadata["timestamp"] = get_timestamp()

    # Save the summaries
    all_risk_results_df.to_csv(os.path.join(run_save_dir, "Risk_Results_{}.csv".format(get_timestamp())))
    json.dump(risk_run_metadata, open(os.path.join(run_save_dir, "metadata.json"), "w"), indent=4)
