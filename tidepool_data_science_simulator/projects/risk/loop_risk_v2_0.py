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
    print(f"build_risk_sim_generator called with scenario_json_filepath: {scenario_json_filepath}")
    print(f"TIDEPOOL_RISK_SCENARIOS_DIR: {TIDEPOOL_RISK_SCENARIOS_DIR}")

    # Define the base directory and selected subdirectory
    BASE_DIR = os.path.join(PROJECT_ROOT_DIR, "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/")
    SELECTED_SUBDIR = "loop_risk_v2_510k"  # Change this to select different subdirectories

    # Construct the full path to the selected subdirectory
    SELECTED_DIR_PATH = os.path.join(BASE_DIR, SELECTED_SUBDIR)
    print(f"Selected directory path: {SELECTED_DIR_PATH}")
    print(f"Selected directory exists: {os.path.exists(SELECTED_DIR_PATH)}")

    # Get all TLR- directories within the selected directory
    risk_dirs = [risk_dir for risk_dir in os.listdir(SELECTED_DIR_PATH)
                 if os.path.isdir(os.path.join(SELECTED_DIR_PATH, risk_dir)) and risk_dir.startswith("TLR-")]
    print(f"Found risk directories: {risk_dirs}")

    for risk_dir_name in risk_dirs:
        print(f"Processing risk directory: {risk_dir_name}")
        #for use in filtering to just one risk. If wanting to run all of them, comment out lines 35-37
        if ("TLR-1053") not in risk_dir_name:
            print(f"Skipping {risk_dir_name} as it doesn't contain 'TLR-1049'")
            continue
        print(f"Processing: {risk_dir_name}")

        risk_dir_path = os.path.join(SELECTED_DIR_PATH, risk_dir_name)
        scenario_json_filenames = [filename for filename in os.listdir(risk_dir_path) if ".json" in filename]
        print(f"JSON files found in {risk_dir_name}: {scenario_json_filenames}")

        for scenario_json_name in scenario_json_filenames:
            print(f"Processing JSON file: {scenario_json_name}")
            scenario_json_path = os.path.join(risk_dir_path, scenario_json_name)
            parser = ScenarioParserV2(path_to_json_config=scenario_json_path)
            print(f"Parsing: {scenario_json_path}")
            sim_suite = parser.get_sims(override_json_save_dir=override_config_save_dir)
            print(f"Yielding: {risk_dir_name}, {scenario_json_name}")
            yield risk_dir_name, scenario_json_name, sim_suite

    print("build_risk_sim_generator completed")


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

    try:
        for risk_name, scenario_json_name, sim_suite in sim_suite_generator:
            risk_result_dirpath = os.path.join(run_save_dir, risk_name)
            if not os.path.exists(risk_result_dirpath):
                os.mkdir(risk_result_dirpath)

            try:
                full_results_dict, summary_results_df = run_simulations(sim_suite,
                                                                        save_dir=risk_result_dirpath,
                                                                        save_results=True,
                                                                        num_procs=4)

                if summary_results_df.empty:
                    print(f"Warning: Empty summary results for {risk_name}, {scenario_json_name}")
                    # Create a dummy result for debugging
                    summary_results_df = pd.DataFrame({'dummy': [1]})

                summary_results_df["scenario_name"] = scenario_json_name
                summary_results_df["risk_name"] = risk_name

                # Save figure
                figure_filepath = os.path.join(risk_result_dirpath,
                                               "{}_{}_{}.png".format(risk_name, scenario_json_name, get_timestamp()))
                plot_sim_results(full_results_dict, save=True, save_path=figure_filepath)

                all_risk_results.append(summary_results_df)
                print(f"Processed: {risk_name}, {scenario_json_name}")
            except Exception as e:
                print(f"Error processing {risk_name}, {scenario_json_name}: {str(e)}")

    except Exception as e:
        print(f"Error in main loop: {str(e)}")

    print(f"Total scenarios processed: {len(all_risk_results)}")

    if not all_risk_results:
        print("No results were generated. Creating a dummy result for debugging.")
        dummy_df = pd.DataFrame({'dummy': [1]})
        all_risk_results.append(dummy_df)

    try:
        all_risk_results_df = pd.concat(all_risk_results)
        print(f"Final dataframe shape: {all_risk_results_df.shape}")
    except Exception as e:
        print(f"Error concatenating results: {str(e)}")

    # Add high-level metadata
    risk_run_metadata["timestamp"] = get_timestamp()

    # Save the summaries
    try:
        all_risk_results_df.to_csv(os.path.join(run_save_dir, "Risk_Results_{}.csv".format(get_timestamp())))
        print("Results saved to CSV")
    except Exception as e:
        print(f"Error saving results to CSV: {str(e)}")

    json.dump(risk_run_metadata, open(os.path.join(run_save_dir, "metadata.json"), "w"), indent=4)

    all_risk_results_df = pd.concat(all_risk_results)

    # Add high-level metadata
    # simulator_git_commit = subprocess.check_output(["git", "describe"]).strip()
    # risk_run_metadata["simulator_git_commit"] = simulator_git_commit
    risk_run_metadata["timestamp"] = get_timestamp()

    # Save the summaries
    all_risk_results_df.to_csv(os.path.join(run_save_dir, "Risk_Results_{}.csv".format(get_timestamp())))
    json.dump(risk_run_metadata, open(os.path.join(run_save_dir, "metadata.json"), "w"), indent=4)
