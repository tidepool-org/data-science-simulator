__author__ = "Shawn Foster"

"""This file takes the results from the risk simulations run in loop_risk_v2_0.py and compiles the severity
level across profiles for each state (pre-mitigation Loop, pump only, and post-mitigation Loop)."""

# import libraries
import pandas as pd
import glob
import os
from tidepool_data_science_simulator.utils import  PROJECT_ROOT_DIR, DATA_DIR

# locate files - can be obsolete if i can just import the correct variable from loop_risk_v2.0
THIS_DIR = os.path.abspath(__file__)
TIDEPOOL_RISK_SCENARIOS_DIR = os.path.join(PROJECT_ROOT_DIR, "scenario_configs/tidepool_risk_v2/loop_risk_v2.0/")
RESULTS_SAVE_DIR = os.path.join(DATA_DIR, "results/tidepool_loop_risk_v2.0")
directory = max(RESULTS_SAVE_DIR)

# concatenate csv files
for root, subdirectories in os.walk(directory):
    for subdirectory in subdirectories:
        joined_files = os.path.join(subdirectory, "summary_results*.csv")
        joined_list = glob.glob(joined_files)
        df = pd.concat(map(pd.read_csv, joined_list), ignore_index=True)

# determine which is more pertinent, overdelivery or underdelivery

# underdelivery severity

# overdelivery severity

# return results