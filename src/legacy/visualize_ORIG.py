# -*- coding: utf-8 -*-


"""
+CS General Comments

- Doc strings with plain language description and justifications of algos would be helpful.

- Having a hard time following the flow in some cases. Maybe we can
use a naming convention for some variables, e.g. iob_sbr_pred for a time
series prediction of insulin on board from scheduled basal rate.

- Maybe we can move all arithmetic to vars, e.g. 8*12 and sbr / 12, to make more
general and descriptive of what's going on.


"""

# LOAD LIBRARIES, FUNCTIONS, AND DATA
import os
import numpy as np
import plotly.express as px

from src.models.loop_simulation import simulate_loop
from src.data.read_input_scenarios import transform_input_scenario_to_simulation_df

# %% CREATE PATHS, DATAFRAMES, AND LOAD SCENARIO
# select a scenario scenario
path = os.path.join(".", "example_files")
# load in example scenario files

scenario_file_names = {
    0: "Scenario-0-simulation-template",
    1: "Scenario-1-sensor-inaccurate - no file for this one",
    2: "Scenario-2-watch-comm",
    3: "Scenario-3-accessibility",
    4: "Scenario-4-insulin-rationing",
    5: "Scenario-5-EM-interference",
    6: "Scenario-6-malware-bolus",
    7: "Scenario-7-bolus-cancel-fails",
    8: "Scenario-8-Loop-loss - no file for this one.csv",
    9: "Scenario-9-double-carb-entry",
    10: "Scenario-10-1A-sensor-inaccurate",
    11: "Scenario-11-1B-sensor-inaccurate",
    12: "Scenario-12-calc-iob-from-sbr",
}

plot = False
for scenario_number in [2, 3, 4, 5, 6, 7, 9, 10, 11, 12]:
    print(scenario_number)
    # %%  USER INPUTS
    # scenario_number = 0  # see list below
    simulation_duration_hours = 8  # default and minimum is 8 hours
    scenario_file_name = "{} - inputs.tsv".format(scenario_file_names[scenario_number])
    scenario_filepath = os.path.join("scenarios", scenario_file_name)

    print("risk of scenario in a pump only or mdi situation:")
    scenario_results, inputs_from_file = transform_input_scenario_to_simulation_df(
        scenario_filepath, simulation_duration_hours
    )

    print(
        "simulating the scenario through pyloopkit over {} hours:".format(
            simulation_duration_hours
        )
    )
    sim_df = simulate_loop(inputs_from_file)

    if plot:
        # plot results of how situation would play out with pump or mdi situation

        fig = px.line(
            x=[0],
            y=[0],
            labels={"x": "Time (minutes)", "y": "BG (mg/dL)"},
            title="Expected Outcome if Scenario happened on Pump or MDI",
        )
        fig.add_scatter(
            x=ts_array_metab, y=bg_timeseries, name="Expected BG Every 5 Minutes"
        )
        fig.add_scatter(
            x=ts_array_metab,
            y=delta_bg_array_metab,
            name="Expected Change in BG Every 5 Minutes",
        )
        fig.show()

        # %% VISUALIZE THE RESULTS
        fig = px.line(
            x=[0],
            y=[0],
            labels={"x": "Time (minutes)", "y": "BG (mg/dL)"},
            title="Simulation Results (Resulting BGs) for {}".format(
                scenario_file_names[scenario_number]
            ),
        )
        actual_carbs = sim_df.loc[0, "carbActual"].astype(int)
        actual_dose = sim_df.loc[0, "insulinActual"].astype(int)
        fig.add_scatter(
            x=[0],
            y=sim_df.loc[0, ["bg_actual"]],
            name="Time of {}g Carbs & {}U Insulin".format(actual_carbs, actual_dose),
        )
        fig.add_scatter(x=sim_df.index, y=sim_df["bg_actual"], name="Actual BGs")
        fig.add_scatter(x=sim_df.index, y=sim_df["pump_bgs"], name="Pump BGs")
        fig.show()

        fig = px.line(
            x=[0],
            y=[0],
            labels={"x": "Time (minutes)", "y": "Insulin (U or U/hr)"},
            title="Simulation Results (Insulin) for {}".format(
                scenario_file_names[scenario_number]
            ),
        )
        fifty_percent_steady_state_iob = (
            get_steady_state_iob_from_sbr(sbr_actual_scalar) / 2
        )
        fig.add_scatter(x=sim_df.index, y=sim_df["temp_basal"], name="Temp Basals")
        fig.add_scatter(
            x=sim_df.index,
            y=np.repeat(fifty_percent_steady_state_iob, len(sim_df)),
            name="1/2 SBR IOB",
        )
        fig.add_scatter(
            x=sim_df.index,
            y=sim_df["insulin_relative_to_actual_basal"],
            name="Impact of Insulin",
        )
        fig.add_scatter(x=sim_df.index, y=sim_df["iob"], name="Insulin On Board")

        fig.show()

    # simulation output dataframe
    sim_df
