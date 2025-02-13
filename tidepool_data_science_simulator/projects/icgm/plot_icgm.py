import os
import datetime
from pathlib import Path
import numpy as np
import itertools
import pandas as pd
# import seaborn as sns
import matplotlib.dates as mdates

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tidepool_data_science_simulator.evaluation.inspect_results import load_result


result_directory = Path('/Users/mconn/Library/CloudStorage/GoogleDrive-mark.connolly@tidepool.org/My Drive/projects/Autobolus/')
result_pattern = "vp_35_tbg=45_sbg=70_TEMP_BASAL_no_RC.tsv"
# result_pattern = "vp_35_tbg=40_sbg=120_no_RC.tsv"
# result_pattern = "vp_35_tbg=40_sbg=120_with_RC.tsv"
# result_pattern = "vp_2_tbg=110_sbg=250.tsv"
# result_pattern = "vp_2_tbg=110_sbg=150.tsv"
# result_pattern = "vp_35_tbg=45_sbg=50.tsv"

result_path = result_directory / result_pattern

(sim_id, sim_results_df) = load_result(result_path, ext="tsv")

# sim_results_df = sim_results_df[sim_results_df.index >= datetime.datetime(2019, 8, 15, 12, 0, 0)]

fig, ax = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
color_cycle = itertools.cycle(mcolors.BASE_COLORS)

sim_color = next(color_cycle)

ax[0].plot(sim_results_df["bg"],
            label="{}".format("True Blood Glucose"),
            color=sim_color,
            linestyle="dashed",
            alpha=0.5)
ax[0].plot(sim_results_df.index.to_pydatetime(), sim_results_df["bg_sensor"],
            label="{}".format("iCGM Blood Glucose"),
            color=sim_color,
            markersize=4,
            marker=".",
            linestyle="none")

ax[0].set_title("Blood Glucose Over Time")
ax[0].set_ylabel("Blood Glucose (mg/dL)")
ax[0].set_ylim((0, 300))
ax[0].legend()

# ====== Insulin ============
ax[1].set_title("Insulin")
ax[1].set_ylabel("Insulin (U or U/hr)")
ax[1].set_xlabel("Time (5 mins)")
ax[1].plot(sim_results_df.index.to_pydatetime(), sim_results_df["sbr"],
            label="{}".format("Scheduled Basal Rate"),
            linestyle="dotted",
            color=sim_color,
            alpha=0.5)
ax[1].plot(sim_results_df.index.to_pydatetime(), sim_results_df["temp_basal"],
            label="{}".format("Temporary Basal Rate"),
            linestyle="-.",
            color=sim_color)
ax[1].stem(sim_results_df.index.to_pydatetime(), sim_results_df["true_bolus"],
            linefmt='{}-'.format(sim_color),
            label="{}".format("True Bolus"),
            markerfmt='{}P'.format(sim_color))
ax[1].stem(sim_results_df.index.to_pydatetime(), sim_results_df["reported_bolus"],
            linefmt='{}--'.format(sim_color),
            markerfmt='{}X'.format(sim_color),
            label="{}".format("Reported Bolus"))
ax[1].plot(sim_results_df.index.to_pydatetime(), sim_results_df["iob"],
            label="{}".format("Insulin on Board"),
            color=sim_color,
            alpha=0.5)
ax[1].set_ylim((0, 12))

ax[1].set_xlabel("Time")
ax[1].set_xlim((datetime.datetime(2019,8,15,11,30), datetime.datetime(2019,8,15,20)))
time_format = mdates.DateFormatter('%H:%M')  # Specify the desired time format (e.g., HH:MM)
ax[1].xaxis.set_major_formatter(time_format)

ax[1].legend()


if 0:
    if save_path is None:
        save_path = "./data-science-simulator-image_{}.png".format(datetime.datetime.now().isoformat())
    plt.savefig(save_path)
else:
    plt.show()
