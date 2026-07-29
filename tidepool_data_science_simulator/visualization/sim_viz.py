__author__ = "Cameron Summers"

import os
import datetime
import numpy as np
import itertools

# import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.dates as mdates

# style.use("seaborn-poster")  # sets the size of the charts
# style.use("ggplot")

# sns.set_style("darkgrid")

# ============ Centralized Legend Configuration ============
# Standardized legend settings for consistent, readable charts
LEGEND_CONFIG = {
    'fontsize': 8,           # Balanced size: readable but compact
    'framealpha': 0.9,       # Slight transparency to see data underneath
    'loc': 'upper right',    # Default position
    'borderaxespad': 0.5,    # Padding from axes
}

# For charts with many entries (e.g., Insulin chart with 6 entries per sim)
LEGEND_CONFIG_DENSE = {
    **LEGEND_CONFIG,
    'fontsize': 7,           # Slightly smaller for dense legends
    'ncol': 2,               # Multiple columns to reduce vertical footprint
    'loc': 'upper right',
}


# ============ X-Axis Datetime Formatting ============
class MidnightDateFormatter(mdates.DateFormatter):
    """
    Custom formatter that shows time (HH:MM) for all ticks,
    but adds the date below only at midnight crossings.
    """
    def __init__(self, time_fmt='%H:%M', date_fmt='%-m/%d'):
        super().__init__(time_fmt)
        self.time_fmt = time_fmt
        self.date_fmt = date_fmt

    def __call__(self, x, pos=None):
        dt = mdates.num2date(x)
        time_str = dt.strftime(self.time_fmt)
        # Show date only at midnight (00:00)
        if dt.hour == 0 and dt.minute == 0:
            date_str = dt.strftime(self.date_fmt)
            return f"{time_str}\n{date_str}"
        return time_str


def configure_datetime_axis(ax, interval_hours=2):
    """
    Configure x-axis to display timestamps at specified intervals
    with date shown only at midnight crossings.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to configure
    interval_hours : int
        Hour interval for major ticks (default: 2)
    """
    # Set major ticks at specified hour intervals
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=interval_hours))

    # Use custom formatter: time always, date only at midnight
    ax.xaxis.set_major_formatter(MidnightDateFormatter())

    # Rotate labels slightly for readability
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right', fontsize=8)


def plot_sim_icgm_paired(all_results):

    for sim_id, ctrl_result_df in all_results.items():
        if "Ideal" in sim_id:
            plt.plot(ctrl_result_df["bg"].to_numpy(), label="True Glucose - Ideal", color="black")
            plt.plot(ctrl_result_df["bg_sensor"].to_numpy(), label="CGM - Ideal", color="grey", marker="^", markersize=6, alpha=0.7)
        else:
            plt.plot(ctrl_result_df["bg"].to_numpy(), label="True Glucose - iCGM", color="purple")
            plt.plot(ctrl_result_df["bg_sensor"].to_numpy(), label="iCGM", color="green", marker="^", markersize=6, alpha=0.7)

    plt.legend(**LEGEND_CONFIG)
    plt.title("Example: Positive Bias iCGM Paired Simulation")
    plt.xlabel("Time (5 min)")
    plt.ylabel("BG (mg/dL)")
    plt.ylim((0, 400))
    plt.show()


def plot_sim_results(all_results, save=False, n_sims_max_legend=5, save_path=None):
    """
    Default multi-sim plot
    """

    fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    color_cycle = itertools.cycle(mcolors.BASE_COLORS)

    for sim_id, ctrl_result_df in all_results.items():

        sim_color = next(color_cycle)

        ax[0].plot(ctrl_result_df["bg"],
                   label="{} {}".format("bg", sim_id),
                   color=sim_color,
                   linestyle="dashed",
                   alpha=0.5)
        ax[0].plot(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["bg_sensor"],
                   label="{} {}".format("bg_sensor", sim_id),
                   color=sim_color,
                   markersize=4,
                   marker=".",
                   linestyle="none")

        ax[0].set_title("BG Over Time")
        ax[0].set_xlabel("Time")
        ax[0].set_ylabel("BG (mg/dL)")
        ax[0].set_ylim((0, 400))

        if len(all_results) <= n_sims_max_legend:
            ax[0].legend(**LEGEND_CONFIG)

        # ====== Insulin ============

        ax[1].set_title("Insulin")
        ax[1].set_ylabel("Insulin (U or U/hr)")
        ax[1].set_xlabel("Time")
        ax[1].plot(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["sbr"],
                   label="{} {}".format("sbr", sim_id),
                   linestyle="dotted",
                   color=sim_color,
                   alpha=0.5)
        ax[1].plot(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["temp_basal"],
                   label="{} {}".format("tmp_br", sim_id),
                   linestyle="-.",
                   color=sim_color)
        ax[1].stem(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["true_bolus"],
                   linefmt='{}-'.format(sim_color),
                   label="{} {}".format("true bolus", sim_id),
                   markerfmt='{}P'.format(sim_color))
        ax[1].stem(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["reported_bolus"],
                   linefmt='{}--'.format(sim_color),
                   markerfmt='{}X'.format(sim_color),
                   label="{} {}".format("reported bolus", sim_id))
        ax[1].plot(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["iob"],
                   label="{} {}".format("iob", sim_id),
                   color=sim_color,
                   alpha=0.5)
        ax[1].plot(ctrl_result_df.index.to_pydatetime(), ctrl_result_df["ei"] * 12,
                   label="{} {}".format("ei", sim_id),
                   linestyle="dashed",
                   color=sim_color,
                   alpha=0.5)
        ax[1].set_ylim((0, 8))

        if len(all_results) <= n_sims_max_legend:
            ax[1].legend(**LEGEND_CONFIG_DENSE)

        # ======== Carbs ============
        ax[2].stem(ctrl_result_df.index.to_pydatetime(),
                   ctrl_result_df["true_carb_value"],
                   linefmt='{}-'.format(sim_color),
                   label="{} {}".format("true carb", sim_id),
                   markerfmt='{}P'.format(sim_color))
        ax[2].stem(ctrl_result_df.index.to_pydatetime(),
                   ctrl_result_df["reported_carb_value"],
                   linefmt='{}--'.format(sim_color),
                   markerfmt='{}X'.format(sim_color),
                   label="{} {}".format("reported carb", sim_id))
        ax[2].set_title("Carb Events")
        ax[2].set_ylabel("Carbs (g)")
        ax[2].set_xlabel("Time")
        ax[2].set_ylim((0, 100))
        ax[2].set_xlim((datetime.datetime(2019,8,15,11,30), datetime.datetime(2019,8,16,12)))
                       
        if len(all_results) <= n_sims_max_legend:
            ax[2].legend(**LEGEND_CONFIG)

    # Configure x-axis datetime formatting (applies to all axes via sharex)
    configure_datetime_axis(ax[2], interval_hours=2)
    ax[2].set_xlabel("Time")  # Update label since format now shows HH:MM
    fig.tight_layout()

    if save:
        if save_path is None:
            save_path = "./data-science-simulator-image_{}.png".format(datetime.datetime.now().isoformat())
        # fig.savefig (not the global plt.savefig) so this always saves the figure
        # built above, never whatever pyplot currently considers "current" -- and
        # close it afterward so repeated calls in the same process (e.g. once per
        # scenario file in a GUI run) can't leak figure state into the next call.
        # Only done in the save path -- callers using save=False (e.g. existing
        # tests) inspect the current figure via plt.gcf() right after this returns.
        fig.savefig(save_path)
        plt.close(fig)
    else:
        plt.show()


def plot_sim_results_missing_insulin(all_results):

    fig, ax = plt.subplots(4, 1, figsize=(16, 20))
    for sim_id, ctrl_result_df in all_results.items():
        ax[0].scatter(range(len(ctrl_result_df['time'])), ctrl_result_df["bg"],
                   label="{} {}".format("bg", sim_id),
                   color="purple",
                      s=6)
        ax[0].scatter(range(len(ctrl_result_df['time'])), ctrl_result_df["bg_sensor"],
                      label="{} {}".format("bg", sim_id),
                      color="green",
                      s=6)
        ax[0].set_title("BG Over Time")
        ax[0].set_xlabel("Time (5min)")
        ax[0].set_ylabel("BG (mg/dL)")
        ax[0].set_ylim((0, 400))
        median = ctrl_result_df["bg"].median()
        std = round(ctrl_result_df["bg"].std())
        # ax[0].axhline(median, label="BG Median {}".format(median), color="green")
        # ax[0].axhline(median + std, label="BG Std {}".format(std), color="green")
        # ax[0].axhline(median - std, label="BG Std {}".format(std), color="green")
        ax[0].legend(**LEGEND_CONFIG)

        ax[1].plot(ctrl_result_df["sbr"], label="{} {}".format("sbr", sim_id), color="gray")
        ax[1].set_ylabel("Insulin (U or U/hr)")
        ax[1].set_xlabel("Time (5 mins)")
        ax[1].set_title("Insulin Delivery")
        ax[1].plot(ctrl_result_df["temp_basal"], label="{} {}".format("tmp_br", sim_id), color="green")
        ax[1].plot(ctrl_result_df["bolus"], label="{} {}".format("bolus", sim_id), color="brown")
        ax[1].set_ylim((0, 3))
        ax[1].legend(**LEGEND_CONFIG)

        ax[2].stem(ctrl_result_df["delivered_basal_insulin"],
                   label="{} {}".format("delivered_basal", sim_id), linefmt="C1-")
        ax[2].set_title("Delivered Basal Insulin")
        ax[2].set_ylabel("Insulin (U)")
        ax[2].set_xlabel("Time (5 mins)")

        ax[3].stem(ctrl_result_df["undelivered_basal_insulin"],
                   label="{} {}".format("undelivered_basal", sim_id), linefmt="C4-")
        ax[3].set_title("Undelivered Basal Insulin")
        ax[3].set_ylabel("Insulin (U)")
        ax[3].set_xlabel("Time (5 mins)")

        print(
            "Patient Bg min {} max {}".format(
                ctrl_result_df["bg"].min(), ctrl_result_df["bg"].max()
            )
        )

        delivered_sum = np.sum(ctrl_result_df["delivered_basal_insulin"])
        undelivered_sum = np.sum(ctrl_result_df["undelivered_basal_insulin"])
        total = delivered_sum + undelivered_sum
        print("Delivered Basal", delivered_sum, delivered_sum / total)
        print("Undelivered Basal", undelivered_sum, undelivered_sum / total)

    plt.show()
