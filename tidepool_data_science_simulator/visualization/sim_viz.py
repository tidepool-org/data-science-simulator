__author__ = "Cameron Summers"

import os
import datetime
import numpy as np
import itertools

import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

# style.use("seaborn-poster")  # sets the size of the charts
# style.use("ggplot")

sns.set_style("darkgrid")


def plot_sim_icgm_paired(all_results):

    for sim_id, ctrl_result_df in all_results.items():
        if "Ideal" in sim_id:
            plt.plot(ctrl_result_df["bg"].to_numpy(), label="True Glucose - Ideal", color="black")
            plt.plot(ctrl_result_df["bg_sensor"].to_numpy(), label="CGM - Ideal", color="grey", marker="^", markersize=6, alpha=0.7)
        else:
            plt.plot(ctrl_result_df["bg"].to_numpy(), label="True Glucose - iCGM", color="purple")
            plt.plot(ctrl_result_df["bg_sensor"].to_numpy(), label="iCGM", color="green", marker="^", markersize=6, alpha=0.7)

    plt.legend()
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
        ax[0].set_xlabel("Time (5min)")
        ax[0].set_ylabel("BG (mg/dL)")
        ax[0].set_ylim((0, 400))

        if len(all_results) <= n_sims_max_legend:
            ax[0].legend(prop={'size': 6})

        # ====== Insulin ============

        ax[1].set_title("Insulin")
        ax[1].set_ylabel("Insulin (U or U/hr)")
        ax[1].set_xlabel("Time (5 mins)")
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
            ax[1].legend(prop={'size': 12})

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
        ax[2].set_xlabel("Time (5 mins)")
        ax[2].set_ylim((0, 100))
        if len(all_results) <= n_sims_max_legend:
            ax[2].legend(prop={'size': 6})

    if save:
        if save_path is None:
            save_path = "./data-science-simulator-image_{}.png".format(datetime.datetime.now().isoformat())
        plt.savefig(save_path)
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
        ax[0].legend()

        ax[1].plot(ctrl_result_df["sbr"], label="{} {}".format("sbr", sim_id), color="gray")
        ax[1].set_ylabel("Insulin (U or U/hr)")
        ax[1].set_xlabel("Time (5 mins)")
        ax[1].set_title("Insulin Delivery")
        ax[1].plot(ctrl_result_df["temp_basal"], label="{} {}".format("tmp_br", sim_id), color="green")
        ax[1].plot(ctrl_result_df["bolus"], label="{} {}".format("bolus", sim_id), color="brown")
        ax[1].set_ylim((0, 3))
        ax[1].legend()

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
