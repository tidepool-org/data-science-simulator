__author__ = "Cameron Summers"

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.style as style

style.use("seaborn-poster")  # sets the size of the charts
style.use("ggplot")


def plot_sim_results(all_results, save=False):

    # ==== TMP ====
    # TODO - This is a placeholder for dev. Replace with viz tools module.
    fig, ax = plt.subplots(3, 1, figsize=(16, 20))
    for sim_id, ctrl_result_df in all_results.items():

        ax[0].plot(ctrl_result_df["bg"], label="{} {}".format("bg", sim_id))
        ax[0].plot(ctrl_result_df["bg_sensor"], label="{} {}".format("bg_sensor", sim_id))
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

        ax[1].plot(ctrl_result_df["sbr"], label="{} {}".format("sbr", sim_id))
        ax[1].set_title("Insulin")
        ax[1].set_ylabel("Insulin (U or U/hr)")
        ax[1].set_xlabel("Time (5 mins)")
        ax[1].plot(ctrl_result_df["temp_basal"], label="{} {}".format("tmp_br", sim_id))
        ax[1].plot(ctrl_result_df["bolus"], label="{} {}".format("bolus", sim_id))
        ax[1].plot(ctrl_result_df["iob"], label="{} {}".format("iob", sim_id))
        ax[1].set_ylim((0, 3))
        ax[1].legend()

        ax[2].plot(ctrl_result_df["carb"], label="{} {}".format("carb", sim_id))
        ax[2].set_title("Carb Events")
        ax[2].set_ylabel("Carbs (g)")
        ax[2].set_xlabel("Time (5 mins)")
        ax[2].set_ylim((0, 40))
        ax[2].legend()

        print(
            "Patient Bg min {} max {}".format(
                ctrl_result_df["bg"].min(), ctrl_result_df["bg"].max()
            )
        )

        log_bg = np.log(ctrl_result_df["bg"])
        geo_mean = np.mean(log_bg)
        geo_var = np.var(log_bg)

        # counts, bins, patches = ax[2].hist(log_bg, bins=50, label="{} {} {}".format("bg", vp_name, ctr_name), alpha=0.1)
        # # ax[2].set_xscale("log")
        # ax[2].set_xticklabels(np.exp(bins).astype(int))
        # ax[2].axvline(geo_mean, label="{} {} {}".format("Geo Mean", vp_name, ctr_name))
        # ax[2].set_xlabel("BG (mg/dL)")
        # ax[2].legend()
        #
        # counts, bins, patches = ax[3].hist(ctrl_result_df['bg'], bins=50, label="{} {}".format("bg", sim_id), alpha=0.1)
        # ax[3].set_xscale("log")
        # ax[3].set_title("BG Distribution")
        # ax[3].set_xlabel("BG (mg/dL)")
        # ax[3].legend()


    if save:
        plt.savefig("data-science-simulator-image.png")
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
