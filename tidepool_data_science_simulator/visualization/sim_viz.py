__author__ = "Cameron Summers"

import os
import datetime
import numpy as np
import itertools

# import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from tidepool_data_science_simulator.evaluation.inspect_results import load_result

# style.use("seaborn-poster")  # sets the size of the charts
# style.use("ggplot")

# sns.set_style("darkgrid")


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


def plot_sim_results(all_results, save=False, n_sims_max_legend=5, save_path=None, plot_cumulative_insulin=False):
    """
    Default multi-sim plot
    """
    if plot_cumulative_insulin:
        fig, ax = plt.subplots(4, 1, figsize=(8, 10), sharex=True)
    else:
        fig, ax = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    color_cycle = itertools.cycle(mcolors.BASE_COLORS)

    # Track max time for xlim
    max_time = 0
    
    for sim_id, ctrl_result_df in all_results.items():

        sim_color = next(color_cycle)
        
        # Convert datetime index to hours from 12:00 (noon)
        # Set reference time to 12:00 on the same day as the first timestamp
        first_timestamp = ctrl_result_df.index[0]
        reference_time = datetime.datetime(first_timestamp.year, first_timestamp.month, first_timestamp.day, 12, 0, 0)
        time_hours = [(t - reference_time).total_seconds() / 3600 for t in ctrl_result_df.index.to_pydatetime()]
        
        # Track max time for setting xlim
        if time_hours:
            max_time = max(max_time, max(time_hours))

        ax[0].plot(time_hours, ctrl_result_df["bg"],
                   label="{} {}".format("bg", sim_id),
                   color=sim_color,
                   linestyle="dashed",
                   alpha=0.5)
        ax[0].plot(time_hours, ctrl_result_df["bg_sensor"],
                   label="{} {}".format("bg_sensor", sim_id),
                   color=sim_color,
                   markersize=4,
                   marker=".",
                   linestyle="none")

        ax[0].set_title("BG Over Time")
        ax[0].set_ylabel("BG (mg/dL)")
        ax[0].set_ylim((0, 400))

        if len(all_results) <= n_sims_max_legend:
            ax[0].legend(prop={'size': 6}, loc='upper right')

        # ====== Insulin ============

        ax[1].set_title("Insulin")
        ax[1].set_ylabel("Insulin (U or U/hr)")
        ax[1].plot(time_hours, ctrl_result_df["sbr"],
                   label="{} {}".format("sbr", sim_id),
                   linestyle="dotted",
                   color=sim_color,
                   alpha=0.5)
        ax[1].plot(time_hours, ctrl_result_df["temp_basal"],
                   label="{} {}".format("tmp_br", sim_id),
                   linestyle="-.",
                   color=sim_color)
        ax[1].stem(time_hours, ctrl_result_df["true_bolus"],
                   linefmt='{}-'.format(sim_color),
                   label="{} {}".format("true bolus", sim_id),
                   markerfmt='{}P'.format(sim_color))
        ax[1].stem(time_hours, ctrl_result_df["reported_bolus"],
                   linefmt='{}--'.format(sim_color),
                   markerfmt='{}X'.format(sim_color),
                   label="{} {}".format("reported bolus", sim_id))
        ax[1].plot(time_hours, ctrl_result_df["iob"],
                   label="{} {}".format("iob", sim_id),
                   color=sim_color,
                   alpha=0.5)
        
        # Only plot ei if data is not empty and not all zeros
        if "ei" in ctrl_result_df.columns:
            ei_data = ctrl_result_df["ei"].dropna()  # Remove NaN values
            if len(ei_data) > 0 and (ei_data != 0).any():  # Check if any non-zero values exist
                ax[1].plot(time_hours, ctrl_result_df["ei"] * 12,
                           label="{} {}".format("ei", sim_id),
                           linestyle="dashed",
                           color=sim_color,
                           alpha=0.5)
        ax[1].set_ylim((0, 8))

        if len(all_results) <= n_sims_max_legend:
            ax[1].legend(prop={'size': 6}, loc='upper right')
        
        # ======== Carbs ============
        ax[2].stem(time_hours,
                   ctrl_result_df["true_carb_value"],
                   linefmt='{}-'.format(sim_color),
                   label="{} {}".format("true carb", sim_id),
                   markerfmt='{}P'.format(sim_color))
        ax[2].stem(time_hours,
                   ctrl_result_df["reported_carb_value"],
                   linefmt='{}--'.format(sim_color),
                   markerfmt='{}X'.format(sim_color),
                   label="{} {}".format("reported carb", sim_id))
        ax[2].set_title("Carb Events")
        ax[2].set_ylabel("Carbs (g)")
        # ax[2].set_xlabel("Time (hours)")
        ax[2].set_ylim((0, 100))
        
        if plot_cumulative_insulin:
            # ======== Cumulative Insulin ============
            cumulative_insulin = np.cumsum(
                np.nan_to_num(ctrl_result_df["temp_basal"]/12, nan=0.0) +
                np.nan_to_num(ctrl_result_df["true_bolus"], nan=0.0)
            )
            ax[3].plot(time_hours,
                     cumulative_insulin,
                     label="{} {}".format("cumulative insulin", sim_id),
                     color=sim_color)
            ax[3].set_title("Cumulative Insulin Over Time")
            ax[3].set_ylabel("Cumulative Insulin (U)")
            ax[3].set_xlabel("Time (hours)")

        if len(all_results) <= n_sims_max_legend:
            ax[2].legend(prop={'size': 6}, loc='lower right')
            
        if plot_cumulative_insulin and len(all_results) <= n_sims_max_legend:
            ax[3].legend(prop={'size': 6}, loc='lower right')
    
    # Add subfigure labels (a, b, c, d)
    labels = ['A', 'B', 'C', 'D']
    for i, axis in enumerate(ax):
        axis.text(-0.1, 1.1, labels[i], transform=axis.transAxes,
                 fontsize=12, va='top')
    
    # Set x-axis limits for all subplots
    for axis in ax:
        axis.set_xlim((-0.5, max_time))

    if save:
        if save_path is None:
            save_path = "./data-science-simulator-image_{}.png".format(datetime.datetime.now().isoformat())
        plt.savefig(save_path)
    else:
        pass #plt.show()

    return fig, ax


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

def load_and_plot_tsv(tsv_path: str, save: bool = False, save_path: str = None):
    """
    Load a TSV file and plot simulation results.
    
    Args:
        tsv_path: Path to the TSV file
        save: Whether to save the plot instead of displaying it
        save_path: Path to save the plot (if save=True)
    """
    # Load the TSV file
    sim_id, result_df = load_result(tsv_path, ext="tsv")
    
    # Wrap in dictionary format expected by plot_sim_results
    all_results = {sim_id: result_df}
    
    # Plot the results
    plot_sim_results(all_results, save=save, save_path=save_path)

def calculate_cumulative_insulin(results_df):
    """
    Calculate cumulative insulin delivered over time.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Simulation results dataframe
        
    Returns
    -------
    np.array
        Cumulative insulin in units
    """
    # Initialize cumulative insulin
    cumulative_insulin = np.zeros(len(results_df))
    
    # Calculate insulin delivered at each timestep
    # Temp basal is in U/hr, need to convert to U per 5-min interval
    temp_basal_delivered = results_df['temp_basal'].values / 12  # 5 min = 1/12 hr
    
    # Bolus insulin
    bolus_delivered = results_df['true_bolus'].values
    
    # Convert NaN to 0
    temp_basal_delivered = np.nan_to_num(temp_basal_delivered, nan=0.0)
    bolus_delivered = np.nan_to_num(bolus_delivered, nan=0.0)
    
    # Total insulin at each timestep
    total_insulin_per_step = temp_basal_delivered + bolus_delivered
    
    # Calculate cumulative sum
    cumulative_insulin = np.cumsum(total_insulin_per_step)
    
    return cumulative_insulin

if __name__ == "__main__":
    data_dir = "/Users/mconn/data/simulator/processed_data/insulin_algorithm_testing_framework/icgm_spurious/with_gradual_transition_mitigation/icgm_sensitivity_analysis_paf=0.4_posrc=True_gradthresh=20.0_2025_11_03_T_19_56_16_3747cf91/"
    file_name = "icgm_analysis_vp_35_65a71e48e64827646838032c8d29d3ac91ac36da94de018be6f668559ff4f9c2_tbg=40_sbg=100.tsv"
    file_path = os.path.join(data_dir, file_name)

    load_and_plot_tsv(file_path, save=False)
