__author__ = "Cameron Summers"

from tidepool_data_science_simulator.visualization.sim_viz import plot_sim_results
from tidepool_data_science_simulator.evaluation.inspect_results import load_results


if __name__ == "__main__":

    test_run_results = "../../../data/results/simulations/guardrails_gsl_tr/2021_02_10_19_33_43"  # from compute-1

    all_results = load_results(test_run_results, max_dfs=16)

    plot_sim_results(all_results)