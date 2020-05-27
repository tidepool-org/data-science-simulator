__author__ = "Cameron Summers"

from collections import defaultdict
import numpy as np


def get_first_order_indices(param_names, sim_id_params, all_results):
    """
    Sobol method - First Order Indices

    Equations from https://en.wikipedia.org/wiki/Variance-based_sensitivity_analysis

    TODO: CS: This is exploration and is not complete (or even correct).
    """

    params_outcomes = [
        (sim_id_params[sim_id], np.mean(np.log(all_results[sim_id]["bg"])))
        for sim_id in all_results.keys()
    ]

    V_i = []
    for param_name in param_names:
        y_given_x = defaultdict(list)
        for pgrid, y in params_outcomes:
            y_given_x[pgrid[param_name]].append(y)

        V = np.var([np.mean(ys) for x_val, ys in y_given_x.items()])
        V_i.append(V)

    V_y = np.sum(V_i)
    S_i = [V / V_y for V in V_i]
    return list(zip(param_names, S_i))
