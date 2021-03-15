__author__ = "Cameron Summers"

import pdb
import os
import datetime as dt
import time
import numpy as np

import logging

logger = logging.getLogger(__name__)



def get_bernoulli_trial_uniform_step_prob(num_trials, prob_of_occurring):
    """
    Given an event has a probability P of happening in a set number of trials,
    what is the trial bias B that the coin should have to yield P
    on average of events occurring?

    For meals:
    P(meal=False) = (1 - B) ^ (num_trials)
    P(meal=True) = 1 - P(meal=False) = 1 - (1 - B) ^ num_trials = prob_of_occuring

    1 - prob_of_occurring = (1 - B) ^ num_trials
    (1 - prob_of_occurring) ^ -num_trials = 1 - B
    B = 1 - (1 - prob_of_occurring) ^ -num_trials

    Parameters
    ----------
    num_trials: int
        How many trials are happening

    prob_of_occurring: float
        Probability of event

    Returns
    -------
    float
        Bias
    """

    return 1.0 - np.power(1 - prob_of_occurring, 1.0 / num_trials)


def findDiff(d1, d2, path=""):
    """
    Utility function for debugging that prints the difference between two nested dictionaries.

    Parameters
    ----------
    d1: dict
    d2
    path

    Returns
    -------

    """
    for k in d1:
        if k not in d2:
            print(path, ":")
            print(k + " as key not in d2", "\n")
        else:
            if type(d1[k]) is dict:
                if path == "":
                    path = k
                else:
                    path = path + "->" + k
                findDiff(d1[k], d2[k], path)
            else:
                if d1[k] != d2[k]:
                    print(path, ":")
                    print(" - ", k, " : ", d1[k])
                    print(" + ", k, " : ", d2[k])


def get_equivalent_isf(total_delta_bg, basal_rates=None):
    """
    For a given change in bg over some time and list of basal rates, return the complementary
     ISFs that achieve it.

    Parameters
    ----------
    total_delta_bg
    basal_rates

    Returns
    -------
    list

    """
    if basal_rates is None:
        basal_rates = np.arange(0.1, 1.0, 0.1)

    isfs = []
    for br in basal_rates:
        isfs.append(total_delta_bg / (br + 1.0))

    return isfs


def timing(f):
    """
    Util decorator for timing functions
    """
    def wrap(*args, **kwargs):
        time1 = time.time()
        ret = f(*args, **kwargs)
        time2 = time.time()
        logger.debug(
            "{:s} function took {:.3f} ms".format(f.__name__, (time2 - time1) * 1000.0)
        )

        return ret

    return wrap


def save_df(df_results, analysis_name, save_dir, save_type="tsv"):
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    filename = "{}".format(analysis_name, utc_string)
    path = os.path.join(save_dir, filename)
    if "tsv" in save_type:
        df_results.to_csv("{}.tsv".format(path), sep="\t")
    else:
        df_results.to_csv("{}.csv".format(path))
    logger.debug("Saving sim to {}...".format(path))


def get_sim_results_save_dir(description):
    this_dir = os.path.dirname(os.path.realpath(__file__))
    utc_string = dt.datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S")
    results_dir = "../data/results/simulations/{}/{}".format(description, utc_string)
    abs_path = os.path.join(this_dir, results_dir)
    os.makedirs(abs_path)
    return abs_path
