__author__ = "Cameron Summers"

import time
import numpy as np


def get_bernoulli_trial_uniform_step_prob(num_trials, prob_of_occurring):
    """
    Given an event has a probability P of happening in a set number of trials,
    what is the bias B that the coin should have to yield P on average?

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
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print(
            "{:s} function took {:.3f} ms".format(f.__name__, (time2 - time1) * 1000.0)
        )

        return ret

    return wrap