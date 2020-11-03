__author__ = "Cameron Summers"

import pdb
import os
import datetime
import time
import json

from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.sensor import SensorBase
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace


iCGM_THRESHOLDS = {
    "A": 0.85,
    "B": 0.7,
    "C": 0.8,
    "D": 0.98,
    "E": 0.99,
    "F": 0.99,
    "G": 0.87,
    "J": 0.01,
    "K": 0.01,
}

G6_THRESHOLDS = {
    "A": 85.40255821075945,
    "B": 74.93381359722157,
    "C": 84.9952769987456,
    "D": 98.02131957706338,
    "E": 99.23213676012934,
    "F": 99.84860393926321,
    "G": 0.87,
    "J": 0.01,
    "K": 0.01,
}


class iCGMState():
    """
    State object for iCGM sensor to manage state space in stochastic process.
    """
    def __init__(self, time, num_initial_values, num_history_values, conf_lower_bounds, random_state):

        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.RandomState(0)

        self.time = time

        self.prev_true_bg = None
        self.prev_sensor_bg = None

        self.num_current_values = num_initial_values
        self.num_history_values = num_history_values

        self.special_controls = conf_lower_bounds

        self.special_controls["AD_complement"] = 1.0 - self.special_controls["D"]
        self.special_controls["BE_complement"] = 1.0 - self.special_controls["E"]
        self.special_controls["CF_complement"] = 1.0 - self.special_controls["F"]
        self.special_controls["G_complement"] = 1.0 - self.special_controls["G"]
        self.special_controls["J_complement"] = 1.0 - self.special_controls["J"]
        self.special_controls["K_complement"] = 1.0 - self.special_controls["K"]

        self.criteria_to_key_map = {
            "range": {
                "A": ("<70", "<15"),
                "B": ("70-180", "<15%"),
                "C": (">180", "<15%"),
                "D": ("<70", "15-40"),
                "E": ("70-180", "15%-40%"),
                "F": (">180", "15%-40%"),
                "AD_complement": ("<70", ">40"),
                "BE_complement": ("70-180", ">40%"),
                "CF_complement": (">180", ">40%"),
            },
            "overall": {
                "G": "<20%",
                "G_complement": ">20%",
            },
            "true_pos_rate": {
                "J": "ok",
                "J_complement": "extreme",
            },
            "true_neg_rate": {
                "K": "ok",
                "K_complement": "extreme"
            }
        }

        self.state_history_fifo_keys = {
            "range": [],  # historical values in special controls criteria A-F
            "overall": [],  # historical values in special controls criteria G overall
            "true_pos_rate": [],  # historical values in special controls criteria J
            "true_neg_rate": []  # historical values in special controls criteria K
        }

        self.state_history = {
            "range": {
                "<70": {
                    "<15": 0,
                    "15-40": 0,
                    ">40": 0
                },
                "70-180": {
                    "<15%": 0,
                    "15%-40%": 0,
                    ">40%": 0
                },
                ">180": {
                    "<15%": 0,
                    "15%-40%": 0,
                    ">40%": 0
                },
            },
            "overall": {
                "<20%": 0,
                ">20%": 0
            },
            "true_neg_rate": {
                "ok": 0,
                "extreme": 0,
            },
            "true_pos_rate": {
                "ok": 0,
                "extreme": 0,
            }
        }

        self.initial_probabilities = {
            "range": {
                "<70": {
                    "<15": self.special_controls["A"],
                    "15-40": self.special_controls["D"] - self.special_controls["A"],
                    ">40": 1.0 - self.special_controls["D"]
                },
                "70-180": {
                    "<15%": self.special_controls["B"],
                    "15%-40%": self.special_controls["E"] - self.special_controls["B"],
                    ">40%": 1.0 - self.special_controls["E"]
                },
                ">180": {
                    "<15%": self.special_controls["C"],
                    "15%-40%": self.special_controls["F"] - self.special_controls["C"],
                    ">40%": 1.0 - self.special_controls["F"]
                },
            },
            "overall": {
                "<20%": self.special_controls["G"],
                ">20%": 1.0 - self.special_controls["G"]
            },
            "true_neg_rate": {
                "ok": 1.0 - self.special_controls["J"],
                "extreme": self.special_controls["J"],
            },
            "true_pos_rate": {
                "ok": 1.0 - self.special_controls["K"],
                "extreme": self.special_controls["K"],
            }
        }

        # Populate initial state - unfortunate wonkiness here due to different nesting depths, maybe fix later
        for criteria_type, criteria in self.criteria_to_key_map.items():

            for criteria_key, history_key in criteria.items():

                if criteria_type == "range":
                    initial_probability = self.initial_probabilities[criteria_type][history_key[0]][history_key[1]]
                else:
                    initial_probability = self.initial_probabilities[criteria_type][history_key]


                num_total_values = num_initial_values
                if criteria_type == "range":
                    num_total_values = int(num_initial_values / 3.0)

                num_values = int(num_total_values * initial_probability)

                for i in range(num_values):
                    self.state_history_fifo_keys[criteria_type].append(history_key)

                    if criteria_type == "range":
                        self.state_history[criteria_type][history_key[0]][history_key[1]] += 1
                    else:
                        self.state_history[criteria_type][history_key] += 1

                self.random_state.shuffle(self.state_history_fifo_keys[criteria_type])

    def prime_next_state(self):

        for criteria_type, criteria in self.criteria_to_key_map.items():
            fifo_key = self.state_history_fifo_keys[criteria_type][-1]

            if criteria_type == "range":
                self.state_history[criteria_type][fifo_key[0]][fifo_key[1]] -= 1
            else:
                self.state_history[criteria_type][fifo_key] -= 1


    def get_state_transition_probabilities(self, true_bg, sensor_candidate_range=range(40, 400)):

        icgm_probs = []

        self.prime_next_state()
        for candidate_icgm_bg in sensor_candidate_range:
            meets_special_controls, max_loss, failed_reason = self.compute_candidate_sensor_bg_conditional_probability(
                true_bg, candidate_icgm_bg)
            icgm_prob = int(meets_special_controls)

            icgm_probs.append(icgm_prob)

            if candidate_icgm_bg == true_bg and failed_reason != "":
                print(candidate_icgm_bg, true_bg, failed_reason)
                raise Exception("Perfect value failed to special controls.")

        if np.sum(icgm_probs) == 0:
            raise Exception("Error: No iCGM values available")

        icgm_probs = np.array(icgm_probs) / np.sum(icgm_probs)

        # TODO: add behavior here to icgm probabilities, ie worst case loss, max bias, etc.

        return icgm_probs

    def compute_candidate_sensor_bg_conditional_probability(self, true_bg, candidate_sensor_bg):
        """
        Compute the conditional probability of the candidate being icgm: P(candidate_bg | history_window)

        NOTE: The state of the history window has already removed the oldest sensor value.

        Parameters
        ----------
        true_bg
        candidate_sensor_bg

        Returns
        -------
            (float, str): (probability, outcome reason)
        """
        a_loss = 0
        b_loss = 0
        c_loss = 0
        d_loss = 0
        e_loss = 0
        f_loss = 0

        error = np.abs(true_bg - candidate_sensor_bg)
        error_percent = error / true_bg

        # Criteria H
        if true_bg < 70 and candidate_sensor_bg > 180:
            return 0.0, np.inf, "H"

        # Criteria I
        if true_bg > 180 and candidate_sensor_bg < 70:
            return 0.0, np.inf, "I"

        # Criteria A & D
        if candidate_sensor_bg < 70:
            range_key = "<70"
            denom = np.sum([v for k, v in self.state_history["range"][range_key].items()]) + 1  # keep this way for 0 data case

            a_LB = self.special_controls["A"]
            d_LB = self.special_controls["D"]

            a_loss = self.state_history["range"][range_key]["<15"] / denom - a_LB
            d_loss = (1 + self.state_history["range"][range_key]["<15"] + self.state_history["range"][range_key]["15-40"]) / denom - d_LB
            if error > 15:
                if self.state_history["range"][range_key]["<15"] / denom < a_LB:
                    return 0.0, np.inf, "A"
                elif (1 + self.state_history["range"][range_key]["<15"] + self.state_history["range"][range_key]["15-40"]) / denom < d_LB:
                    return 0.0, np.inf, "D"

        # Criteria B & E
        if 70 <= candidate_sensor_bg <= 180:
            range_key = "70-180"
            denom = np.sum([v for k, v in self.state_history["range"][range_key].items()]) + 1  # keep this way for 0 data case

            b_LB = self.special_controls["B"]
            e_LB = self.special_controls["E"]

            b_loss = (self.state_history["range"][range_key]["<15%"]) / denom - b_LB
            e_loss = (1 + self.state_history["range"][range_key]["<15%"] + self.state_history["range"][range_key]["15%-40%"]) / denom - e_LB
            if error_percent > 0.15:
                if (self.state_history["range"][range_key]["<15%"]) / denom < b_LB:
                    return 0.0, np.inf, "B"
                elif (1 + self.state_history["range"][range_key]["<15%"] + self.state_history["range"][range_key]["15%-40%"]) / denom < e_LB:
                    return 0.0, np.inf, "E"

        # Criteria C & F
        if candidate_sensor_bg > 180:
            range_key = ">180"
            gt180_denom = np.sum([v for k, v in self.state_history["range"][range_key].items()]) + 1  # keep this way for 0 data case

            c_LB = self.special_controls["C"]
            f_LB = self.special_controls["F"]

            c_loss = (self.state_history["range"][range_key]["<15%"]) / gt180_denom - c_LB
            f_loss = (1 + self.state_history["range"][range_key]["<15%"] + self.state_history["range"][range_key]["15%-40%"]) / gt180_denom - f_LB
            if error_percent > 0.15:
                if (self.state_history["range"][range_key]["<15%"]) / gt180_denom < c_LB:
                    return 0.0, np.inf, "C"
                elif (1 + self.state_history["range"][range_key]["<15%"] + self.state_history["range"][range_key]["15%-40%"]) / gt180_denom < f_LB:
                    return 0.0, np.inf, "F"

        # Criteria G
        g_LB = self.special_controls["G"]
        overall_denom = np.sum([v for k, v in self.state_history["overall"].items()])
        if error_percent > 0.2 and (1 + self.state_history["overall"][">20%"]) / overall_denom < g_LB:
            return 0.0, np.inf, "G"

        true_rate, cgm_rate = self.get_bg_rates(true_bg, candidate_sensor_bg)

        # Criteria J
        j_bound = self.special_controls["J"]
        neg_rate_denom = np.sum([v for k, v in self.state_history["true_neg_rate"].items()])
        if true_rate < -2.0 and cgm_rate > 1.0:
            if (1 + self.state_history["true_neg_rate"]["extreme"]) / neg_rate_denom > j_bound:
                return 0.0, np.inf, "J"

        # Criteria K
        k_bound = self.special_controls["K"]
        pos_rate_denom = np.sum([v for k, v in self.state_history["true_pos_rate"].items()])
        if true_rate > 2.0 and cgm_rate < -1.0:
            if (1 + self.state_history["true_pos_rate"]["extreme"]) / pos_rate_denom > k_bound:
                return 0.0, np.inf, "K"

        loss = min([a_loss, b_loss, c_loss, d_loss, e_loss, f_loss])
        return 1.0, loss, ""

    def update(self, time, true_bg, sensor_bg):

        # Update range state
        bg_range_key = self.get_bg_range_key(sensor_bg)
        bg_error_key = self.get_bg_range_error_key(true_bg, sensor_bg)
        self.state_history["range"][bg_range_key][bg_error_key] += 1
        self.state_history_fifo_keys["range"].insert(0, (bg_range_key, bg_error_key))

        # Update overall state
        bg_overall_error_key = self.get_overall_error_key(true_bg, sensor_bg)
        self.state_history["overall"][bg_overall_error_key] += 1
        self.state_history_fifo_keys["overall"].insert(0, (bg_overall_error_key))

        # Update true_pos rate state
        bg_rate_error_key = self.get_true_pos_rate_error_key(true_bg, sensor_bg)
        self.state_history["true_pos_rate"][bg_rate_error_key] += 1
        self.state_history_fifo_keys["true_pos_rate"].insert(0, (bg_rate_error_key))

        # Update true_pos rate state
        bg_rate_error_key = self.get_true_neg_rate_error_key(true_bg, sensor_bg)
        self.state_history["true_neg_rate"][bg_rate_error_key] += 1
        self.state_history_fifo_keys["true_neg_rate"].insert(0, (bg_rate_error_key))

        # Activate windowed memory once there are enough values
        if self.num_current_values >= self.num_history_values:
            self.state_history_fifo_keys["range"].pop()
            self.state_history_fifo_keys["overall"].pop()
            self.state_history_fifo_keys["true_pos_rate"].pop()
            self.state_history_fifo_keys["true_neg_rate"].pop()

        self.num_current_values += 1

        self.prev_true_bg = true_bg
        self.prev_sensor_bg = sensor_bg
        self.minutes_since_last_update = (self.time - time).total_seconds() / 60.0

    def display(self):

        # TODO: make this useful
        print(json.dumps(self.state_history, indent=4, sort_keys=True))

    def get_bg_error_pecentage(self, true_bg, sensor_bg):

        bg_error_percentage = np.abs(true_bg - sensor_bg) / true_bg
        return bg_error_percentage

    def get_bg_abs_error(self, true_bg, sensor_bg):

        bg_error = np.abs(true_bg - sensor_bg)
        return bg_error

    def get_bg_range_key(self, bg):

        if -np.inf < bg <= 70:
            key = "<70"
        elif 70 < bg <= 180:
            key = "70-180"
        elif 180 < bg < np.inf:
            key = ">180"
        else:
            raise Exception

        return key

    def get_bg_range_error_key(self, true_bg, sensor_bg):

        bg_range_key = self.get_bg_range_key(sensor_bg)

        if bg_range_key == "<70":
            bg_error = self.get_bg_abs_error(true_bg, sensor_bg)
            if 0 <= bg_error < 15:
                bg_error_key = "<15"
            elif 15 <= bg_error <= 40:
                bg_error_key = "15-40"
            elif bg_error > 40:
                bg_error_key = ">40"
            else:
                raise Exception

        else:
            bg_error_percentage = self.get_bg_error_pecentage(true_bg, sensor_bg)

            if 0 <= bg_error_percentage <= 0.15:
                bg_error_key = "<15%"
            elif 0.15 < bg_error_percentage <= 0.40:
                bg_error_key = "15%-40%"
            elif bg_error_percentage > 0.40:
                bg_error_key = ">40%"
            else:
                raise Exception

        return bg_error_key

    def get_overall_error_key(self, true_bg, sensor_bg):

        if self.get_bg_error_pecentage(true_bg, sensor_bg) > 0.2:
            return ">20%"
        else:
            return "<20%"

    def get_bg_rates(self, true_bg, sensor_bg):
        true_rate = 0.0
        cgm_rate = 0.0
        if self.prev_true_bg is not None and self.prev_sensor_bg is not None:
            true_rate = (true_bg - self.prev_true_bg) / self.minutes_since_last_update
            cgm_rate = (sensor_bg - self.prev_sensor_bg) / self.minutes_since_last_update

        return true_rate, cgm_rate

    def get_true_pos_rate_error_key(self, true_bg, sensor_bg):

        true_rate, cgm_rate = self.get_bg_rates(true_bg, sensor_bg)

        if true_rate > 2.0 and cgm_rate < -1.0:
            key = "extreme"
        else:
            key = "ok"

        return key

    def get_true_neg_rate_error_key(self, true_bg, sensor_bg):

        true_rate, cgm_rate = self.get_bg_rates(true_bg, sensor_bg)

        if true_rate < -2.0 and cgm_rate > 1.0:
            key = "extreme"
        else:
            key = "ok"

        return key


class SensoriCGM(SensorBase):
    """
    iCGM Sensor Model
    """
    def __init__(self, time, sensor_config, random_state=None):

        super().__init__(time, sensor_config)

        self.num_values_per_criteria_init = 300
        self.true_bg_history = []

        num_history_values = sensor_config.history_window_hrs * 12
        self.state = iCGMState(
            time,
            num_initial_values=300,
            num_history_values=num_history_values,
            conf_lower_bounds=sensor_config.special_controls,
            random_state=random_state
        )


        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.RandomState(0)

    def get_bg(self, true_bg):

        return self.current_sensor_bg

    def get_info_stateless(self):

        stateless_info = self.sensor_config.special_controls
        return stateless_info

    def update(self, time, **kwargs):

        self.time = time
        self.set_random_values()

        true_bg = kwargs["patient_true_bg"]
        self.true_bg_history.append(true_bg)

        sensor_candidate_range = self.sensor_config.sensor_range
        icgm_transition_probabilities = self.state.get_state_transition_probabilities(true_bg, sensor_candidate_range)

        if np.sum(icgm_transition_probabilities) == 0:
            raise Exception("No iCGM values available")

        behavior_probabilities = sensor_config.behavior_model.get_conditional_probabilities(
            sensor_candidate_range,
            self.true_bg_history
        )

        transition_probabilities = np.multiply(icgm_transition_probabilities, behavior_probabilities)
        transition_probabilities /= np.sum(transition_probabilities)

        do_plot_step_probs = kwargs.get("do_plot", False)
        if do_plot_step_probs or np.isnan(transition_probabilities).any():
            self.plot_internals(sensor_candidate_range,
                                icgm_transition_probabilities,
                                behavior_probabilities,
                                transition_probabilities)

        sensor_bg = self.random_state.choice(sensor_candidate_range, p=transition_probabilities)

        self.current_sensor_bg = sensor_bg

        self.sensor_bg_history.append(self.time, self.current_sensor_bg)
        self.state.update(time, true_bg, sensor_bg)

    def plot_internals(self, sensor_candidate_range,
                       icgm_transition_probabilities,
                       behavior_probabilities,
                       transition_probabilities):

        fig, ax = plt.subplots(4, 1, figsize=(10, 10))
        plt.tight_layout()
        ax[0].plot(self.sensor_bg_history.bg_values, label="iCGM")
        ax[0].plot(self.true_bg_history, label="true")
        ax[0].set_xlabel("Time (5 min)")
        ax[0].legend()
        ax[1].plot(sensor_candidate_range, icgm_transition_probabilities)
        ax[1].set_ylabel("iCGM Probabilities")
        ax[2].plot(sensor_candidate_range, behavior_probabilities)
        ax[2].set_ylabel("Behavior Probabilities")
        ax[3].plot(sensor_candidate_range, transition_probabilities)
        ax[3].set_ylabel("iCGM X Behavior Probabilities")
        ax[3].set_xlabel("CGM Values")
        plt.show()

    def set_random_values(self):
        # TODO
        return


class SensoriCGMModelOverlayBase():

    def get_conditional_probabilities(self, sensor_candidate_range, true_bg_history):
        raise NotImplementedError


class SensoriCGMModelOverlayV1(SensoriCGMModelOverlayBase):

    def __init__(self, bias=0, sigma=1, delay=0, spurious_value_prob=0.0, max_consecutive_spurious=1, random_state=None):

        self.bias = bias
        self.sigma = sigma
        self.delay = delay
        self.spurious_value_prob = spurious_value_prob
        self.max_consecutive_spurious = max_consecutive_spurious

        self.spurious_time_ctr = 0

        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.RandomState(0)

    def get_conditional_probabilities(self, sensor_candidate_range, true_bg_history):

        try:
            mu = true_bg_history[-self.delay]
        except IndexError:
            mu = true_bg

        behavior_probabilities = norm.pdf(sensor_candidate_range, mu + self.bias, self.sigma)
        if self.is_sensor_spurious():
            behavior_probabilities = 1.0 - behavior_probabilities

        return behavior_probabilities

    def is_sensor_spurious(self):

        is_spurious = False

        u = self.random_state.uniform()
        spurious_ctr = self.random_state.choice(range(self.max_consecutive_spurious))  # TODO: set random values function
        if self.spurious_time_ctr > 0:
            is_spurious = True
            self.spurious_time_ctr -= 1
        elif u < self.spurious_value_prob:
            is_spurious = True
            self.spurious_time_ctr = spurious_ctr

        return is_spurious


class SensoriCGMModelUniform(SensoriCGMModelOverlayBase):

    def get_conditional_probabilities(self, sensor_candidate_range, true_bg_history):
        behavior_probabilities = np.ones(shape=len(sensor_candidate_range))
        return behavior_probabilities


def compute_standard_normal_95_LB_CI_moments(LB_95_CI, n=30):
    """
    For a given 95% CI and n, show the relationship between mu and sigma.

    Parameters
    ----------
    LB_95_CI
    n
    """

    z = 1.96  # CI z score

    mu_sweep = np.arange(LB_95_CI, 1.0, 0.001)
    computed_stds = []
    for mu in mu_sweep:
        std = np.sqrt(n) * (mu - LB_95_CI) / z
        computed_stds.append(std)
        print(mu, std, n)

    for mu, sigma in zip(mu_sweep, computed_stds):
        x = np.arange(0.5, 1.0, 0.01)
        pdf = norm.pdf(x, loc=mu, scale=sigma)
        plt.plot(x, pdf, label="mu={} sigma={}".format(mu, sigma), alpha=0.4)
        plt.axvline(mu, alpha=0.4)
    plt.show()
    # plt.plot(mu_sweep, computed_stds)
    # plt.title("Moments of Distribution for {} Lower Bound CI with {} sensors".format(LB_95_CI, n))
    # plt.xlabel("Mu")
    # plt.ylabel("Sigma")
    # plt.show()

    return computed_stds


def compute_bionomial_95_LB_CI_moments(LB_95_CI=0.8):
    """
    # LB_95 = mu - ((1.644854 / N) * np.sqrt(Ns * Nf / N))
    """

    z = 1.644854  # 90% z-score, which

    def solve_mu(n, z):
        val = n / np.power(z, 2)
        closest = np.inf

        solution = None
        for p_hat in np.arange(0, 1.0, 0.0001):
            comparator = p_hat * (1 - p_hat) / np.power(p_hat - LB_95_CI, 2)
            distance = np.abs(comparator - val)
            if distance < closest and p_hat > LB_95_CI:
                closest = distance
                solution = p_hat

        return solution

    for N in [100, 1000, 2880, 10000, 2880*100]:

        mu = solve_mu(N, z=z)
        print(mu, N)

    # plt.plot(mu_sweep, computed_stds)
    # plt.title("Moments of Distribution for {} Lower Bound CI with {} data points".format(LB_95_CI, N))
    # plt.xlabel("Mu")
    # plt.ylabel("Sigma")
    # plt.show()


def sample_sensor_batch_bounds(n=30, mu=0.83, sigma=0.08383508533242345, random_state=None):

    if random_state is None:
        random_state = np.random.RandomState(0)

    sensor_batch_bounds = random_state.normal(mu, sigma, size=n)
    print("Lowest bound in sample:", np.min(sensor_batch_bounds))

    return sensor_batch_bounds


def get_sine_data(num_hours=24*10):

    num_data_points = num_hours * 12
    return np.sin(np.arange(num_data_points) * 2 * np.pi * 3 / num_data_points) * 180 + 220


def get_test_data():

    not_fitting_list_mean_shift_filelist = sorted(list(set(list(pd.read_csv(
        "/Users/csummers/Downloads/mean_shift_special_controls_not_passed_icgm-sensitivity-analysis-scenarios-2020-07-10-nogit.csv")[
                                                             "training_scenario_filename"]))))

    scenarios_dir = "/Users/csummers/dev/data-science-simulator/data/raw/icgm-sensitivity-analysis-scenarios-2020-07-10"
    df = pd.read_csv(os.path.join(scenarios_dir, not_fitting_list_mean_shift_filelist[0]))
    true_bg_trace = df.iloc[50, 2:].astype(float).values
    return true_bg_trace


if __name__ == "__main__":

    # TODO: Implement criteria G, J, K with error window updates
    # TODO: add icgm behavior, see other todo above in function
    # TODO: How best to initialize - have icgm values in every bin so will always have enough data?
    # TODO: How best to analyze risk -
    #   - sweep icgm controls params: criteria LB, batch mu/sigma, n sensors, history window, initialization
    #   - sweep physical params: bias, noise, delay, spurious values

    # Examples
    # iCGM + Uniform Behavior
    # iCGM + simple model behavior
    # iCGM worst case + Uniform behavior

    # Look at how to initial a batch of sensors
    # compute_standard_normal_95_LB_CI_moments(LB_95_CI=0.8, n=30)
    # compute_bionomial_95_LB_CI_moments(LB_95_CI=0.8)
    # true_bg_trace = get_test_data()[2500:3000]
    # sensor_batch_bounds = sample_sensor_batch_bounds(n=300, mu=0.83, sigma=0.09989747859174734)
    # sensor_batch_bounds_binomial = sample_sensor_batch_bounds(n=300, mu=0.802, sigma=0.005589005688828231)

    true_bg_trace = get_sine_data(num_hours=24)

    sensor_run_durations_minutes = []

    bias_sweep = [
        0,
        # 20,
        # 40,
        # 60,
        # 80,
        # 100
    ]
    # bias_sweep = range(-25, 25, 5)
    noise_sweep = [
        1,
        # 25,
        # 10,
        # 50
    ]

    special_controls = [
        iCGM_THRESHOLDS,
        G6_THRESHOLDS,
    ]

    for delay in [2]:
        for sp_ctrls in special_controls:
            for i, bias in enumerate(bias_sweep):
                for j, sigma in enumerate(noise_sweep):
                    sensor_id = "bias: {}. noise: {}".format(bias, sigma)
                    sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace(datetimes=[datetime.datetime.now()], values=[100]))
                    sensor_config.history_window_hrs = 24 * 1

                    sensor_config.behavior_model = SensoriCGMModelOverlayV1(
                        bias=bias,
                        sigma=sigma,
                        delay=delay,
                        spurious_value_prob=0.01,
                        max_consecutive_spurious=1)
                    # sensor_config.behavior_model = SensoriCGMModelUniform()

                    sensor_config.sensor_range = range(40, 400)
                    sensor_config.special_controls = sp_ctrls

                    sensor = SensoriCGM(datetime.datetime.now(), sensor_config=sensor_config)

                    start_time = time.time()

                    time_delta = datetime.timedelta(minutes=5)
                    prev_datetime = datetime.datetime.now()
                    sensor_bg_trace = []
                    for true_bg in true_bg_trace:
                        next_time = prev_datetime + time_delta
                        true_bg = round(true_bg)
                        sensor.update(next_time, patient_true_bg=true_bg, do_plot=False)
                        sensor_bg = sensor.get_bg(true_bg)
                        sensor_bg_trace.append(sensor_bg)
                        prev_datetime = next_time

                    sensor_run_durations_minutes.append((time.time() - start_time) / 60.0)

                    plt.plot(sensor_bg_trace, label=sensor_id, linestyle="-.", marker="*")
                    plt.plot(true_bg_trace, label="true")
                    plt.legend()

                    sensor.state.display()

    print("Sensor Avg Run Minutes for {} sensors: {}".format(len(sensor_run_durations_minutes), np.average(sensor_run_durations_minutes)))
    plt.show()




