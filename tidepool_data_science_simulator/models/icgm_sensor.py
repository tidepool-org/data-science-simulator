__author__ = "Cameron Summers"

import pdb
import os
import datetime
import time

from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.sensor import SensorBase
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace


iCGM_THRESHOLDS = {
    "A": 0.8,
    "B": 0.7,
    "C": 0.8,
    "D": 0.98,
    "E": 0.99,
    "F": 0.99
}

G6_THRESHOLDS = {
    "A": 85.40255821075945,
    "B": 98.02131957706338,
    "C": 74.93381359722157,
    "D": 99.23213676012934,
    "E": 84.9952769987456,
    "F": 99.84860393926321
}


class SensorNotiCGM(Exception):
    pass


class SensoriCGM(SensorBase):
    """
    iCGM Sensor Model
    """
    def __init__(self, time, sensor_config, icgm_confidence_lower_bounds=None, random_state=None):

        super().__init__(time, sensor_config)

        self.total_sensor_values = 300
        self.current_true_bg = None

        self.true_bg_history = []
        self.tmp_sensor_bg_history = []

        self.fifo_pop_keys = []

        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.RandomState(0)

        # Populate historical error windows
        # <70
        for j in range(85):
            self.fifo_pop_keys.append(("<70", "<15"))

        for j in range(13):
            self.fifo_pop_keys.append(("<70", "15-40"))

        for j in range(2):
            self.fifo_pop_keys.append(("<70", ">40"))

        # 70-180
        for j in range(70):
            self.fifo_pop_keys.append(("70-180", "<15%"))

        for j in range(29):
            self.fifo_pop_keys.append(("70-180", "15%-40%"))

        for j in range(1):
            self.fifo_pop_keys.append(("70-180", ">40%"))

        # >180
        for j in range(80):
            self.fifo_pop_keys.append((">180", "<15%"))

        for j in range(19):
            self.fifo_pop_keys.append((">180", "15%-40%"))

        for j in range(1):
            self.fifo_pop_keys.append((">180", ">40%"))

        self.random_state.shuffle(self.fifo_pop_keys)

        self.criteria_history = {
            "<70": {
                "<15": 85,
                "15-40": 13,
                ">40": 2
            },
            "70-180": {
                "<15%": 70,
                "15%-40%": 29,
                ">40%": 1
            },
            ">180": {
                "<15%": 80,
                "15%-40%": 19,
                ">40%": 1
            },
            # "overall": {
            #     "<20%": 87 * 3,
            #     ">20": 13 * 3
            # }
        }

        self.icgm_confidence_lower_bounds = icgm_confidence_lower_bounds
        if icgm_confidence_lower_bounds is None:
            self.icgm_confidence_lower_bounds = {
                "A": 0.8,
                "B": 0.7,
                "C": 0.8,
                "D": 0.98,
                "E": 0.99,
                "F": 0.99
            }

    def compute_canditate_sensor_bg_conditional_probability(self, true_bg, candidate_sensor_bg):
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
            denom = np.sum([v for k, v in self.criteria_history[range_key].items()]) + 1  # keep this way for 0 data case

            a_LB = self.icgm_confidence_lower_bounds["A"]
            d_LB = self.icgm_confidence_lower_bounds["D"]

            a_loss = self.criteria_history[range_key]["<15"] / denom - a_LB
            d_loss = (1 + self.criteria_history[range_key]["<15"] + self.criteria_history[range_key]["15-40"]) / denom - d_LB
            if error > 15:
                if self.criteria_history[range_key]["<15"] / denom < a_LB:
                    return 0.0, np.inf, "A"
                elif (1 + self.criteria_history[range_key]["<15"] + self.criteria_history[range_key]["15-40"]) / denom < d_LB:
                    return 0.0, np.inf, "D"

        # Criteria B & E
        if 70 <= candidate_sensor_bg <= 180:
            range_key = "70-180"
            denom = np.sum([v for k, v in self.criteria_history[range_key].items()]) + 1  # keep this way for 0 data case

            b_LB = self.icgm_confidence_lower_bounds["B"]
            e_LB = self.icgm_confidence_lower_bounds["E"]

            b_loss = (self.criteria_history[range_key]["<15%"]) / denom - b_LB
            e_loss = (1 + self.criteria_history[range_key]["<15%"] + self.criteria_history[range_key]["15%-40%"]) / denom - e_LB
            if error_percent > 0.15:
                if (self.criteria_history[range_key]["<15%"]) / denom < b_LB:
                    return 0.0, np.inf, "B"
                elif (1 + self.criteria_history[range_key]["<15%"] + self.criteria_history[range_key]["15%-40%"]) / denom < e_LB:
                    return 0.0, np.inf, "E"

        # Criteria C & F
        if candidate_sensor_bg > 180:
            range_key = ">180"
            denom = np.sum([v for k, v in self.criteria_history[range_key].items()]) + 1  # keep this way for 0 data case

            c_LB = self.icgm_confidence_lower_bounds["C"]
            f_LB = self.icgm_confidence_lower_bounds["F"]

            c_loss = (self.criteria_history[range_key]["<15%"]) / denom - c_LB
            f_loss = (1 + self.criteria_history[range_key]["<15%"] + self.criteria_history[range_key]["15%-40%"]) / denom - f_LB
            if error_percent > 0.15:
                if (self.criteria_history[range_key]["<15%"]) / denom < c_LB:
                    return 0.0, np.inf, "C"
                elif (1 + self.criteria_history[range_key]["<15%"] + self.criteria_history[range_key]["15%-40%"]) / denom < f_LB:
                    return 0.0, np.inf, "F"

        # Criteria G
        # overall_key = "overall"
        # overall_denom = sum([v for k, v in self.criteria_history[overall_key]])
        # if error_percent > 0.2 and (1 + self.criteria_history[overall_key][">20%"]) / overall_denom > 0.13:
        #     meets_special_controls = False

        # # Criteria J
        # if (true_bg - prev_true_bg) / 5.0 > -2.0 and (candidate_icgm_bg - prev_sensor_bg) > 1.0:
        #     return 0.0
        #
        # # Criteria K
        # if (true_bg - prev_true_bg) / 5.0 > 2.0 and (candidate_icgm_bg - prev_sensor_bg) > 1.0:
        #     return 0.0

        loss = max([a_loss, b_loss, c_loss, d_loss, e_loss, f_loss])
        return 1.0, loss, ""

    def get_bg(self, true_bg):

        return self.current_sensor_bg

    def update(self, time, **kwargs):

        self.time = time
        self.set_random_values()

        do_plot_step_probs = kwargs.get("do_plot", False)

        true_bg = kwargs["patient_true_bg"]
        if self.current_true_bg == None:
            self.current_true_bg = true_bg

        self.true_bg_history.append(true_bg)

        sensor_candidate_range = self.sensor_config.sensor_range
        window_fifo_keys = self.fifo_pop_keys[-1]
        self.criteria_history[window_fifo_keys[0]][window_fifo_keys[1]] -= 1
        icgm_probabilities = self.compute_conditional_icgm_probabilities(true_bg,
                                                                         sensor_candidate_range=sensor_candidate_range)

        if np.sum(icgm_probabilities) == 0:
            print("No iCGM values available")

        behavior_probabilities = sensor_config.behavior_model.get_conditional_probabilities(
            sensor_candidate_range,
            self.true_bg_history
        )

        transition_probabilities = np.multiply(icgm_probabilities, behavior_probabilities)
        transition_probabilities /= np.sum(transition_probabilities)

        if do_plot_step_probs or np.isnan(transition_probabilities).any():
            fig, ax = plt.subplots(4, 1)
            ax[0].plot(self.tmp_sensor_bg_history, label="icgm")
            ax[0].plot(self.true_bg_history, label="true")
            ax[0].legend()
            ax[1].plot(sensor_candidate_range, icgm_probabilities)
            ax[2].plot(sensor_candidate_range, behavior_probabilities)
            ax[3].plot(sensor_candidate_range, transition_probabilities)
            plt.show()

        sensor_bg = self.random_state.choice(sensor_candidate_range, p=transition_probabilities)

        self.current_sensor_bg = sensor_bg
        self.current_true_bg = true_bg

        # Store the value
        self.sensor_bg_history.append(self.time, self.current_sensor_bg)
        self.tmp_sensor_bg_history.append(sensor_bg)

        self.update_error_windows(true_bg, sensor_bg)

    def compute_conditional_icgm_probabilities(self, true_bg, sensor_candidate_range=range(40, 400)):

        icgm_probs = []

        for candidate_icgm_bg in sensor_candidate_range:
            meets_special_controls, max_loss, failed_reason = self.compute_canditate_sensor_bg_conditional_probability(true_bg, candidate_icgm_bg)
            icgm_prob = int(meets_special_controls)
            # if max_loss < np.inf:
            #     icgm_prob += (max_loss * 1e12)

            icgm_probs.append(icgm_prob)

            if candidate_icgm_bg == true_bg and failed_reason != "":
                print(candidate_icgm_bg, true_bg, failed_reason)
                raise Exception("Perfect value failed to special controls.")

        if np.sum(icgm_probs) == 0:
            raise SensorNotiCGM("No iCGM values available")

        icgm_probs = np.array(icgm_probs) / np.sum(icgm_probs)

        # TODO: add behavior here to icgm probabilities, ie worst case loss, max bias, etc.

        # Worst case
        # idx = np.where(icgm_probabilities != 0)[0][-1:]
        # idx = np.where(icgm_probabilities != 0)[0][:10]
        # behavior_probabilities = np.zeros(shape=len(sensor_candidate_range))
        # behavior_probabilities[idx] = 1

        return icgm_probs

    def update_error_windows(self, true_bg, sensor_bg):

        num_readings_in_window = self.sensor_config.history_window_hrs * 12

        if self.total_sensor_values <= num_readings_in_window:
            self.total_sensor_values += 1

        bg_range_key = self.get_bg_range_key(sensor_bg)
        bg_error_key = self.get_bg_relative_error_domain_key(true_bg, sensor_bg)
        self.criteria_history[bg_range_key][bg_error_key] += 1

        # TODO: update overall and rate windows

        self.fifo_pop_keys.insert(0, (bg_range_key, bg_error_key))
        self.fifo_pop_keys.pop()

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

    def get_bg_relative_error_domain_key(self, true_bg, sensor_bg):

        bg_range_key = self.get_bg_range_key(sensor_bg)

        if bg_range_key == "<70":
            bg_error = np.abs(true_bg - sensor_bg)
            if 0 <= bg_error < 15:
                bg_error_key = "<15"
            elif 15 <= bg_error <= 40:
                bg_error_key = "15-40"
            elif bg_error > 40:
                bg_error_key = ">40"
            else:
                raise Exception

        else:
            bg_error_percentage = np.abs(true_bg - sensor_bg) / true_bg

            if 0 <= bg_error_percentage <= 0.15:
                bg_error_key = "<15%"
            elif 0.15 < bg_error_percentage <= 0.40:
                bg_error_key = "15%-40%"
            elif bg_error_percentage > 0.40:
                bg_error_key = ">40%"
            else:
                raise Exception

        return bg_error_key

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
            behavior_probabilities = 1 - behavior_probabilities

        return behavior_probabilities

    def is_sensor_spurious(self):

        is_spurious = False

        u = self.random_state.uniform()
        spurious_ctr = self.random_state.choice(range(self.max_consecutive_spurious)) # TODO: set random values function
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
    true_bg_trace = get_test_data()[2500:3000]
    # sensor_batch_bounds = sample_sensor_batch_bounds(n=300, mu=0.83, sigma=0.09989747859174734)
    # sensor_batch_bounds_binomial = sample_sensor_batch_bounds(n=300, mu=0.802, sigma=0.005589005688828231)

    # true_bg_trace = get_sine_data(num_hours=24)

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
        # 5,
        # 10,
        # 50
    ]

    special_controls = [
        iCGM_THRESHOLDS,
        G6_THRESHOLDS,
        {
            "A": 0.8,
            "B": 0.7,
            "C": 0.8,
            "D": 0.99999,
            "E": 0.999999,
            "F": 0.999999
        }
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

                    sensor = SensoriCGM(datetime.datetime.now(),
                                        sensor_config=sensor_config,
                                        icgm_confidence_lower_bounds=sp_ctrls)

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

                    print(sensor.criteria_history)

                    plt.plot(sensor_bg_trace, label=sensor_id, linestyle="-.", marker="*")
                    plt.plot(true_bg_trace, label="true")
                    plt.legend()

    print("Sensor Avg Run Minutes for {} sensors: {}".format(len(sensor_run_durations_minutes), np.average(sensor_run_durations_minutes)))
    plt.show()




