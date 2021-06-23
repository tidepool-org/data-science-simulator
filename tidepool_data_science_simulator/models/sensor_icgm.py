__author__ = "Cameron Summers"

import pdb
import os
import datetime
import time
import json
import copy
import logging

logger = logging.getLogger(__name__)

from collections import Counter

import pandas as pd
import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.sensor import SensorBase, NoisySensor
from tidepool_data_science_simulator.makedata.scenario_parser import SensorConfig, GlucoseTrace

from tidepool_data_science_models.models.icgm_sensor_generator_functions import preprocess_data, calc_icgm_sc_table
from tidepool_data_science_simulator.evaluation.icgm_eval import iCGMEvaluator


DEXCOM_CONCURRENCY_PG23_ADULT = [
            [0.519, 0.050, 0.011, 0.001, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.407, 0.527, 0.117, 0.007, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.074, 0.410, 0.637, 0.110, 0.002, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.013, 0.234, 0.758, 0.197, 0.010, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.122, 0.669, 0.248, 0.014, 0.000, 0.001, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.001, 0.130, 0.599, 0.253, 0.017, 0.004, 0.002, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.002, 0.141, 0.619, 0.306, 0.051, 0.002, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.113, 0.562, 0.359, 0.096, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.113, 0.480, 0.380, 0.261],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.105, 0.460, 0.478],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.059, 0.261]
        ]

DEXCOM_CONCURRENCY_PG24_PED = [
            [0.500, 0.063, 0.018, 0.012, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.500, 0.480, 0.125, 0.018, 0.006, 0.001, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.441, 0.571, 0.079, 0.008, 0.002, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.016, 0.286, 0.780, 0.124, 0.008, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.110, 0.673, 0.145, 0.010, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.002, 0.187, 0.676, 0.241, 0.030, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.001, 0.166, 0.578, 0.228, 0.035, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.168, 0.568, 0.227, 0.027, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.165, 0.574, 0.378, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.009, 0.160, 0.446, 0.200],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004, 0.149, 0.800]
        ]


TP_iCGM_CONCURRENCY = [
            [0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.500, 0.543, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.441, 0.717, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.016, 0.286, 0.889, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.110, 0.812, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.002, 0.187, 0.832, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.001, 0.166, 0.829, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.168, 0.826, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.165, 0.836, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.009, 0.160, 0.851, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004, 0.149, 1.000]
        ]

TP_iCGM_CONCURRENCY_TEST = [
            [0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.500, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.984, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.016, 1.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.998, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.002, 0.999, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.001, 0.999, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.997, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.003, 0.991, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.009, 0.996, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.004, 1.000, 1.000]
        ]


DEXCOM_CONCURRENCY_PG23_ADULT_WORST = [
            [0.519, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.407, 0.577, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.074, 0.410, 0.766, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.013, 0.234, 0.877, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.122, 0.868, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.001, 0.130, 0.858, 0.000, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.002, 0.141, 0.886, 0.000, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.113, 0.886, 0.000, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.113, 0.894, 0.000, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.105, 0.941, 0.000],
            [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000, 0.001, 0.059, 1.000]
        ]


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
    "A": 0.8540255821075945,
    "B": 0.7493381359722157,
    "C": 0.849952769987456,
    "D": 0.9802131957706338,
    "E": 0.9923213676012934,
    "F": 0.9984860393926321,
    "G": 0.906,
    "J": 0.01,
    "K": 0.01,
}

CLEAN_INITIAL_CONTROLS = {
    "A": 0.98,
    "B": 0.98,
    "C": 0.98,
    "D": 1.0,
    "E": 1.0,
    "F": 1.0,
    "G": 1.0,
    "J": 0.0,
    "K": 0.0,
}


class LegitimateNoniCGMSensor(Exception):
    pass


class iCGMState():
    """
    State object for iCGM sensor to manage state space in stochastic process.
    """
    def __init__(self, time, num_initial_values, num_history_values, special_controls, initial_controls, do_look_ahead, look_ahead_min_prob, random_state):

        self.time = time

        self.prev_true_bg = None
        self.prev_sensor_bg = None

        self.num_current_values = num_initial_values
        self.num_history_values = num_history_values

        self.special_controls = special_controls
        self.initial_controls = initial_controls

        self.do_look_ahead = do_look_ahead
        self.look_ahead_min_prob = look_ahead_min_prob

        self.random_state = random_state

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
                    "<15": self.initial_controls["A"],
                    "15-40": self.initial_controls["D"] - self.initial_controls["A"],
                    ">40": 1.0 - self.initial_controls["D"]
                },
                "70-180": {
                    "<15%": self.initial_controls["B"],
                    "15%-40%": self.initial_controls["E"] - self.initial_controls["B"],
                    ">40%": 1.0 - self.initial_controls["E"]
                },
                ">180": {
                    "<15%": self.initial_controls["C"],
                    "15%-40%": self.initial_controls["F"] - self.initial_controls["C"],
                    ">40%": 1.0 - self.initial_controls["F"]
                },
            },
            "overall": {
                "<20%": self.initial_controls["G"],
                ">20%": 1.0 - self.initial_controls["G"]
            },
            "true_neg_rate": {
                "ok": 1.0 - self.initial_controls["J"],
                "extreme": self.initial_controls["J"],
            },
            "true_pos_rate": {
                "ok": 1.0 - self.initial_controls["K"],
                "extreme": self.initial_controls["K"],
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
        icgm_losses = []
        failure_reasons = {}

        # self.prime_next_state()  # FIXME
        for candidate_icgm_bg in sensor_candidate_range:
            meets_special_controls, loss, fail_reason = self.compute_candidate_sensor_bg_conditional_probability(
                true_bg, candidate_icgm_bg, self.prev_true_bg, self.prev_sensor_bg)
            icgm_prob = int(meets_special_controls)
            icgm_losses.append(loss)

            icgm_probs.append(icgm_prob)
            failure_reasons[candidate_icgm_bg] = fail_reason

            if self.do_look_ahead and np.abs(candidate_icgm_bg - true_bg) <= 1.0 and fail_reason != "":
                # print(candidate_icgm_bg, true_bg, fail_reason)
                pass

        if self.do_look_ahead:
            # ==================================================================================
            # Look ahead at next true bgs that would potentially violate rate criteria J & K
            # on *next* iteration. Remove from this prob distribution values that if chosen would
            # disallow all icgm values next.
            #
            # TODO: This should be parameterized and all hard coded values removed.
            # ==================================================================================
            before_probs_sum = np.sum(icgm_probs)
            # print("Before", before_probs_sum)

            current_prob_idxs = np.where(np.array(icgm_probs) > 0)[0]
            sensor_candidate_list = list(sensor_candidate_range)

            next_tbg_pos_rate = true_bg + 10.000001  # makes a true rate > 2.0 mg/dL / minute
            next_tbg_neg_rate = true_bg - 10.000001  # makes a true rate < -2.0 mg/dL / minute

            best_icgm_val_high = max(40, min(400, int(round(next_tbg_pos_rate))))
            best_icgm_val_low = max(40, min(400, int(round(next_tbg_neg_rate))))

            possible_next_tbgs_neg_rate = list(range(40, best_icgm_val_low))
            possible_next_tbgs_pos_rate = list(range(best_icgm_val_high, 401))

            for icgm_prob_idx in current_prob_idxs:
                future_probs = []
                for next_tbg in possible_next_tbgs_pos_rate:

                    if (next_tbg - true_bg) / 5.0 > 7.0:  # Don't consider impossible true rates
                        continue

                    icgm_value = sensor_candidate_list[icgm_prob_idx]
                    if (next_tbg - icgm_value) < -5.0:
                        future_probs.append(0)
                    else:
                        future_probs.append(1)

                for next_tbg in possible_next_tbgs_neg_rate:

                    if (next_tbg - true_bg) / 5.0 < -7.0:  # Don't consider impossible true rates
                        continue

                    icgm_value = sensor_candidate_list[icgm_prob_idx]
                    if (next_tbg - icgm_value) > 5.0:
                        future_probs.append(0)
                    else:
                        future_probs.append(1)

                success_probability = np.sum(future_probs) / len(future_probs)
                if success_probability < self.look_ahead_min_prob:
                    # success_probability = 0.0

                    icgm_probs[icgm_prob_idx] = 0.0
                # icgm_probs[icgm_prob_idx] *= success_probability

            after_probs_sum = np.sum(icgm_probs)
            # print("After", after_probs_sum, "\n")
            # ======================

        if np.sum(icgm_probs) == 0 and (true_bg < min(sensor_candidate_range) or true_bg > max(sensor_candidate_range)):
            raise LegitimateNoniCGMSensor("True BG out of sensor range and no available iCGM values.")

        # if np.sum(icgm_probs) == 0 and after_probs_sum != 0:
        #     print(true_bg, self.state_history)
        #     raise Exception("Error: No iCGM values available")

        icgm_probs = np.array(icgm_probs) / np.sum(icgm_probs)

        return icgm_probs, icgm_losses

    def compute_candidate_sensor_bg_conditional_probability(self, true_bg, candidate_sensor_bg, prev_true_bg, prev_sensor_bg):
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
        # TODO: use builtin functions
        error = np.abs(true_bg - candidate_sensor_bg)
        error_percent = error / true_bg

        loss = 0.0

        # Criteria H
        if true_bg < 70 and candidate_sensor_bg > 180:
            return 0.0, np.inf, "H"

        # Criteria I
        if true_bg > 180 and candidate_sensor_bg < 70:
            return 0.0, np.inf, "I"

        # Criteria A & D
        if candidate_sensor_bg < 70:
            range_key = "<70"
            denom_lt70 = np.sum([v for k, v in self.state_history["range"][range_key].items()]) + 1  # keep this way for 0 data case

            a_LB = self.special_controls["A"]
            d_LB = self.special_controls["D"]

            a_percent = self.state_history["range"][range_key]["<15"] / denom_lt70
            ad_compl_percent = (1 + self.state_history["range"][range_key][">40"]) / denom_lt70

            # loss += np.abs(a_percent - a_LB)
            loss += np.abs(ad_compl_percent - round((1.0 - d_LB), 6))

            if error < 15:
                pass # always allowed
            elif 15 <= error < 40 and a_percent <= a_LB:
                return 0.0, np.inf, "A"
            elif error >= 40:
                if a_percent <= a_LB or ad_compl_percent >= round((1.0 - d_LB), 6):
                    return 0.0, np.inf, "D"

        # Criteria B & E
        if 70 <= candidate_sensor_bg <= 180:
            range_key = "70-180"
            denom_70_180 = np.sum([v for k, v in self.state_history["range"][range_key].items()]) + 1  # keep this way for 0 data case

            b_LB = self.special_controls["B"]
            e_LB = self.special_controls["E"]

            b_percent = (self.state_history["range"][range_key]["<15%"]) / denom_70_180
            be_compl_percent = (1 + self.state_history["range"][range_key][">40%"]) / denom_70_180

            # loss += np.abs(b_percent - b_LB)
            loss += np.abs(be_compl_percent - round((1.0 - e_LB), 6))

            if error_percent < 0.15:
                pass  # always allowed
            elif 0.15 <= error_percent < 0.4 and b_percent <= b_LB:
                return 0.0, np.inf, "B"  # takes away from B
            elif error_percent >= 0.4:
                if b_percent <= b_LB or be_compl_percent >= round(1.0 - e_LB, 6):
                    return 0.0, np.inf, "E"

        # Criteria C & F
        if candidate_sensor_bg > 180:
            range_key = ">180"
            denom_gt180 = np.sum([v for k, v in self.state_history["range"][range_key].items()]) + 1  # keep this way for 0 data case

            c_LB = self.special_controls["C"]
            f_LB = self.special_controls["F"]

            c_percent = (self.state_history["range"][range_key]["<15%"]) / denom_gt180
            cf_complement = (1 + self.state_history["range"][range_key][">40%"]) / denom_gt180

            # loss += np.abs(c_percent - c_LB)
            loss += np.abs(cf_complement - round((1.0 - f_LB), 6))

            if error_percent < 0.15:
                pass  # always allowed
            elif 0.15 <= error_percent < 0.4 and c_percent <= c_LB:
                return 0.0, np.inf, "C"  # takes away from C
            elif error_percent >= 0.4:
                if c_percent <= c_LB or cf_complement >= round((1.0 - f_LB), 6):
                    return 0.0, np.inf, "F"

        # Criteria G
        g_LB = self.special_controls["G"]
        overall_denom = np.sum([v for k, v in self.state_history["overall"].items()]) + 1
        g_percent = (1 + self.state_history["overall"][">20%"]) / overall_denom

        # loss += np.abs(g_percent - round(1.0 - g_LB, 6))

        if error_percent >= 0.2 and g_percent > round(1.0 - g_LB, 6):
            return 0.0, np.inf, "G"

        true_rate, cgm_rate = self.get_bg_rates(true_bg, candidate_sensor_bg, prev_true_bg, prev_sensor_bg)

        # Criteria J
        j_bound = self.special_controls["J"]
        neg_rate_denom = np.sum([v for k, v in self.state_history["true_neg_rate"].items()]) + 1
        j_percent = (1 + self.state_history["true_neg_rate"]["extreme"]) / neg_rate_denom

        loss += np.abs(j_percent - j_bound)

        if true_rate < -2.0 and cgm_rate > 1.0:
            if j_percent > j_bound:
                return 0.0, np.inf, "J"

        # Criteria K
        k_bound = self.special_controls["K"]
        pos_rate_denom = np.sum([v for k, v in self.state_history["true_pos_rate"].items()]) + 1
        k_percent = (1 + self.state_history["true_pos_rate"]["extreme"]) / pos_rate_denom

        loss += np.abs(k_percent - k_bound)

        if true_rate > 2.0 and cgm_rate < -1.0:
            if k_percent > k_bound:
                return 0.0, np.inf, "K"

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

        # # Activate windowed memory once there are enough values
        # if self.num_current_values >= self.num_history_values:
        #     self.state_history_fifo_keys["range"].pop()
        #     self.state_history_fifo_keys["overall"].pop()
        #     self.state_history_fifo_keys["true_pos_rate"].pop()
        #     self.state_history_fifo_keys["true_neg_rate"].pop()
        #
        # else:
        #     self.num_current_values += 1

        self.prev_true_bg = true_bg
        self.prev_sensor_bg = sensor_bg

        # self.minutes_since_last_update = (time - self.time).total_seconds() / 60.0
        # assert 4.99 < self.minutes_since_last_update < 5.01

        self.time = time

        # self.validate_current_state()  # TODO: TMP

    def validate_current_state(self):

        # FIXME: use passed in special controls

        range_key = "<70"
        total_values = np.sum([v for v in self.state_history["range"][range_key].values()])

        num_good = self.state_history["range"][range_key]["<15"]
        num_ok = num_good + self.state_history["range"][range_key]["15-40"]
        num_bad = self.state_history["range"][range_key][">40"]
        if num_good / total_values < 0.8:
            raise Exception("Invalid state")
            a = 1
        elif num_ok / total_values < 0.98:
            raise Exception("Invalid state")
            a = 1
        elif num_bad / total_values > 0.02:
            raise Exception("Invalid state")
            a = 1

        range_key = "70-180"
        total_values = np.sum([v for v in self.state_history["range"][range_key].values()])
        num_good = self.state_history["range"][range_key]["<15%"]
        num_ok = num_good + self.state_history["range"][range_key]["15%-40%"]
        num_bad = self.state_history["range"][range_key][">40%"]
        if num_good / total_values < 0.7:
            raise Exception("Invalid state {}".format(range_key))
            a = 1
        elif num_ok / total_values < 0.99:
            raise Exception("Invalid state {}".format(range_key))
            a = 1
        elif num_bad / total_values > 0.01:
            raise Exception("Invalid state {}".format(range_key))
            a = 1

        range_key = ">180"
        total_values = np.sum([v for v in self.state_history["range"][range_key].values()])
        num_good = self.state_history["range"][range_key]["<15%"]
        num_ok = num_good + self.state_history["range"][range_key]["15%-40%"]
        num_bad = self.state_history["range"][range_key][">40%"]
        if num_good / total_values < 0.8:
            raise Exception("Invalid state {}".format(range_key))
            a = 1
        elif num_ok / total_values < 0.99:
            raise Exception("Invalid state {}".format(range_key))
            a = 1
        elif num_bad / total_values > 0.01:
            raise Exception("Invalid state {}".format(range_key))
            a = 1

    def display(self):

        # TODO: make this useful
        print(json.dumps(self.state_history, indent=4, sort_keys=True))

    def get_bg_error_pecentage(self, true_bg, sensor_bg):

        bg_error_percentage = np.abs(np.abs(true_bg - sensor_bg) / true_bg)

        if bg_error_percentage < 0:
            print(true_bg, sensor_bg)
            raise ValueError("Invalid percentage")

        return bg_error_percentage

    def get_bg_abs_error(self, true_bg, sensor_bg):

        bg_error = np.abs(true_bg - sensor_bg)
        return bg_error

    def get_bg_range_key(self, bg):

        if -np.inf < bg < 70:
            key = "<70"
        elif 70 <= bg <= 180:
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
            elif 15 <= bg_error < 40:
                bg_error_key = "15-40"
            elif bg_error >= 40:
                bg_error_key = ">40"
            else:
                raise Exception

        else:
            bg_error_percentage = self.get_bg_error_pecentage(true_bg, sensor_bg)

            if 0 <= bg_error_percentage < 0.15:
                bg_error_key = "<15%"
            elif 0.15 <= bg_error_percentage < 0.40:
                bg_error_key = "15%-40%"
            elif bg_error_percentage >= 0.40:
                bg_error_key = ">40%"
            else:
                print(bg_error_percentage)
                raise Exception

        return bg_error_key

    def get_overall_error_key(self, true_bg, sensor_bg):

        if self.get_bg_error_pecentage(true_bg, sensor_bg) > 0.2:
            return ">20%"
        else:
            return "<20%"

    def get_bg_rates(self, true_bg, sensor_bg, prev_true_bg=None, prev_sensor_bg=None):
        true_rate = 0.0
        cgm_rate = 0.0
        if prev_true_bg is not None and prev_sensor_bg is not None:
            true_rate = (true_bg - prev_true_bg) / 5.0  # self.minutes_since_last_update
            cgm_rate = (sensor_bg - prev_sensor_bg) / 5.0  # self.minutes_since_last_update

        return true_rate, cgm_rate

    def get_true_pos_rate_error_key(self, true_bg, sensor_bg):

        true_rate, cgm_rate = self.get_bg_rates(true_bg, sensor_bg, prev_true_bg=self.prev_true_bg, prev_sensor_bg=self.prev_sensor_bg)

        if true_rate > 2.0 and cgm_rate < -1.0:
            key = "extreme"
        else:
            key = "ok"

        return key

    def get_true_neg_rate_error_key(self, true_bg, sensor_bg):

        true_rate, cgm_rate = self.get_bg_rates(true_bg, sensor_bg, prev_true_bg=self.prev_true_bg, prev_sensor_bg=self.prev_sensor_bg)

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

        self.name = "iCGM"

        self.true_bg_history = []

        num_history_values = sensor_config.history_window_hrs * 12
        self.state = iCGMState(
            time,
            num_initial_values=300,  # zero means no history is populated
            num_history_values=num_history_values,
            special_controls=sensor_config.special_controls,
            initial_controls=sensor_config.initial_controls,
            do_look_ahead=sensor_config.do_look_ahead,
            look_ahead_min_prob=sensor_config.look_ahead_min_prob,
            random_state=random_state
        )

        self.random_state = random_state

    def get_bg(self, true_bg):

        return self.current_sensor_bg

    def get_info_stateless(self):

        stateless_info = dict()
        stateless_info.update(
            {
                "special_controls": self.sensor_config.special_controls,
                "initial_controls": self.sensor_config.initial_controls,
                "behavior_models": [m.get_info_stateless() for m in self.sensor_config.behavior_models],
                "history_window_hrs": self.sensor_config.history_window_hrs,
                "true_start_bg": self.true_start_bg,
                "start_bg_with_offset": self.start_bg_with_offset
            }
        )

        return stateless_info

    def update(self, time, **kwargs):

        self.time = time
        self.set_random_values()

        true_bg = kwargs["patient_true_bg"]
        self.true_bg_history.append(true_bg)

        sensor_candidate_range = self.sensor_config.sensor_range

        try:
            probability_chain_for_plotting = []
            icgm_transition_probabilities, icgm_losses = self.state.get_state_transition_probabilities(true_bg, sensor_candidate_range)

            probability_chain_for_plotting.append(copy.copy(icgm_transition_probabilities))

            for model in self.sensor_config.behavior_models:
                behavior_probabilities = model.get_conditional_probabilities(
                    sensor_candidate_range,
                    true_bg_history=self.true_bg_history,
                    sensor_bg_history=self.sensor_bg_history.bg_values,
                    icgm_transition_probabilities=icgm_transition_probabilities,
                    icgm_losses=icgm_losses
                )

                icgm_transition_probabilities = np.multiply(icgm_transition_probabilities, behavior_probabilities)
                probability_chain_for_plotting.append(copy.copy(icgm_transition_probabilities))

            if np.sum(icgm_transition_probabilities) == 0:
                raise Exception("Transition probabilities are zero.")

            icgm_transition_probabilities /= np.sum(icgm_transition_probabilities)

            do_plot_step_probs = kwargs.get("do_plot", False)
            if do_plot_step_probs or np.isnan(icgm_transition_probabilities).any():
                self.plot_internals(sensor_candidate_range,
                                    probability_chain_for_plotting,
                                    icgm_losses)

            sensor_bg = self.random_state.choice(sensor_candidate_range, p=icgm_transition_probabilities)
            self.state.update(time, true_bg, sensor_bg)

        except LegitimateNoniCGMSensor:
            sensor_bg = np.nan

        self.current_sensor_bg = sensor_bg
        self.sensor_bg_history.append(self.time, self.current_sensor_bg)

    def plot_internals(self, sensor_candidate_range,
                       probability_chain_for_plotting,
                       icgm_losses):

        fig, ax = plt.subplots(len(probability_chain_for_plotting) + 2, 1, figsize=(10, 10))
        plt.tight_layout()
        ax[0].plot(self.sensor_bg_history.bg_values, label="iCGM")
        ax[0].plot(self.true_bg_history, label="true")
        ax[0].set_xlabel("Time (5 min)")
        ax[0].legend()
        for i, probs in enumerate(probability_chain_for_plotting):
            ax[i+1].plot(sensor_candidate_range, probs)
            ax[i+1].set_ylabel("iCGM Probabilities")
        ax[-1].plot(sensor_candidate_range, icgm_losses)
        ax[-1].set_ylabel("iCGM Losses")
        plt.show()

    def set_random_values(self):
        # TODO
        return


class NoisySensorInitialOffset(NoisySensor):
    """
    Noisy sensor that with ability to manually inject value at sim start time.
    """
    def __init__(self, time, sensor_config, t0_error_bg, random_state=None, sim_start_time=None):
        super().__init__(time, sensor_config, random_state)
        self.sim_start_time = sim_start_time
        self.t0_error_bg = t0_error_bg

    def update(self, time, **kwargs):

        if self.sim_start_time is not None and time == self.sim_start_time:

            self.time = time
            true_bg = kwargs["patient_true_bg"]

            sensor_bg = self.t0_error_bg
            self.true_start_bg = true_bg
            self.start_bg_with_offset = sensor_bg

            self.current_sensor_bg = sensor_bg
            self.sensor_bg_history.append(self.time, self.current_sensor_bg)
        else:
            super().update(time, **kwargs)


class SensoriCGMInitialOffset(SensoriCGM):
    """
    Inherit all behavior of iCGM sensor, except manually set the initial value of the sensor at
    simulation start time.
    """
    def __init__(self, time, sensor_config, t0_error_bg, random_state=None, sim_start_time=None):
        super().__init__(time, sensor_config, random_state)
        self.sim_start_time = sim_start_time
        self.t0_error_bg = t0_error_bg

    def update(self, time, **kwargs):

        if self.sim_start_time is not None and time == self.sim_start_time:

            self.time = time
            true_bg = kwargs["patient_true_bg"]
            self.true_bg_history.append(true_bg)

            sensor_bg = self.t0_error_bg
            self.true_start_bg = true_bg
            self.start_bg_with_offset = sensor_bg

            self.state.update(time, true_bg, sensor_bg)
            self.current_sensor_bg = sensor_bg
            self.sensor_bg_history.append(self.time, self.current_sensor_bg)
        else:
            super().update(time, **kwargs)


class SensoriCGMModelOverlayBase():

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):
        raise NotImplementedError

    def get_info_stateless(self):
        raise NotImplementedError


class SensoriCGMModelOverlayNoiseBiasWorst(SensoriCGMModelOverlayBase):

    def __init__(self, max_bias_percentage):

        self.max_bias_percentage = max_bias_percentage

        noise_percentage_amplitude = min(max_bias_percentage, 100 - max_bias_percentage)
        self.min_percentile = max_bias_percentage - noise_percentage_amplitude
        self.max_percentile = max_bias_percentage + noise_percentage_amplitude

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):

        icgm_probabilities = kwargs["icgm_transition_probabilities"]

        icgm_value_idxs = np.where(np.array(icgm_probabilities) > 0)[0]
        num_probs = len(icgm_value_idxs)

        low_icgm_idx = int(round(np.percentile(range(num_probs), self.min_percentile)))
        high_icgm_idx = int(round(np.percentile(range(num_probs), self.max_percentile))) + 1

        model_probs = np.zeros(shape=(len(icgm_probabilities)))

        accept_icgm_mask = icgm_value_idxs[low_icgm_idx:high_icgm_idx]
        model_probs[accept_icgm_mask] = 1.0

        if np.sum(model_probs) == 0:
            print(num_probs, low_icgm_idx, high_icgm_idx)

        return model_probs

    def get_info_stateless(self):

        return {
            "type": "Bias-Noise-Worst-Case",
            "max_bias_percentage": self.max_bias_percentage
        }


class SensoriCGMModelOverlayV1(SensoriCGMModelOverlayBase):

    def __init__(self, bias=0, sigma=1, delay=0, spurious_value_prob=0.0, num_consecutive_spurious=1, random_state=None):

        self.bias = bias
        self.sigma = sigma
        self.delay = delay
        self.spurious_value_prob = spurious_value_prob
        self.num_consecutive_spurious = num_consecutive_spurious

        self.spurious_time_ctr = 0

        self.random_state = random_state
        if random_state is None:
            self.random_state = np.random.RandomState(0)

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):

        true_bg_history = kwargs["true_bg_history"]

        try:
            delay_idx = -1
            if self.delay > 0:
                delay_idx -= self.delay
            mu = true_bg_history[delay_idx]
        except IndexError:
            mu = true_bg

        behavior_probabilities = norm.pdf(sensor_candidate_range, mu + self.bias, self.sigma)
        if self.is_sensor_spurious():
            behavior_probabilities = 1.0 - behavior_probabilities

        return behavior_probabilities

    def is_sensor_spurious(self):

        is_spurious = False

        u = self.random_state.uniform()
        # spurious_ctr = self.random_state.choice(range(self.max_consecutive_spurious))  # TODO: set random values function
        spurious_ctr = self.num_consecutive_spurious
        if self.spurious_time_ctr > 0:
            is_spurious = True
            self.spurious_time_ctr -= 1
        elif u < self.spurious_value_prob:
            is_spurious = True
            self.spurious_time_ctr = spurious_ctr

        return is_spurious

    def get_info_stateless(self):
        return {
            "name": "SensoriCGMModelOverlayV1",
            "bias": self.bias,
            "delay": self.delay,
            "sigma": self.sigma,
            "spurious_value_prob": self.spurious_value_prob,
            "max_consecutive_spurious": self.num_consecutive_spurious
        }


class SensoriCGMModelUniform(SensoriCGMModelOverlayBase):

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):

        behavior_probabilities = np.ones(shape=len(sensor_candidate_range))
        return behavior_probabilities

    def get_info_stateless(self):
        return {
            "name": "SensoriCGMModelUniform"
        }


class SensoriCGMModelControlsBoundary(SensoriCGMModelOverlayBase):

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):

        icgm_losses = kwargs["icgm_losses"]
        icgm_transition_probabilities = kwargs["icgm_transition_probabilities"]

        icgm_losses = np.multiply(icgm_losses, icgm_transition_probabilities)

        available_loss_indices = np.where(icgm_losses > 0)[0]
        lowest_available_loss_idx = np.argmin(icgm_losses[available_loss_indices])

        lowest_icgm_loss_idx = available_loss_indices[lowest_available_loss_idx]

        behavior_probabilities = np.zeros(shape=len(sensor_candidate_range))
        print(lowest_icgm_loss_idx)
        behavior_probabilities[lowest_icgm_loss_idx] = 1.0

        return behavior_probabilities

    def get_info_stateless(self):
        return {
            "name": "SensoriCGMModelControlsBoundary"
        }


class DexcomG6RateModel(SensoriCGMModelOverlayBase):

    # https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170088.pdf
    # See Trend Accuracy Tables

    def __init__(self):

        self.p_true_given_icgm = np.array([
            [0.533, 0.350, 0.099, 0.015, 0.000, 0.002],
            [0.074, 0.569, 0.325, 0.029, 0.003, 0.000],
            [0.004, 0.095, 0.769, 0.125, 0.006, 0.001],
            [0.001, 0.010, 0.262, 0.606, 0.106, 0.016],
            [0.000, 0.004, 0.031, 0.268, 0.529, 0.168],
            [0.001, 0.001, 0.008, 0.056, 0.221, 0.713]
        ])

        self.icgm_rate_counts = [463, 2077, 7986, 5199, 1734, 1367]

        self.joint_counts = np.zeros(shape=self.p_true_given_icgm.shape)
        for i in range(self.p_true_given_icgm.shape[1]):
            icgm_counts = [self.icgm_rate_counts[i] * v for v in self.p_true_given_icgm[i]]
            self.joint_counts[i] = icgm_counts

        self.true_rate_counts = np.sum(self.joint_counts, axis=0)
        self.p_icgm_given_true = np.zeros(shape=self.p_true_given_icgm.shape)
        for i in range(self.p_true_given_icgm.shape[1]):
            self.p_icgm_given_true[:, i] = self.joint_counts[:, i] / self.true_rate_counts[i]

    def get_rate_idx(self, rate):
        idx = None
        if rate < -2.0:
            idx = 0
        elif -2.0 <= rate < -1.0:
            idx = 1
        elif -1 <= rate < 0:
            idx = 2
        elif 0 <= rate <= 1:
            idx = 3
        elif 1 < rate <= 2:
            idx = 4
        elif rate > 2.0:
            idx = 5

        return idx

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):

        true_bg_history = kwargs["true_bg_history"]
        sensor_bg_history = kwargs["sensor_bg_history"]

        if len(true_bg_history) < 2:  # Initial condition
            behavior_probabilities = np.ones(shape=len(sensor_candidate_range))
        else:

            true_rate = (true_bg_history[-1] - true_bg_history[-2]) / 5.0
            behavior_probabilities = np.zeros(shape=len(sensor_candidate_range))

            for i, icgm_val in enumerate(sensor_candidate_range):
                icgm_rate = (icgm_val - sensor_bg_history[-1]) / 5.0  # TODO: need to consider edge cases, nan & missing

                true_idx = self.get_rate_idx(true_rate)
                icgm_idx = self.get_rate_idx(icgm_rate)
                try:
                    behavior_probabilities[i] = self.p_icgm_given_true[icgm_idx, true_idx]
                except ValueError:
                    print(true_rate, icgm_rate, true_idx, icgm_idx, self.p_icgm_given_true)
                    behavior_probabilities[i] = self.p_icgm_given_true[true_idx, true_idx]
                    # pdb.set_trace()

        return behavior_probabilities

    def get_info_stateless(self):
        return {
            "name": "DexcomRateModel"
        }


class DexcomG6ValueModel(SensoriCGMModelOverlayBase):

    # https://www.accessdata.fda.gov/cdrh_docs/reviews/DEN170088.pdf
    # See Concurrence of Dexcom G6 System Readings and Comparator Values by Comparator

    def __init__(self, concurrency_table="adult"):

        if concurrency_table == "adult":
            self.p_icgm_given_true = np.array(DEXCOM_CONCURRENCY_PG23_ADULT)
            self.comparator_totals = [27, 1180, 2191, 3503, 2910, 2457, 2755, 2383, 1601, 437, 23]
        elif concurrency_table == "pediatric":
            self.p_icgm_given_true = np.array(DEXCOM_CONCURRENCY_PG24_PED)
            self.comparator_totals = [2, 127, 559, 1254, 1081, 955, 913, 570, 282, 74, 10]
        elif concurrency_table == "TP_iCGM":
            self.p_icgm_given_true = np.array(TP_iCGM_CONCURRENCY)
            self.comparator_totals = [2, 127, 559, 1254, 1081, 955, 913, 570, 282, 74, 10]

        self.total = np.sum(self.comparator_totals)

        self.joint_prob = np.zeros(shape=self.p_icgm_given_true.shape)
        for col_idx in range(self.p_icgm_given_true.shape[1]):
            for row_idx in range(self.p_icgm_given_true.shape[0]):
                self.joint_prob[row_idx][col_idx] = self.p_icgm_given_true[row_idx][col_idx] * self.comparator_totals[col_idx] / self.total

        assert np.abs(1.0 - np.sum(self.joint_prob)) < 1e-3

    def get_rate_idx(self, value):

        idx = None
        if value < 40:
            idx = 0
        elif 40 <= value <= 60:
            idx = 1
        elif 60 < value <= 80:
            idx = 2
        elif 80 < value <= 120:
            idx = 3
        elif 120 < value <= 160:
            idx = 4
        elif 160 < value <= 200:
            idx = 5
        elif 200 < value <= 250:
            idx = 6
        elif 250 < value <= 300:
            idx = 7
        elif 300 < value <= 350:
            idx = 8
        elif 350 < value <= 400:
            idx = 9

        return idx

    def get_conditional_probabilities(self, sensor_candidate_range, **kwargs):

        true_bg_history = kwargs["true_bg_history"]
        behavior_probabilities = np.zeros(shape=len(sensor_candidate_range))

        for i, icgm_val in enumerate(sensor_candidate_range):

            true_idx = self.get_rate_idx(true_bg_history[-1])
            icgm_idx = self.get_rate_idx(icgm_val)
            behavior_probabilities[i] = self.p_icgm_given_true[icgm_idx, true_idx]

        return behavior_probabilities

    def get_joint_probability(self, true_bg, icgm_bg):

        true_idx = self.get_rate_idx(true_bg)
        icgm_idx = self.get_rate_idx(icgm_bg)
        return self.joint_prob[icgm_idx][true_idx]

    def get_info_stateless(self):
        return {
            "name": "DexcomRateModel"
        }


def get_sine_data(num_hours=24*10):

    num_data_points = num_hours * 12
    return np.sin(np.arange(num_data_points) * 2 * np.pi * 3 / num_data_points) * 180 + 220


def get_test_data():

    not_fitting_list_mean_shift_filelist = sorted(list(set(list(pd.read_csv(
        "/Users/csummers/Downloads/mean_shift_special_controls_not_passed_icgm-sensitivity-analysis-scenarios-2020-07-10-nogit.csv")[
                                                             "training_scenario_filename"]))))

    scenarios_dir = "/Users/csummers/dev/data-science-simulator/data/raw/icgm-sensitivity-analysis-scenarios-2020-07-10"

    # scenario_filename = "train_80a5c60283c2b095d69cca4f64c26e2564958a07e2f0e19fafd073ed47d2b5e7.csv_condition9.csv"
    scenario_filename = not_fitting_list_mean_shift_filelist[3]

    df = pd.read_csv(os.path.join(scenarios_dir, scenario_filename))
    true_bg_trace = df.iloc[50, 2:].astype(float).values
    return true_bg_trace


def does_meet_special_controls_independent(true_bg_trace, sensor_batch_bg_trace):

    df = preprocess_data(true_bg_trace, sensor_batch_bg_trace, icgm_range=[40, 400], ysi_range=[0, 900])

    acc_results = calc_icgm_sc_table(df, "generic")
    has_pairs_mask = acc_results["nPairs"] > 0
    difference = acc_results[has_pairs_mask]["icgmSensorResults"] - acc_results[has_pairs_mask]["icgmSpecialControls"]

    meets_special_controls = not (difference < 0).any()
    return meets_special_controls, acc_results, difference


def single_sensor_to_sensor_batch(sensor_bg_trace, n_sensors=30):

    sensor_batch_bg_traces = np.repeat(np.array(sensor_bg_trace)[:, np.newaxis], n_sensors, axis=1)
    return sensor_batch_bg_traces


def get_special_controls_sweep(n_iters=5):

    special_controls_sweep = [iCGM_THRESHOLDS]
    for i in range(1, n_iters + 1):
        new_controls = {}
        for criteria, value in iCGM_THRESHOLDS.items():

            if criteria not in ["J", "K"]:
                new_controls[criteria] = min(1.0, value + i * (1 - value) / n_iters)
            else:
                new_controls[criteria] = max(0.0, value - i * value / n_iters)

        special_controls_sweep.append(new_controls)

    return special_controls_sweep


if __name__ == "__main__":

    true_bg_trace = get_test_data()
    # true_bg_trace = true_bg_trace[-300:]
    # true_bg_trace = true_bg_trace[:300]
    true_bg_trace = true_bg_trace[500:800]

    # true_bg_trace = get_sine_data(num_hours=24*1)

    sensor_run_durations_minutes = []

    special_controls = {
        # "fda": iCGM_THRESHOLDS,
        "g6": G6_THRESHOLDS,
        # "worst": {
        #     "A": 0.01,
        #     "B": 0.01,
        #     "C": 0.01,
        #     "D": 0.02,
        #     "E": 0.02,
        #     "F": 0.02,
        #     "G": 0.01,
        #     "J": 0.01,
        #     "K": 0.01,
        # }
    }

    # special_controls = get_special_controls_sweep(n_iters=5)

    fig, ax = plt.subplots(2, 1)

    behavior_model_compare = {
        "dexcom_values_rates": [DexcomG6ValueModel(), DexcomG6RateModel()],
        # "dexcom_values": [DexcomG6ValueModel()],
        # "dexcom_rates": [DexcomG6RateModel()],
        # "max_noise": [SensoriCGMModelOverlayNoiseBiasWorst(50)],
        # "spurious": [SensoriCGMModelOverlayV1(bias=0, sigma=5, delay=2, spurious_value_prob=0.1, num_consecutive_spurious=2)]
    }
    # SensoriCGMModelUniform()
    # SensoriCGMModelControlsBoundary()

    np.random.seed(0)
    for _ in range(1):
        common_seed = np.random.randint(0, 1e6)
        for ctrls_name, sp_ctrls in special_controls.items():
            for models_name, behavior_models in behavior_model_compare.items():

                random_state = np.random.RandomState(common_seed)

                sensor_id = "{}. model: {}".format(ctrls_name, models_name)
                sensor_config = SensorConfig(sensor_bg_history=GlucoseTrace())
                sensor_config.history_window_hrs = 24 * 1

                sensor_config.behavior_models = behavior_models

                sensor_config.do_look_ahead = True
                sensor_config.look_ahead_min_prob = 0.0

                sensor_config.sensor_range = range(40, 401)
                sensor_config.special_controls = sp_ctrls
                sensor_config.initial_controls = CLEAN_INITIAL_CONTROLS

                t0 = datetime.datetime.now()
                sensor = SensoriCGM(t0, sensor_config=sensor_config, random_state=random_state)

                start_time = time.time()

                time_delta = datetime.timedelta(minutes=5)
                prev_datetime = t0
                sensor_bg_trace = []
                for i, true_bg in enumerate(true_bg_trace):
                    next_time = prev_datetime + time_delta
                    sensor.update(next_time, patient_true_bg=true_bg, do_plot=True)
                    sensor_bg = sensor.get_bg(true_bg)
                    sensor_bg_trace.append(sensor_bg)
                    prev_datetime = next_time

                icgm_evaluator = iCGMEvaluator(special_controls=iCGM_THRESHOLDS)

                sensor_batch_bg_traces = single_sensor_to_sensor_batch(sensor_bg_trace)
                try:
                    meets_special_controls, reason = icgm_evaluator.does_sensor_batch_meet_special_controls(true_bg_trace, sensor_batch_bg_traces)
                    print("Independent Validation New")
                    print("Meets special controls: {}. Reason {}.".format(meets_special_controls, reason))
                    sensor_mard = np.mean(np.abs(np.array(true_bg_trace) - sensor_bg_trace) / true_bg_trace)
                    sensor_mbe = np.mean(np.array(sensor_bg_trace) - true_bg_trace)
                    print("MARD: {}. MBE: {}".format(sensor_mard, sensor_mbe))
                except Exception as e:
                    print("Exception in icgm_evaluator", e)

                meets_special_controls, acc_results, difference = does_meet_special_controls_independent(true_bg_trace, sensor_batch_bg_traces.T)

                print("Independent Validation Old")
                print("Meets special controls: {}. Acc Results: {}".format(meets_special_controls, acc_results))

                sensor_run_durations_minutes.append((time.time() - start_time) / 60.0)

                ax[0].plot(sensor_bg_trace, label=sensor_id, linestyle="-.", marker="*")
                ax[0].plot(true_bg_trace, label="true")
                ax[0].set_ylabel("BG (mg/dL)")
                ax[0].set_xlabel("Time (5min)")
                ax[0].legend()

                from scipy import stats
                error_distr = sensor_bg_trace - np.array(true_bg_trace)
                x = np.linspace(np.min(error_distr), np.max(error_distr), 100)
                mu = np.mean(error_distr)
                sigma = np.std(error_distr)
                print("Mu {}. Sigma {}.".format(mu, sigma))
                ax[1].plot(x, stats.norm.pdf(x, mu, sigma))
                # ax[1].plot(x, stats.norm.pdf(x, mu, 4.99))
                ax[1].hist(error_distr, density=True, alpha=0.3, label=sensor_id)
                ax[1].legend()

                sensor.state.display()

    print("Sensor Avg Run Minutes for {} sensors: {}".format(len(sensor_run_durations_minutes), np.average(sensor_run_durations_minutes)))
    plt.show()

