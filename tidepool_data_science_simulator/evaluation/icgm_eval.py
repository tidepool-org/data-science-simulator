__author__ = "Cameron Summers"

import numpy as np
from scipy.stats import norm

import matplotlib.pyplot as plt


class iCGMEvaluator(object):
    """
    Evaluates whether traces meet icgm special controls.

    Assumes traces passed are contiguous with no missing values.
    """
    def __init__(self, special_controls, bootstrap_num_values=5000):

        self.range_counter = None
        self.special_controls = special_controls
        self.bootstrap_num_values = bootstrap_num_values

        self.sensor_criteria_values = {
            criteria_key: [] for criteria_key in special_controls.keys()
        }

    def get_sensor_range_mask(self, sensor_bg_trace, low_bound, high_bound):
        """
        Get a mask for a sensor range bounds, inclusive

        Parameters
        ----------
        sensor_bg_trace: ndarray
        low_bound: int
        high_bound: int

        Returns
        -------
        ndarray
            Boolean mask of values within the bounds, inclusive
        """

        sensor_bg_array = np.array(sensor_bg_trace)
        range_mask = (low_bound <= sensor_bg_array) & (sensor_bg_array <= high_bound)

        return range_mask

    def get_errors(self, true_bg_trace, sensor_bg_array):
        """
        Get the absolute and percentage errors between the true bg trace and the sensor bg trace

        Parameters
        ----------
        true_bg_trace
        sensor_bg_array

        Returns
        -------
        (ndarray, ndarray)
            The absolute errors, the percentage errors
        """

        abs_errors = np.abs(sensor_bg_array - true_bg_trace)
        percent_errors = abs_errors / true_bg_trace

        return abs_errors, percent_errors

    def validate_true_sensor_trace_pairs(self, true_bg_trace, sensor_bg_array):
        """
        Validate incoming bg trace values for computation

        Parameters
        ----------
        true_bg_trace
        sensor_bg_array

        Raises
        -------
        ValueError
        """

        if not isinstance(sensor_bg_array, np.ndarray):
            raise ValueError("Expected sensor_bg_array to be numpy array")

        if len(true_bg_trace) == 0:
            raise ValueError("True bg trace has zero values")

        if len(sensor_bg_array) == 0:
            raise ValueError("Sensor bg array have zero values")

        if len(true_bg_trace) != sensor_bg_array.shape[0]:
            raise ValueError("True bg and sensor bg array time dimensions don't match")

        if np.isnan(sensor_bg_array).any():
            raise ValueError("NaNs not handled here. TODO")

        if np.isnan(true_bg_trace).any():
            raise ValueError("NaNs not handled here. TODO")

    def bootstrap_95_lower_confidence_bound(self, value_list, percentile=2.5):
        """
        Compute the 95% lower confidence bound using the Bootstrap method.

        Parameters
        ----------
        num_sensors: int
            Number of sensors in the sample

        value_list: list
            Values from the sensor batch, e.g. criteria A values

        Returns
        -------
        float
            The 95% Lower Confidence Bound
        """
        bootstrapped_means = []
        for i in range(self.bootstrap_num_values):
            sensors_sample = []
            for s_i in range(len(value_list)):
                value = np.random.choice(value_list)
                sensors_sample.append(value)

            mean_estimate = np.mean(sensors_sample)
            bootstrapped_means.append(mean_estimate)

        return np.percentile(bootstrapped_means, percentile), bootstrapped_means

    def does_sensor_batch_meet_special_controls(self, true_bg_trace, sensor_batch_bg_array):
        """
        Determine if a true bg trace and a batch of sensor traces meet FDA special controls.

        Parameters
        ----------
        true_bg_trace: list
            The true bg trace

        sensor_bg_trace: ndarray
            The sensor bg traces, shape=(n rows x m sensors)

        Returns
        -------
        (bool, str)
            Whether it meets and the relevant criteria for failure
        """
        self.validate_true_sensor_trace_pairs(true_bg_trace, sensor_batch_bg_array)

        does_meet, reason = self.does_meet_non_confidence_bound_criteria(true_bg_trace, sensor_batch_bg_array)
        if not does_meet:
            return does_meet, reason

        does_meet, reason = self.does_meet_confidence_bound_criteria(true_bg_trace, sensor_batch_bg_array)
        if not does_meet:
            return does_meet, reason

        return True, ""

    def does_meet_non_confidence_bound_criteria(self, true_bg_trace, sensor_batch_bg_array):
        """
        Determine if the traces meet the confidence bound criteria.

        Parameters
        ----------
        true_bg_trace: list
            True bg trace

        sensor_batch_bg_array: ndarray
            Sensor batch array

        Returns
        -------
        (bool, str)
            True if meets, reason for failure
        """

        for s_i in range(sensor_batch_bg_array.shape[1]):

            sensor_bg_array = sensor_batch_bg_array[:, s_i]
            true_bg_array = np.array(true_bg_trace)

            true_bg_gt180_mask = true_bg_array > 180
            sensor_bg_lt70_mask = sensor_bg_array < 70

            if (true_bg_gt180_mask & sensor_bg_lt70_mask).any():
                return False, "H: Extreme values"

            true_bg_lt70_mask = true_bg_array < 70
            sensor_bg_gt180_mask = sensor_bg_array > 180

            if (true_bg_lt70_mask & sensor_bg_gt180_mask).any():
                return False, "I: Extreme values"

            tbg_rates = np.diff(true_bg_array) / 5.0  # TODO: make configurable
            sbg_rates = np.diff(sensor_bg_array) / 5.0

            mask_J_tbg = tbg_rates < -2.0
            # total_J = np.sum(mask_J_tbg)  # FIXME: get the right denominator
            total_J = len(sensor_bg_array)
            mask_J_sbg = sbg_rates[mask_J_tbg] > 1.0
            num_J_sbg = np.sum(mask_J_sbg)
            if num_J_sbg / total_J > self.special_controls["J"]:
                return False, "J: Extreme rates"

            mask_K_tbg = tbg_rates > 2.0
            # total_K = np.sum(mask_K_tbg)
            total_K = len(sensor_bg_array)
            mask_K_sbg = sbg_rates[mask_K_tbg] < -1.0
            num_K_sbg = np.sum(mask_K_sbg)
            if num_K_sbg / total_K > self.special_controls["J"]:
                return False, "K: Extreme rates"

        return True, ""

    def does_meet_confidence_bound_criteria(self, true_bg_trace, sensor_batch_bg_array):

        for sensor_idx in range(sensor_batch_bg_array.shape[1]):

            sensor_bg_array = sensor_batch_bg_array[:, sensor_idx]

            abs_error, percent_errors = self.get_errors(true_bg_trace, sensor_bg_array)

            # Overall Criteria G
            total_values = len(true_bg_trace)
            total_within_20_percent = np.sum(percent_errors < 0.20)
            criteria_G = total_within_20_percent / total_values
            self.sensor_criteria_values["G"].append(criteria_G)

            # Range Criteria A-F
            mask_lt70 = self.get_sensor_range_mask(sensor_bg_array, -np.inf, 69)
            mask_70_180 = self.get_sensor_range_mask(sensor_bg_array, 70, 180)
            mask_gt180 = self.get_sensor_range_mask(sensor_bg_array, 181, np.inf)

            num_lt70 = np.sum(mask_lt70)
            num_70_180 = np.sum(mask_70_180)
            num_gt180 = np.sum(mask_gt180)

            if num_lt70 > 0:

                abs_errors_lt70 = abs_error[mask_lt70]
                mask_A = abs_errors_lt70 < 15
                num_A_small = np.sum(mask_A)

                criteria_A = num_A_small / num_lt70
                self.sensor_criteria_values["A"].append(criteria_A)

                mask_D = (abs_errors_lt70 >= 15) & (abs_errors_lt70 < 40)
                num_D_medium = np.sum(mask_D)

                criteria_D = (num_A_small + num_D_medium) / num_lt70
                self.sensor_criteria_values["D"].append(criteria_D)

            if num_70_180 > 0:

                percent_errors_70_180 = percent_errors[mask_70_180]
                mask_B = percent_errors_70_180 < 0.15
                num_B_small = np.sum(mask_B)

                criteria_B = num_B_small / num_70_180
                self.sensor_criteria_values["B"].append(criteria_B)

                mask_E = (percent_errors_70_180 >= 0.15) & (percent_errors_70_180 < 0.4)
                num_E_medium = np.sum(mask_E)

                critieria_E = (num_B_small + num_E_medium) / num_70_180
                self.sensor_criteria_values["E"].append(critieria_E)

            if num_gt180 > 0:

                percent_errors_gt180 = percent_errors[mask_gt180]
                mask_C = percent_errors_gt180 < 0.15
                num_C_small = np.sum(mask_C)

                criteria_C = num_C_small / num_gt180
                self.sensor_criteria_values["C"].append(criteria_C)

                mask_F = (percent_errors_gt180 >= 0.15) & (percent_errors_gt180 < 0.4)
                num_F_medium = np.sum(mask_F)

                criteria_F = (num_C_small + num_F_medium) / num_gt180
                self.sensor_criteria_values["F"].append(criteria_F)

        for criteria_key, sensor_values in self.sensor_criteria_values.items():

            boostrap_lower_95_bound = self.bootstrap_95_lower_confidence_bound(sensor_values)
            if boostrap_lower_95_bound <= self.special_controls[criteria_key]:
                return False, criteria_key

        return True, ""


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


def get_test_abs_icgm_trace(low_sbg=40, high_sbg=70, low_error=0, high_error=15):

    tbg_trace = []
    sbg_trace = list(range(low_sbg, high_sbg))
    errors = []

    for sbg in sbg_trace:
        error = np.random.randint(low_error, high_error)
        tbg = sbg + error * np.random.choice([-1, 1])

        tbg_trace.append(tbg)
        errors.append(error)

    return tbg_trace, sbg_trace, errors


def get_test_percent_icgm_trace(low_sbg, high_sbg, low_error, high_error):

    tbg_trace = []
    sbg_trace = list(range(low_sbg, high_sbg))

    for sbg in sbg_trace:
        error = np.random.uniform(low_error, high_error)
        tbg = sbg / (1 + error * np.random.choice([-1, 1]))

        tbg_trace.append(tbg)

    return tbg_trace, sbg_trace


def get_test_rate_icgm_trace(n_values, time_delta_minutes, low_sbg_rate, high_sbg_rate, low_tbg_rate, high_tbg_rate):

    tbg_trace = [np.random.randint(40, 400)]
    sbg_trace = [np.random.randint(40, 400)]

    for i in range(n_values):
        sbg_rate = np.random.uniform(low_sbg_rate, high_sbg_rate)
        sbg_value = sbg_trace[-1] + sbg_rate * time_delta_minutes

        tbg_rate = np.random.uniform(low_tbg_rate, high_tbg_rate)
        tbg_value = tbg_trace[-1] + tbg_rate * time_delta_minutes

        if not (40 <= sbg_value <= 400) or not (40 <= tbg_value <= 400):
            break

        tbg_trace.append(tbg_value)
        sbg_trace.append(sbg_value)

    return tbg_trace, sbg_trace


def test_icgm_validation_trace():

    tbg_trace_J, sbg_trace_J = get_test_rate_icgm_trace(n_values=10,
                                           time_delta_minutes=5,
                                           low_sbg_rate=1.1,
                                           high_sbg_rate=1.5,
                                           low_tbg_rate=-2.5,
                                           high_tbg_rate=-2.1)

    tbg_trace_K, sbg_trace_K = get_test_rate_icgm_trace(n_values=10,
                                                        time_delta_minutes=5,
                                                        low_sbg_rate=-1.5,
                                                        high_sbg_rate=-1.1,
                                                        low_tbg_rate=2.1,
                                                        high_tbg_rate=2.5)

    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_J, single_sensor_to_sensor_batch(sbg_trace_J).T)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_K, single_sensor_to_sensor_batch(sbg_trace_K).T)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_J + tbg_trace_K, single_sensor_to_sensor_batch(sbg_trace_J + sbg_trace_K).T)


    # tbg_trace_A, icgm_trace_A, e_A = get_test_abs_icgm_trace(low_sbg=40, high_sbg=70, low_error=5, high_error=6)
    # tbg_trace_D, icgm_trace_D, e_D = get_test_abs_icgm_trace(low_sbg=40, high_sbg=70, low_error=20, high_error=21)
    # tbg_trace_ADc, icgm_trace_ADc, e_ADc = get_test_abs_icgm_trace(low_sbg=40, high_sbg=70, low_error=41, high_error=50)
    #
    # tbg_trace_B, icgm_trace_B = get_test_percent_icgm_trace(low_sbg=70, high_sbg=181, low_error=0, high_error=0.15)
    # tbg_trace_E, icgm_trace_E = get_test_percent_icgm_trace(low_sbg=70, high_sbg=181, low_error=0.15, high_error=0.40)
    # tbg_trace_BEc, icgm_trace_BEc = get_test_percent_icgm_trace(low_sbg=70, high_sbg=181, low_error=0.40, high_error=0.50)
    #
    # tbg_trace_C, icgm_trace_C = get_test_percent_icgm_trace(low_sbg=181, high_sbg=400, low_error=0, high_error=0.15)
    # tbg_trace_F, icgm_trace_F = get_test_percent_icgm_trace(low_sbg=181, high_sbg=400, low_error=0.15, high_error=0.40)
    # tbg_trace_CFc, icgm_trace_CFc = get_test_percent_icgm_trace(low_sbg=181, high_sbg=400, low_error=0.40, high_error=0.50)
    #
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_A, icgm_trace_A)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_D, icgm_trace_D)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_ADc, icgm_trace_ADc)
    #
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_A + tbg_trace_D, icgm_trace_A + icgm_trace_D)
    #
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_B, icgm_trace_B)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_E, icgm_trace_E)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_BEc, icgm_trace_BEc)
    #
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_B + tbg_trace_E, icgm_trace_B + icgm_trace_E)
    #
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_C, icgm_trace_C)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_F, icgm_trace_C)
    # meets, results, diff = does_meet_special_controls_independent(tbg_trace_CFc, icgm_trace_CFc)


if __name__ == "__main__":

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
    evaluator = iCGMEvaluator(iCGM_THRESHOLDS)

    mu = 81
    # sigma = 10
    num_sensors = 300
    for worst in range(1):
        # values = [81] * 99 + [worst]
        # values = [np.random.normal(mu, sigma) for i in range(num_sensors - 1)] + [worst]
        values = [100] * 100 + [worst]

        lower_95_bound = evaluator.bootstrap_95_lower_confidence_bound(values)
        print(worst, lower_95_bound, np.mean(values))


