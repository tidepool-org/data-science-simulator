import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

#%% Functions
def calc_p_hat(true_bg, icgm_bg):
    within_15_mg_dl = (true_bg - 15 <= icgm_bg) & (icgm_bg <= true_bg + 15)
    within_40_mg_dl = (true_bg - 40 <= icgm_bg) & (icgm_bg <= true_bg + 40)
    within_15_percent = (true_bg * 0.85 <= icgm_bg) & (icgm_bg <= true_bg * 1.15)
    within_20_percent = (true_bg * 0.80 <= icgm_bg) & (icgm_bg <= true_bg * 1.20)
    within_40_percent = (true_bg * 0.60 <= icgm_bg) & (icgm_bg <= true_bg * 1.40)

    lt_70 = icgm_bg < 70
    bt_70_180 = (icgm_bg >= 70) & (icgm_bg <= 180)
    gt_180 = icgm_bg > 180

    return (pd.DataFrame.from_dict(
        {"A": [np.mean(within_15_mg_dl[lt_70]), np.sum([lt_70])],
         "B": [np.mean(within_15_percent[bt_70_180]), np.sum(bt_70_180)],
         "C": [np.mean(within_15_percent[gt_180]), np.sum(gt_180)],
         "D": [np.mean(within_40_mg_dl[lt_70]), np.sum(lt_70)],
         "E": [np.mean(within_40_percent[bt_70_180]), np.sum(bt_70_180)],
         "F": [np.mean(within_40_percent[gt_180]), np.sum(gt_180)],
         "G": [np.mean(within_20_percent), np.size(within_20_percent)]},
        orient='index',
        columns=['p_hat', 'n']))

def calc_lb_normal_approx(true_bg, icgm_bg, alpha = 0.05):
    z = norm.ppf(alpha)
    p_hat = calc_p_hat(true_bg, icgm_bg)
    return(p_hat['p_hat'] + z * np.sqrt(p_hat['p_hat'] * (1 - p_hat['p_hat']) / p_hat['n']))

def compare_CILB(
    n_bootstrap_samples = 5000,
    alpha = 0.05, # for confidence interval
    n_sensors = 100,
    n_days_per_sensor = 10,
    sensor_time_interval_in_min = 5,
    true_bg_center = 140,
    icgm_bias = 0,
    true_bg_var = 2000.0,
    icgm_var = 160.0
):
    # Simulate
    time = np.arange(0, n_days_per_sensor * 24 * 60 + sensor_time_interval_in_min, sensor_time_interval_in_min)
    true_bg = np.abs(
        GaussianProcessRegressor(kernel=true_bg_var * Matern(length_scale=85.0, nu=1.5)).sample_y(time[:, np.newaxis],
                                                                                             n_sensors) + true_bg_center)
    icgm_error = GaussianProcessRegressor(kernel=icgm_var * Matern(length_scale=40.0, nu=0.5)).sample_y(
        time[:, np.newaxis], n_sensors) + icgm_bias
    icgm_bg = np.abs(true_bg + icgm_error)

    # Run Bootstrap
    bootstrap_results = np.empty([7, n_bootstrap_samples])

    for i in range(0, n_bootstrap_samples):
        sampled_sensor_ids = np.random.choice(n_sensors, n_sensors)
        bootstrap_results[:, i] = calc_p_hat(true_bg[:, sampled_sensor_ids], icgm_bg[:, sampled_sensor_ids])["p_hat"]

    # Create Comparison Table
    comp_table = pd.concat([calc_p_hat(true_bg, icgm_bg),
        calc_lb_normal_approx(true_bg, icgm_bg, alpha=alpha).to_frame("LB_Normal_Approx"),
        pd.DataFrame(np.percentile(bootstrap_results, alpha * 100, 1),
                            index=["A", "B", "C", "D", "E", "F", "G"],
                            columns=["LB_Bootstrap"])], 1)

    return(comp_table)

#%% Do some comparisons
r1 = compare_CILB(n_sensors=100, n_days_per_sensor=10)
r2 = compare_CILB(n_sensors=20, n_days_per_sensor=10)
r3 = compare_CILB(n_sensors=100, n_days_per_sensor=1)
r4 = compare_CILB(n_sensors=20, n_days_per_sensor=1)

all_differences = pd.concat([x["LB_Normal_Approx"] - x["LB_Bootstrap"] for x in [r1, r2, r3, r4]], 1)

# With 100 sensors and 10 days each, the maximum difference in CILB is 0.5%

# Gets worse with smaller sample sizes
np.max(all_differences)
# also worse with lower p (surprising)
np.max(all_differences, 1)

# # Plot some of the simulations if you're curious.
# wrote this before I put everything in a function
# plt.plot(time[:], icgm_bg[:, 1:4], lw=1)
# plt.plot(time[:], icgm_error[:, 1:4], lw=1)
# plt.show()100, n_days_per_sensor = 10)