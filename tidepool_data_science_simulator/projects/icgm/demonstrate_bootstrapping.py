__author__ = "Cameron Summers"

import matplotlib.pyplot as plt

from tidepool_data_science_simulator.evaluation.icgm_eval import iCGMEvaluator
from tidepool_data_science_simulator.models.sensor_icgm import iCGM_THRESHOLDS


if __name__ == "__main__":

    icgm_evaluator = iCGMEvaluator(iCGM_THRESHOLDS, bootstrap_num_values=200000)

    lb, means = icgm_evaluator.bootstrap_95_lower_confidence_bound([0] * 5 + [100]*19)

    print(lb)

    plt.hist(means)
    plt.show()
