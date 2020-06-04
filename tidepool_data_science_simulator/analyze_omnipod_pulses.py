__author__ = "Cameron Summers"


__author__ = "Cameron Summers"

import datetime
import numpy as np
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.makedata.make_patient import get_canonical_risk_patient
from tidepool_data_science_simulator.models.pump import OmnipodMissingPulses


def analyze_omnipod_missing_pulses():

    current_time, patient = get_canonical_risk_patient(pump_class=OmnipodMissingPulses)

    delivered_insulin = []
    undelivered_insulin = []

    temp_basal_scenario = [0.55, 0.45] * 12  # This scenario gives no insulin

    for temp_basal_value in temp_basal_scenario:
        next_time = current_time + datetime.timedelta(minutes=5)

        patient.pump.update(next_time)
        patient.pump.set_temp_basal(temp_basal_value, "U")

        delivered_insulin.append(patient.pump.insulin_delivered_last_update)
        undelivered_insulin.append(patient.pump.undelivered_insulin)

    total_delivered_insulin = np.sum(delivered_insulin)
    total_undelivered_insulin = np.sum(undelivered_insulin)
    total_expected_insulin = total_delivered_insulin + total_undelivered_insulin
    print("Total Delivered Insulin  {:>4} ({:.0f}%)".format(total_delivered_insulin,
                                                           total_delivered_insulin / total_expected_insulin * 100.0)),
    print("Total Undelivered Insulin {:>4} ({:.0f}%)".format(total_undelivered_insulin,
                                                            total_undelivered_insulin / total_expected_insulin * 100.0))

    plt.title("Omnipod Missing Insulin Pulses")
    plt.plot(delivered_insulin, label="delivered")
    plt.plot(undelivered_insulin, label="undelivered")
    plt.plot(temp_basal_scenario, label="Temp Basal Values")
    plt.legend()
    plt.show()


if __name__ == "__main__":

    analyze_omnipod_missing_pulses()