__author__ = "Cameron Summers"

import datetime
import numpy as np
import matplotlib.pyplot as plt

from tidepool_data_science_simulator.models.patient import MealModel


def plot_meal_distribution():

    breakfast = MealModel(
        name="Breakfast",
        time_start=datetime.time(hour=7),
        time_end=datetime.time(hour=9),
        prob_of_eating=0.99
    )
    print("Step probability:", breakfast.step_prob)

    num_steps = breakfast.num_steps
    meals_had = 0
    num_days = 10000
    meal_times = []
    for i in range(num_days):
        start_time = datetime.datetime(year=2020, month=1, day=1, hour=7)

        for j in range(num_steps):
            time = start_time + datetime.timedelta(minutes=5)
            u = np.random.uniform()

            step_prob = breakfast.step_prob
            if breakfast.is_meal_time(time=time) and u < step_prob:
                meals_had += 1
                meal_times.append(j)
                break

    print("Probability of eating breakfast:", meals_had / num_days)
    plt.hist(meal_times)
    plt.xlabel("Step (5 min)")
    plt.title("Meal Time Distr")
    plt.show()


plot_meal_distribution()
