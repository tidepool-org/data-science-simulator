__author__ = "Cameron Summers"

import datetime
import numpy as np

from tidepool_data_science_simulator.models.measures import Carb
#from tidepool_data_science_simulator.models.simulation import CarbTimeline, BolusTimeline, TempBasalTimeline
from tidepool_data_science_simulator.utils import get_bernoulli_trial_uniform_step_prob


class Action(object):
    """
    A class for user executed actions that are not inputs.
    """
    def __init__(self, name):
        self.name = name

    def execute(self, **kwargs):
        raise NotImplementedError


class UserInput(object):
    def __init__(self, name, time_start, time_end=None):
        self.name = name
        self.time_start = time_start
        self.time_end = time_end


class MealModel(UserInput):
    """
    A meal that says if it is time for the meal and probabilistically determines carbs.
    """
    def __init__(self, name, time_start, time_end, prob_of_eating):

        super().__init__(name, time_start, time_end)
        self.prob_of_eating = prob_of_eating

        # Get number of simulation steps in meal time range
        datetime_start = datetime.datetime.combine(datetime.date.today(), time_start)
        datetime_end = datetime.datetime.combine(datetime.date.today(), time_end)
        datetime_delta = datetime_end - datetime_start
        datetime_delta_minutes = datetime_delta.total_seconds() / 60
        datetime_delta_steps = int(datetime_delta_minutes / 5.0)  # 5 min per step
        self.num_steps = datetime_delta_steps

        # num_steps Bernoulli trials to get prob_of_eating
        self.step_prob = get_bernoulli_trial_uniform_step_prob(self.num_steps, prob_of_eating)

    def is_meal_time(self, time):

        return self.time_start <= time.time() < self.time_end

    def get_carb(self):

        carb = Carb(
            value=np.random.choice(range(20, 40)),
            units="g",
            duration_minutes=np.random.choice([3 * 60, 4 * 60, 5 * 60]),
        )

        return carb

    def __repr__(self):

        return "{}".format(self.name)