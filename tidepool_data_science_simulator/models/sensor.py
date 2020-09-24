__author__ = "Cameron Summers"

"""
Sensor model classes
"""

import copy
import datetime as dt

from tidepool_data_science_simulator.models.simulation import SimulationComponent


class Sensor(SimulationComponent):

    def __init__(self, time, sensor_config):

        super().__init__()
        self.time = time
        self.sensor_config = copy.deepcopy(sensor_config)
        self.sensor_bg_history = self.sensor_config.sensor_bg_history

        self.current_sensor_bg = self.sensor_config.sensor_bg_history.bg_values[0]
        self.current_sensor_bg_prediction = None

    def get_bg(self, true_bg):
        raise NotImplementedError

    def get_bg_trace(self, true_bg_trace):
        raise NotImplementedError

    def get_info_stateless(self):

        stateless_info = {
            "name": self.name,
            "config": self.sensor_config.get_info_stateless()
        }
        return stateless_info

    def get_state(self):
        """
        Get the state of the sensor
        """
        return SensorState(
            self.current_sensor_bg,
            self.current_sensor_bg_prediction
        )

    def update(self, time, **kwargs):
        """
        Get the current sensed bg and store.

        Parameters
        ----------
        time: datetime
        """
        self.time = time
        self.set_random_values()

        true_bg = kwargs["patient_true_bg"]
        true_bg_prediction = kwargs["patient_true_bg_prediction"]
        self.current_sensor_bg = self.get_bg(true_bg)
        self.current_sensor_bg_prediction = self.get_bg_trace(true_bg_prediction)

        # Store the value
        self.sensor_bg_history.append(self.time, self.current_sensor_bg)

    def get_loop_inputs(self):
        return self.sensor_config.sensor_bg_history.get_loop_inputs()

    def set_random_values(self):
        return


class NoisySensor(Sensor):
    """
    A simple sensor with Gaussian noise, spurious events, and missing data.
    """
    def __init__(self, time, sensor_config, random_state):
        super().__init__(time, sensor_config)

        self.name = "BasicNoisySensor"
        self.random_state = random_state

        self.std_dev = 5.0
        if hasattr(sensor_config, "std_dev"):
            self.std_dev = sensor_config.std_dev

        self.spurious_prob = 0.0
        if hasattr(sensor_config, "spurious_prob"):
            self.spurious_prob = sensor_config.spurious_prob

        self.spurious_outage_prob = 0.0
        if hasattr(sensor_config, "spurious_outage_prob"):
            self.spurious_outage_prob = sensor_config.spurious_outage_prob

        self.not_working_timer_minutes_remaining = 0.0

    def is_sensor_working(self):

        is_working = True
        if self.not_working_timer_minutes_remaining > 0.0:
            is_working = False

        return is_working

    def get_bg(self, true_bg):
        """
        Get noisy according to internal params
        """
        u1 = self.random_values["uniform"][0]

        if not self.is_sensor_working():
            bg = None
            self.not_working_timer_minutes_remaining = max(0, self.not_working_timer_minutes_remaining - 5.0)
        elif u1 < self.spurious_prob:
            bg_spurious_error_delta = self.random_values["bg_spurious_error_delta"]  # always positive
            bg = int(true_bg + bg_spurious_error_delta)
            u2 = self.random_values["uniform"][1]
            if u2 < self.spurious_outage_prob:
                self.not_working_timer_minutes_remaining = self.random_values["not_working_time_min"]
        else:
            bg_normal_error_delta = self.random_values["bg_normal_error_delta"]  # pos or neg
            bg = int(true_bg + bg_normal_error_delta)

        return bg

    def get_bg_trace(self, true_bg_trace):
        icgm_trace = []
        for tbg in true_bg_trace:
            icgm_bg = self.get_bg(tbg)
            icgm_trace.append(icgm_bg)
        return icgm_trace

    def get_info_stateless(self):
        stateless_info = super().get_info_stateless()
        stateless_info.update({
            "standard_deviation": self.std_dev
        })
        return stateless_info

    def set_random_values(self):

        self.random_values = {
            "uniform": self.random_state.uniform(0, 1, 100),
            "bg_normal_error_delta": self.random_state.normal(0, self.std_dev),
            "bg_spurious_error_delta": self.random_state.uniform(20, 100),
            "not_working_time_min": self.random_state.uniform(10, 45),
            "cgm_offset_minutes": self.random_state.uniform(2, 4.99)
        }

    def update(self, time, **kwargs):
        """
        Get the current sensed bg and store.

        Parameters
        ----------
        time: datetime
        """
        self.time = time
        self.set_random_values()

        true_bg = kwargs["patient_true_bg"]
        true_bg_prediction = kwargs["patient_true_bg_prediction"]
        self.current_sensor_bg = self.get_bg(true_bg)
        self.current_sensor_bg_prediction = self.get_bg_trace(true_bg_prediction)

        # Store the value
        bg_time = copy.deepcopy(time)

        # Noisy Shift in Time for BG
        u = self.random_values["uniform"][2]
        if u < self.sensor_config.time_delta_crunch_prob:
            offset_minutes = self.random_values["cgm_offset_minutes"]
            bg_time = bg_time - dt.timedelta(minutes=offset_minutes)

        self.sensor_bg_history.append(bg_time, self.current_sensor_bg)


class IdealSensor(Sensor):
    """
    Sensor that reads bg perfectly.
    """
    def __init__(self, time, sensor_config):
        super().__init__(time, sensor_config)
        self.name = "IdealSensor"

    def get_bg(self, true_bg):
        return true_bg

    def get_bg_trace(self, true_bg_trace):
        return true_bg_trace


class SensorState(object):

    def __init__(
            self,
            sensor_bg,
            sensor_bg_prediction
    ):

        self.sensor_bg = sensor_bg
        self.sensor_bg_prediction = sensor_bg_prediction

