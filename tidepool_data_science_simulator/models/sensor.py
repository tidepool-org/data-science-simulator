__author__ = "Cameron Summers"

"""
Sensor model classes
"""

import numpy as np

from tidepool_data_science_simulator.models.simulation import SimulationComponent
from tidepool_data_science_simulator.models.measures import GlucoseTrace

from tidepool_data_science_models.models.icgm_sensor import iCGMSensor


class RealSensor(SimulationComponent):

    def __init__(self, time, sensor_config):

        super().__init__()
        self.time = time
        self.sensor_bg_history = sensor_config.sensor_bg_history
        self.sensor_config = sensor_config

    def get_state(self):
        return SensorState(
            self.sensor_bg_history.get_bg_at_time(self.time),
            None
        )

    def update(self, time, **kwargs):
        """
        Get the current sensed bg and store.

        Parameters
        ----------
        time: datetime
        """
        self.time = time

    def get_loop_inputs(self):
        return self.sensor_config.sensor_bg_history.get_loop_inputs()


class Sensor(SimulationComponent):

    def __init__(self, time, sensor_config):

        super().__init__()
        self.time = time
        self.sensor_bg_history = sensor_config.sensor_bg_history
        self.sensor_config = sensor_config

        self.current_sensor_bg = self.sensor_config.sensor_bg_history.bg_values[0]
        self.current_sensor_bg_prediction = None

    def get_bg(self, true_bg):
        raise NotImplementedError

    def get_bg_trace(self, true_bg_trace):
        raise NotImplementedError

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
        true_bg = kwargs["patient_true_bg"]
        true_bg_prediction = kwargs["patient_true_bg_prediction"]
        self.current_sensor_bg = self.get_bg(true_bg)
        self.current_sensor_bg_prediction = self.get_bg_trace(true_bg_prediction)

        # Store the value
        self.sensor_bg_history.append(self.time, self.current_sensor_bg)

    def get_loop_inputs(self):
        return self.sensor_config.sensor_bg_history.get_loop_inputs()


class NoisySensor(Sensor):
    """
    A simple sensor with Gaussian noise.
    """
    def __init__(self, time, sensor_config):
        super().__init__(time, sensor_config)

        self.name = "iCGM"
        try:
            self.std_dev = sensor_config.std_dev
        except:
            self.std_dev = 5.0  # todo: hack for now

    def get_bg(self, true_bg):
        """
        Get noisy according to internal params
        """
        bg = int(np.random.normal(true_bg, self.std_dev))
        return bg

    def get_bg_trace(self, true_bg_trace):
        icgm_trace = []
        for tbg in true_bg_trace:
            icgm_bg = self.get_bg(tbg)
            icgm_trace.append(icgm_bg)
        return icgm_trace


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

