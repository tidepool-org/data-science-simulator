__author__ = "Cameron Summers"

"""
Sensor model classes
"""

import numpy as np

from tidepool_data_science_simulator.models.simulation import SimulationComponent


class Sensor(SimulationComponent):

    def get_state(self):
        """
        Get the state of the sensor
        """
        raise NotImplementedError

    def update(self, time, **kwargs):
        """
        For a stateful sensor, implement this function. E.g. sensor drift in iCGM sensor

        Parameters
        ----------
        time: datetime
        """
        raise NotImplementedError


class NoisySensor(Sensor):
    def __init__(self, sensor_config):
        self.name = "iCGM"
        self.sensor_config = sensor_config

    def get_bg(self, true_bg):
        """
        Get icgm_bg according to internal params
        """
        # Noisy placeholder
        return int(np.random.normal(true_bg, 5.0))

    def get_bg_trace(self, true_bg_trace):
        icgm_trace = []
        for tbg in true_bg_trace:
            icgm_bg = self.get_bg(tbg)
            icgm_trace.append(icgm_bg)
        return icgm_trace

    def update(self, time):
        # No state
        pass


class IdealSensor(Sensor):
    """
    Sensor that reads bg perfectly.
    """

    def __init__(self, sensor_config):

        self.name = "IdealSensor"
        self.sensor_config = sensor_config

    def get_bg(self, true_bg):
        return true_bg

    def get_bg_trace(self, true_bg_trace):
        return true_bg_trace

    def get_state(self):
        pass

    def update(self, time):
        # No state
        pass
