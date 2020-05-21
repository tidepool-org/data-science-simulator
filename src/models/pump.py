__author__ = "Cameron Summers"

import copy

from src.models.simulation import SimulationComponent
from src.models.measures import TempBasal, BasalRate

from src.makedata.scenario_parser import PumpConfig


# ======= Pump stuff ==========
class Pump(SimulationComponent):
    """
    Base class for pumps
    """

    def is_valid_temp_basal(self, value, duration_minutes):
        raise NotImplementedError

    def deliver_insulin(self, bolus):
        """
        Behavior for how the pump delivers insulin. Can be used to
        model poor absorption or old insulin, for example. Default
        behavior here in base class is all insulin is delivered as
        prescribed to the pump, ie returns the same bolus object.

        Parameters
        ----------
        bolus: Bolus
            The bolus intended to be given, ie communicated to the pump.

        Returns
        -------
        Bolus
            The bolus that was actually given.
        """
        return bolus


class PumpState(object):
    """
    A class to house the state information for a pump
    """

    def __init__(self, scheduled_basal_rate, temp_basal_rate):

        self.scheduled_basal_rate = scheduled_basal_rate
        self.temp_basal_rate = temp_basal_rate

    def get_basal_rate(self):
        """
        Get the active basal rate, either scheduled or temporary

        Returns
        -------
        BasalRate
            The active basal rate
        """

        if self.temp_basal_rate is not None:
            basal_rate = self.temp_basal_rate
        else:
            basal_rate = self.scheduled_basal_rate

        return basal_rate

    def get_temp_basal_rate_value(self, default=None):
        """
        Get the value of the temp basal and return None if one is not set.

        Returns
        -------
        float
        """
        value = default
        if self.temp_basal_rate is not None:
            value = self.temp_basal_rate.value

        return value


class Omnipod(Pump):
    """
    Omnipod pump class
    """
    def __init__(self, pump_config, time):
        """
        Parameters
        ----------
        pump_config: PumpConfig
            Configuration for the pump
        time: datetime
            t=0
        """

        self.name = "Omnipod"
        self.time = time

        self.pump_config = copy.deepcopy(pump_config)

        self.min_temp_basal = 0.05
        self.min_temp_basal_units = "mg/dL"

        self.active_temp_basal = None
        self.temp_basal_duration = 30

    def is_valid_temp_basal(self, value, duration_minutes):
        """
        Pump specific temp basal validation.
        """
        request_valid = True
        if value >= self.pump_config.max_temp_basal:
            request_valid = False

        return request_valid

    def set_temp_basal(self, value, units):
        """
        Set a temp basal
        """
        if self.is_valid_temp_basal(value, self.temp_basal_duration):
            self.active_temp_basal = TempBasal(
                self.time, value, self.temp_basal_duration, units
            )
        else:
            raise ValueError("Temp basal request is invalid")

    def has_active_temp_basal(self):
        """
        Check if the temp basal is active

        Returns
        -------
        bool
            True if active
        """
        return self.active_temp_basal is not None

    def get_state(self):
        """
        Get the state of the scheduled and temporary basal rates. Temp basal should be None
        if not active.

        Returns
        -------
        PumpState
            The pump state
        """

        temp_basal_rate = self.active_temp_basal
        scheduled_basal_rate = self.pump_config.basal_schedule.get_state()

        return PumpState(scheduled_basal_rate, temp_basal_rate)

    def update(self, time, **kwargs):
        """
        Update the state of the pump for the time.

        Parameters
        ----------
        time: datetime
            The current time
        """
        self.time = time

        if self.active_temp_basal is not None:  # Temp basal current active

            if not self.active_temp_basal.is_active(self.time):  # Remove if inactive
                self.active_temp_basal = None
