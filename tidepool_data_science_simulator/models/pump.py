__author__ = "Cameron Summers"

import copy

from tidepool_data_science_simulator.models.simulation import SimulationComponent
from tidepool_data_science_simulator.models.measures import TempBasal, BasalRate

from tidepool_data_science_simulator.makedata.scenario_parser import PumpConfig


class ContinuousInsulinPump(SimulationComponent):
    """
    A theoretical pump that operates with continuous insulin delivery. This is
     the pump used in the original FDA risk analysis.
    """
    def __init__(self, pump_config, time):
        super().__init__()

        self.name = "ContinuousInsulinPump"
        self.time = time
        self.pump_config = copy.deepcopy(pump_config)

        self.bolus_event_timeline = self.pump_config.bolus_event_timeline
        self.carb_event_timeline = self.pump_config.carb_event_timeline

        self.active_temp_basal = None
        self.basal_insulin_delivered_last_update = 0
        self.basal_undelivered_insulin_since_last_update = 0

    def init(self):
        """
        Initialize the pump for t0
        """
        self.basal_insulin_delivered_last_update = self.get_delivered_basal_insulin_since_update()

    def set_temp_basal(self, temp_basal):
        """
        Set a temp basal
        """
        is_valid, message = self.is_valid_temp_basal(temp_basal)
        if is_valid:
            self.active_temp_basal = temp_basal
        else:
            raise ValueError("Temp basal request is invalid. {}".format(message))

    def get_delivered_basal_insulin_since_update(self, update_interval_minutes=5):
        """
        Get the insulin delivered since the last update based on continuous
        insulin delivery. There is no state change in this function.

        Parameters
        ----------
        update_interval_minutes: int
            Minutes since the last update

        Returns
        -------
        float
            The amount of insulin in units delivered since last update
        """

        insulin_in_hour = self.get_basal_rate().value
        return update_interval_minutes / 60 * insulin_in_hour

    def is_valid_temp_basal(self, temp_basal):
        request_valid = True
        message = ""

        if temp_basal.value >= self.pump_config.max_temp_basal:
            request_valid = False
            message = "Temp basal value is above the maximum allowed on pump."

        if temp_basal.duration_minutes != 30:
            request_valid = False
            message = "Temp basals must be 30 minutes in duration."

        if temp_basal.value < 0:
            request_valid = False
            message = "Invalid temp basal value."

        if temp_basal.start_time != self.time:
            request_valid = False
            message = "Can only set temp basal for current time."

        return request_valid, message

    def deliver_bolus(self, bolus):
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

    def deliver_basal(self, basal_amount):
        """
        Behavior for how the pump delivers basal insulin. Can be used to
        model poor absorption or old insulin, for example. Default
        behavior here in base class is all insulin is delivered as
        prescribed to the pump.

        Parameters
        ----------
        basal_amount
            Amount of basal to deliver

        Returns
        -------
        float
            Amount of basal actually delivered
        """
        return basal_amount

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

        return PumpState(
            scheduled_basal_rate,
            temp_basal_rate,
            self.basal_insulin_delivered_last_update,
            self.basal_undelivered_insulin_since_last_update
        )

    def get_basal_rate(self):
        """
        Get the current basal rate.

        Returns
        -------
        BasalRate
            The current basal rate
        """

        basal_rate = self.pump_config.basal_schedule.get_state()

        if self.has_active_temp_basal():
            basal_rate = self.active_temp_basal

        return basal_rate

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
                self.deactivate_temp_basal()

        self.basal_insulin_delivered_last_update = self.get_delivered_basal_insulin_since_update()

    def deactivate_temp_basal(self):
        """
        Deactivate current temp basal.
        """
        self.active_temp_basal = None

    def get_scheduled_basal_rate(self):
        """
        Get the scheduled basal rate regardless of if a temp basal is set.

        Returns
        -------
        BasalRate
        """
        return self.pump_config.basal_schedule.get_state()


class Omnipod(ContinuousInsulinPump):
    """
    Omnipod pump class that models insulin delivery in pulses.
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
        super().__init__(pump_config, time)

        self.name = "Omnipod"

        self.current_cummulative_pulses = 0
        self.insulin_units_per_pulse = 0.05

    def get_pulses_per_hour(self):
        """
        Get the number of pulses in an hour for the current basal rate.

        Returns
        -------
        int
            Number of pulses
        """

        return int(round(self.get_basal_rate().value / self.insulin_units_per_pulse))

    def get_delivered_basal_insulin_since_update(self, update_interval_minutes=5):
        """
        Get the insulin delivered since the last update based on Omnipod behavior
        of delivering pulses at the last second between pulse intervals. Also updates
        the pulse state.

        Parameters
        ----------
        update_interval_minutes: int
            Minutes since the last update

        Returns
        -------
        float
            The amount of insulin in units delivered since last update
        """

        num_pulses_per_hour = self.get_pulses_per_hour()

        # Get the fractional pulses delivered since the update and add to existing
        # fractional pulses
        fractional_pulses_in_interval = num_pulses_per_hour * (update_interval_minutes / 60.0)
        self.current_cummulative_pulses += fractional_pulses_in_interval

        # Assume all whole pulses were delivered and keep the remaining
        # fractional pulses for the next call
        num_pulses_delivered = int(self.current_cummulative_pulses)
        self.current_cummulative_pulses -= num_pulses_delivered

        return num_pulses_delivered * self.insulin_units_per_pulse


class OmnipodMissingPulses(Omnipod):
    """
    Omnipod pump class that models the missing pulse issue. When a
     temp basal is set, the fractional pulses accumulated until that point
     are "forgotten".
    """
    def __init__(self, pump_config, time):
        super().__init__(pump_config, time)

        self.name = "OmnipodMissingPulses"

    def set_temp_basal(self, temp_basal):
        """
        Set a temp basal and "forget" any existing fractional pulses.
        """
        super().set_temp_basal(temp_basal)

        # The Omnipod gives the insulin at the last possible moment between pulses.
        # Code below models a current known issue that when a temp basal
        # is set before a pulse is to be delivered where any fractional
        # pulses remaining are "forgotten" by the pump.
        pulses_delivered = int(self.current_cummulative_pulses)
        fractional_pulses_remaining = self.current_cummulative_pulses - pulses_delivered
        self.basal_undelivered_insulin_since_last_update = fractional_pulses_remaining * self.insulin_units_per_pulse
        self.current_cummulative_pulses = pulses_delivered


class PumpState(object):
    """
    A class to house the state information for a pump
    """

    def __init__(
            self,
            scheduled_basal_rate,
            temp_basal_rate,
            delivered_basal_insulin,
            undelivered_basal_insulin=0
    ):

        self.scheduled_basal_rate = scheduled_basal_rate
        self.temp_basal_rate = temp_basal_rate
        self.delivered_basal_insulin = delivered_basal_insulin
        self.undelivered_basal_insulin = undelivered_basal_insulin

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
