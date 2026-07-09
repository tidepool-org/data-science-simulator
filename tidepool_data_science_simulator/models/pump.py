__author__ = "Cameron Summers"

import copy

from tidepool_data_science_simulator.models.measures import TempBasal, BasalRate
from tidepool_data_science_simulator.models.state import PumpState
from tidepool_data_science_simulator.makedata.scenario_parser import PumpConfig
from tidepool_data_science_simulator.models.simulation import SimulationComponent
from tidepool_data_science_simulator.models.events import TempBasalTimeline


class ContinuousInsulinPump(SimulationComponent):
    """
    A theoretical pump that operates with continuous insulin delivery. This is
     the pump used in the original FDA risk analysis.
    """
    def __init__(self, pump_config, time):
        super().__init__()

        self.name = "ContinuousInsulinPump"
        self.time = time
        self.simulation_start_time = time  # Preserve original start time for historical backfill
        self.pump_config = copy.deepcopy(pump_config)

        self.bolus_event_timeline = self.pump_config.bolus_event_timeline
        self.carb_event_timeline = self.pump_config.carb_event_timeline
        self.temp_basal_event_timeline = TempBasalTimeline()  # Not currently in scenario files

        self.active_temp_basal = None
        self.basal_insulin_delivered_last_update = 0
        self.basal_undelivered_insulin_since_last_update = 0

    @classmethod
    def get_classname(cls):
        return cls.__name__

    def init(self):
        """
        Initialize the pump for t0
        """
        self.basal_insulin_delivered_last_update = self.get_delivered_basal_insulin_since_update()
        
        # Record this initial basal delivery in the timeline
        # This represents the first 5-minute interval at t=0
        self._record_scheduled_basal_delivery(self.time)

    def get_info_stateless(self):

        stateless_info = {
            "name": self.name,
            "config": self.pump_config.get_info_stateless()
        }
        return stateless_info

    def set_temp_basal(self, temp_basal):
        """
        Set a temp basal
        """
        is_valid, message = self.is_valid_temp_basal(temp_basal)
        if is_valid:
            if self.has_active_temp_basal():
                self.deactivate_temp_basal()

            self.active_temp_basal = temp_basal
            self.temp_basal_event_timeline.add_event(self.time, temp_basal)
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

    def _record_scheduled_basal_delivery(self, time):
        """
        Record scheduled basal delivery as a basal event in the timeline.
        
        This creates a "virtual" basal event representing the 5-minute interval
        of scheduled basal delivery that just occurred. These events allow Loop
        to see the complete pump history when it queries for doses.
        
        Parameters
        ----------
        time : datetime
            Current simulation time (end of the delivery interval)
        """
        import datetime
        
        # Calculate the time window for this basal delivery
        # Basal was delivered over the past 5 minutes (update interval)
        update_interval_minutes = 5
        start_time = time - datetime.timedelta(minutes=update_interval_minutes)
        
        # Get the scheduled basal rate that was active during this interval
        basal_rate = self.pump_config.basal_schedule.get_state()
        
        # Create a TempBasal object to represent this scheduled basal delivery
        # We use TempBasal for consistency with the existing dose timeline structure
        scheduled_basal_event = TempBasal(
            time=start_time,
            value=basal_rate.value,
            duration_minutes=update_interval_minutes,
            units="U/hr"
        )
        
        # Mark it as inactive since it's already been delivered (historical)
        scheduled_basal_event.active = False
        scheduled_basal_event.actual_end_time = time
        scheduled_basal_event.actual_duration_minutes = update_interval_minutes
        scheduled_basal_event.delivered_units = self.basal_insulin_delivered_last_update
        
        # Add to timeline with input_time set to current time
        # The input_time parameter ensures this event passes the filter in
        # get_recent_event_times() when Loop queries for doses
        self.temp_basal_event_timeline.add_event(
            start_time, 
            scheduled_basal_event, 
            input_time=time
        )

    def populate_historical_basal_doses(self, current_time, num_hours_history=8):
        """
        Populate the pump's dose timeline with historical scheduled basal deliveries.
        
        This method backfills the pump's timeline with basal doses that were delivered
        during the simulation but before Loop was activated. In real life, Loop reads
        this historical data from the pump's internal storage. This ensures Loop has
        the complete dose history it needs for accurate IOB calculations.
        
        Parameters
        ----------
        current_time : datetime
            The current simulation time (typically when Loop is being activated)
        num_hours_history : float
            Number of hours of historical basal to populate (default: 8, matching
            Loop's typical lookback window)
        """
        import datetime
        
        # Calculate how far back to go
        # We backfill the full num_hours_history regardless of when the pump object
        # was created, because we're reconstructing the basal history that the patient
        # experienced (the patient has IOB from basal delivered before pump/Loop activation)
        history_start = current_time - datetime.timedelta(hours=num_hours_history)
        
        # Generate basal events for each 5-minute interval
        update_interval_minutes = 5
        current_interval_start = history_start
        
        while current_interval_start < current_time:
            interval_end = current_interval_start + datetime.timedelta(minutes=update_interval_minutes)
            
            # Don't go past current time
            if interval_end > current_time:
                interval_end = current_time
            
            # Check if this interval already has a basal event (scheduled or temp)
            # This prevents duplicating events that were already recorded during simulation
            if not self._has_basal_event_at_time(current_interval_start):
                # Get the scheduled basal rate for this time
                # Note: Using current schedule is a simplification. For full accuracy,
                # we could save historical basal rates if they change during simulation
                basal_rate = self.pump_config.basal_schedule.get_state()
                
                # Calculate insulin delivered in this interval
                actual_minutes = (interval_end - current_interval_start).total_seconds() / 60
                insulin_delivered = basal_rate.value * (actual_minutes / 60.0)
                
                # Create historical basal event
                historical_basal_event = TempBasal(
                    time=current_interval_start,
                    value=basal_rate.value,
                    duration_minutes=int(actual_minutes),
                    units="U/hr"
                )
                
                # Mark as already delivered (historical)
                historical_basal_event.active = False
                historical_basal_event.actual_end_time = interval_end
                historical_basal_event.actual_duration_minutes = actual_minutes
                historical_basal_event.delivered_units = insulin_delivered
                
                # Add to timeline with interval_end as input_time
                # This ensures the event is "visible" when Loop queries at current_time
                self.temp_basal_event_timeline.add_event(
                    current_interval_start, 
                    historical_basal_event,
                    input_time=interval_end
                )
            
            # Move to next interval
            current_interval_start = interval_end

    def _has_basal_event_at_time(self, time):
        """
        Check if a basal event (temp or scheduled) already exists at the given time.
        
        Parameters
        ----------
        time : datetime
            The time to check
        
        Returns
        -------
        bool
            True if an event exists at this time, False otherwise
        """
        return time in self.temp_basal_event_timeline.events

    def is_valid_temp_basal(self, temp_basal):
        request_valid = True
        message = ""

        if temp_basal.value >= self.pump_config.max_temp_basal:
            request_valid = False
            message = "Temp basal value is above the maximum allowed on pump."

        if temp_basal.scheduled_duration_minutes != 30:
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
        isf = self.pump_config.insulin_sensitivity_schedule.get_state()
        cir = self.pump_config.carb_ratio_schedule.get_state()

        return PumpState(
            scheduled_basal_rate=scheduled_basal_rate,
            scheduled_cir=cir,
            schedule_isf=isf,
            temp_basal_rate=temp_basal_rate,
            bolus=self.bolus_event_timeline.get_event(self.time),
            carb=self.carb_event_timeline.get_event(self.time),
            delivered_basal_insulin=self.basal_insulin_delivered_last_update,
            undelivered_basal_insulin=self.basal_undelivered_insulin_since_last_update
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

        self.basal_undelivered_insulin_since_last_update = 0.0
        self.basal_insulin_delivered_last_update = self.get_delivered_basal_insulin_since_update()

        if self.active_temp_basal is not None:  # Temp basal current active

            self.active_temp_basal.delivered_units += self.basal_insulin_delivered_last_update

            if not self.active_temp_basal.is_active(self.time):  # Remove if inactive
                self.deactivate_temp_basal()
        
        else:  # No temp basal active - record scheduled basal delivery
            # This ensures scheduled basal deliveries are recorded in the pump's
            # dose timeline, making them visible to Loop when it queries for doses
            self._record_scheduled_basal_delivery(time)

        self.pump_config.basal_schedule.update(time)
        self.pump_config.carb_ratio_schedule.update(time)
        self.pump_config.insulin_sensitivity_schedule.update(time)
        self.pump_config.target_range_schedule.update(time)

    def deactivate_temp_basal(self):
        """
        Deactivate current temp basal.
        """
        self.active_temp_basal.actual_end_time = self.time
        self.active_temp_basal.actual_duration_minutes = (self.time - self.active_temp_basal.start_time).total_seconds() / 60
        self.active_temp_basal.active = False
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

        insulin_delivered = num_pulses_delivered * self.insulin_units_per_pulse

        return insulin_delivered


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

