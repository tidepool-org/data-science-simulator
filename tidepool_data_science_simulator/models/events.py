__author__ = "Cameron Summers"

import datetime
import numpy as np

from tidepool_data_science_simulator.models.measures import Carb, Bolus, TempBasal, PhysicalActivity
from tidepool_data_science_simulator.utils import get_bernoulli_trial_uniform_step_prob
from pyloopkit.dose import DoseType


class Action(object):
    """
    A class for user executed actions that are not inputs.
    """
    def __init__(self, name):
        self.name = name

    def execute(self, **kwargs):
        raise NotImplementedError


class VirtualPatientDeleteLoopData(Action):
    """
    User deletes their pump bolus and basal data.

    For Risk analysis TLR337 this is used as a proxy for a user deleting their insulin
    data in Loop. Since the simulator has Loop currently read insulin from the pump,
    this achieves the desired result. Eventually we'll want to have separate data stores
    for Loop and pump because this models reality better.
    """
    def execute(self, virtual_patient):
        virtual_patient.pump.bolus_event_timeline = BolusTimeline()
        virtual_patient.pump.temp_basal_event_timeline = TempBasalTimeline()


class VirtualPatientRemovePump(Action):
    """
    Patient ends pump session. No more insulin is delivered or attempted to be delivered.
    """
    def execute(self, virtual_patient):
        virtual_patient.stop_pump_session()


class VirtualPatientAttachPump(Action):
    """
    User begins pump session.
    """
    def __init__(self, name, pump_class, pump_config):
        super().__init__(name)
        self.pump_class = pump_class
        self.pump_config = pump_config

    def execute(self, virtual_patient):
        virtual_patient.start_pump_session(self.pump_class, self.pump_config)


class EventTimeline(object):
    """
    A class for insulin/carb/etc. events
    """
    def __init__(self, datetimes=None, events=None):

        self.events = dict()  # The event time, e.g. bolus at 1pm

        # The user input times, e.g. input at 1:30pm a bolus that occurred at 1pm
        # Used mainly for filtering information passed to Pyloopkit in a realistic way
        self.events_input = dict()

        if datetimes is not None:
            for dt, event in zip(datetimes, events):
                self.events[dt] = event

    def is_empty_timeline(self):
        """
        Determine if there are events in the timeline.

        Returns
        -------
        bool:
            True if no events
        """

        return len(self.events) == 0

    def add_event(self, time, event, input_time=None):
        """
        Add an event to the timeline.

        Parameters
        ----------
        time: datetime
            The time of the event

        event: Bolus, Carb, etc.
            The event

        input_time: datetime
            The time the event was input into the system.
        """

        self.is_event_valid(event)

        self.events[time] = event

        if input_time is None:
            input_time = time
        self.events_input[event] = input_time

    def is_event_valid(self, event):
        return isinstance(event, self.event_type)

    def remove_event(self, time):
        """
        Removes the event at the given time.
        """
        self.events.pop(time)

    def get_event(self, time):
        """
        Get the event at the given time. If no event, returns None

        Parameters
        ----------
        time: datetime
            Time to check for event

        Returns
        -------
        object
            The insulin/carb/etc. event or None
        """
        try:
            event = self.events[time]
        except KeyError:
            event = None

        return event

    def get_recent_event_times(self, time=None, num_hours_history=6):
        """
        Get event times within the specified history window.

        Parameters
        ----------
        time
        num_hours_history

        Returns
        -------
        list
            Times of recent events
        """
        recent_event_times = []
        for event_time in self.events.keys():
            time_since_event_hrs = (time - event_time).total_seconds() / 3600
            event_input_time = self.events_input.get(self.events[event_time], event_time)
            if time_since_event_hrs <= num_hours_history and event_input_time <= time:
                recent_event_times.append(event_time)

        return recent_event_times

    def merge_timeline(self, event_timeline):
        """
        Merge events from another timeline.

        Parameters
        ----------
        event_timeline: EventTimeline
        """
        self.events.update(event_timeline.events)


class BolusTimeline(EventTimeline):

    def __init__(self, datetimes=None, events=None):
        super().__init__(datetimes, events)
        self.event_type = Bolus

    def get_loop_inputs(self, time, num_hours_history=6):
        """
        Convert event timeline into format for input into Pyloopkit.

        Returns
        -------
        (list, list, list, list)
        """
        dose_types = []
        dose_values = []
        dose_start_times = []
        dose_end_times = []
        dose_delivered_units = []

        recent_event_times = self.get_recent_event_times(time, num_hours_history=num_hours_history)
        sorted_trecent_event_times = sorted(recent_event_times)  # TODO: too slow?
        for time in sorted_trecent_event_times:

            dose_types.append(DoseType.bolus)
            dose_values.append(self.events[time].value)
            dose_start_times.append(time)
            dose_end_times.append(time)
            dose_delivered_units.append(self.events[time].value)  # fixme: shouldn't be same value

        return dose_types, dose_values, dose_start_times, dose_end_times, dose_delivered_units


class TempBasalTimeline(EventTimeline):

    def __init__(self, datetimes=None, events=None):
        super().__init__(datetimes, events)
        self.event_type = TempBasal

    def get_loop_inputs(self, time, num_hours_history=6):
        """
        Convert event timeline into format for input into Pyloopkit.

        Returns
        -------
        (list, list, list, list)
        """

        dose_types = []
        dose_values = []
        dose_start_times = []
        dose_end_times = []
        dose_delivered_units = []

        recent_event_times = self.get_recent_event_times(time, num_hours_history=num_hours_history)
        sorted_trecent_event_times = sorted(recent_event_times)  # TODO: too slow?
        for event_time in sorted_trecent_event_times:
            temp_basal_event = self.events[event_time]
            dose_types.append(DoseType.tempbasal)
            dose_values.append(temp_basal_event.value)
            dose_start_times.append(event_time)

            end_time = temp_basal_event.get_end_time()
            if temp_basal_event.is_active(time):
                end_time = time  # Pyloopkit does not expect doses past the current time.
            dose_end_times.append(end_time)

            dose_delivered_units.append(temp_basal_event.delivered_units)  # fixme: put actual values here

        return dose_types, dose_values, dose_start_times, dose_end_times, dose_delivered_units


class CarbTimeline(EventTimeline):

    def __init__(self, datetimes=None, events=None):
        super().__init__(datetimes, events)
        self.event_type = Carb

    def get_loop_inputs(self, time, num_hours_history=6):
        """
        Convert event timeline into format for input into Pyloopkit.

        Returns
        -------
        (list, list, list, list)
        """

        carb_values = []
        carb_start_times = []
        carb_durations = []

        recent_event_times = self.get_recent_event_times(time, num_hours_history=num_hours_history)
        sorted_recent_event_times = sorted(recent_event_times)  # TODO: too slow?

        for time in sorted_recent_event_times:
            carb_event = self.events[time]
            carb_values.append(carb_event.value)
            carb_start_times.append(time)
            carb_durations.append(carb_event.duration_minutes)

        return carb_values, carb_start_times, carb_durations

class PhysicalActivityTimeline(EventTimeline):
    def __init__(self, datetimes=None, events=None):
        super().__init__(datetimes, events)
        self.event_type = PhysicalActivity

class ActionTimeline(EventTimeline):
    def __init__(self, datetimes=None, events=None):
        super().__init__(datetimes, events)
        self.event_type = Action


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

    def __repr__(self):

        return "{}".format(self.name)