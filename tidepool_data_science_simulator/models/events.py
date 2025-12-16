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
    """
    Timeline for carb events with version tracking support.
    
    This class extends EventTimeline to support Loop's version-based supercession
    model for carb editing. It maintains:
    - Active events (events dict): Current/active carb entries keyed by start_time
    - All versions (all_carb_versions): Complete history of all versions for auditing
    
    The filtering logic considers both entry_time (when Loop learns of the carb)
    and superceded_date (when an edit replaces a previous version).
    """

    def __init__(self, datetimes=None, events=None):
        super().__init__(datetimes, events)
        self.event_type = Carb
        
        # Store all versions of all carb entries for auditing
        # Key: sync_identifier, Value: list of Carb objects (all versions)
        self.all_carb_versions = {}
        
        # Track which version is active at each start_time
        # Key: start_time, Value: sync_identifier of active carb
        self.active_carb_at_time = {}

    def add_event(self, time, event, input_time=None):
        """
        Add a carb event to the timeline.
        
        This overrides the parent method to also track versions for auditing.
        
        Parameters
        ----------
        time : datetime
            The consumption time (start_time) of the carb
        event : Carb
            The carb event
        input_time : datetime, optional
            When the entry was made in Loop (defaults to time)
        """
        super().add_event(time, event, input_time)
        
        # Track this version for auditing
        if event.sync_identifier:
            if event.sync_identifier not in self.all_carb_versions:
                self.all_carb_versions[event.sync_identifier] = []
            self.all_carb_versions[event.sync_identifier].append(event)
            
            # Track active carb at this start_time
            if event.is_active():
                self.active_carb_at_time[time] = event.sync_identifier

    def add_carb_edit(self, original_start_time, edited_carb, new_start_time=None):
        """
        Add an edited version of an existing carb entry.
        
        This method:
        1. Marks the previous version as superceded
        2. Adds the new version to the timeline
        3. Updates the active carb tracking
        
        Parameters
        ----------
        original_start_time : datetime
            The start_time of the original carb entry
        edited_carb : Carb
            The new version of the carb entry
        new_start_time : datetime, optional
            New consumption time if changed. Defaults to original_start_time.
        """
        if new_start_time is None:
            new_start_time = original_start_time
        
        # Find and mark the previous active version as superceded
        if edited_carb.sync_identifier in self.all_carb_versions:
            for prev_carb in self.all_carb_versions[edited_carb.sync_identifier]:
                if prev_carb.is_active() and prev_carb.sync_version == edited_carb.sync_version - 1:
                    prev_carb.superceded_date = edited_carb.entry_time
                    break
        
        # If the start_time changed, remove from old location in events dict
        if new_start_time != original_start_time and original_start_time in self.events:
            # Only remove if this was the active carb at that time
            if self.active_carb_at_time.get(original_start_time) == edited_carb.sync_identifier:
                del self.events[original_start_time]
                del self.active_carb_at_time[original_start_time]
        
        # Add the new version
        self.add_event(new_start_time, edited_carb, input_time=edited_carb.entry_time)

    def get_active_carb_at_query_time(self, start_time, query_time):
        """
        Get the active version of a carb entry at a specific query time.
        
        This considers the edit timeline - an earlier version may be active
        if the edit hasn't happened yet at query_time.
        
        Parameters
        ----------
        start_time : datetime
            The consumption time of the carb
        query_time : datetime
            The time to evaluate which version is active
            
        Returns
        -------
        Carb or None
            The active carb version at query_time, or None if not visible
        """
        sync_id = self.active_carb_at_time.get(start_time)
        if not sync_id or sync_id not in self.all_carb_versions:
            # Fall back to checking events dict
            carb = self.events.get(start_time)
            if carb and carb.is_visible_at_time(query_time):
                return carb
            return None
        
        # Find the version that would be active at query_time
        versions = sorted(self.all_carb_versions[sync_id], 
                         key=lambda c: c.sync_version, reverse=True)
        
        for carb in versions:
            if carb.is_visible_at_time(query_time):
                return carb
        
        return None

    def get_recent_event_times(self, time=None, num_hours_history=6):
        """
        Get event times within the specified history window, considering edit timing.
        
        This overrides the parent method to properly handle versioned carbs.
        A carb is included if:
        - It's within the history window
        - Its entry_time has passed (Loop knows about it)
        - It hasn't been superceded yet at query_time
        - It hasn't been deleted
        
        Parameters
        ----------
        time : datetime
            Current time to query from
        num_hours_history : int
            Hours of history to include
            
        Returns
        -------
        list
            Times of recent active events
        """
        recent_event_times = []
        
        for event_time in self.events.keys():
            time_since_event_hrs = (time - event_time).total_seconds() / 3600
            
            if time_since_event_hrs > num_hours_history:
                continue
            
            # Get the active version at query time
            carb = self.get_active_carb_at_query_time(event_time, time)
            
            if carb is not None:
                # Check entry time
                event_input_time = self.events_input.get(carb, event_time)
                if event_input_time <= time:
                    recent_event_times.append(event_time)
        
        return recent_event_times

    def get_loop_inputs(self, time, num_hours_history=6):
        """
        Convert event timeline into format for input into Pyloopkit.
        
        This properly handles versioned carbs by returning only the active
        version at the query time.

        Returns
        -------
        (list, list, list)
            carb_values, carb_start_times, carb_durations
        """
        carb_values = []
        carb_start_times = []
        carb_durations = []

        recent_event_times = self.get_recent_event_times(time, num_hours_history=num_hours_history)
        sorted_recent_event_times = sorted(recent_event_times)

        for event_time in sorted_recent_event_times:
            carb_event = self.get_active_carb_at_query_time(event_time, time)
            if carb_event:
                carb_values.append(carb_event.value)
                carb_start_times.append(event_time)
                carb_durations.append(carb_event.duration_minutes)

        return carb_values, carb_start_times, carb_durations

    def get_all_versions(self, sync_identifier=None):
        """
        Get all versions of carb entries for auditing.
        
        Parameters
        ----------
        sync_identifier : str, optional
            If provided, return only versions for this entry.
            If None, return all versions of all entries.
            
        Returns
        -------
        dict or list
            If sync_identifier is None: dict mapping sync_identifier to list of versions
            If sync_identifier provided: list of versions for that entry
        """
        if sync_identifier:
            return self.all_carb_versions.get(sync_identifier, [])
        return self.all_carb_versions

    def get_all_carb_events_for_results(self, time):
        """
        Get all carb events (including superceded versions) for results logging.
        
        This returns ALL versions that were ever active at or before query_time,
        for audit trail purposes.
        
        Parameters
        ----------
        time : datetime
            The time to query
            
        Returns
        -------
        list
            List of (start_time, Carb) tuples for all relevant versions
        """
        results = []
        
        # Get all versions from all_carb_versions
        for sync_id, versions in self.all_carb_versions.items():
            for carb in versions:
                # Include if entry_time has passed
                if carb.entry_time and carb.entry_time <= time:
                    # Find the start_time for this carb
                    for start_time, event in self.events.items():
                        if event.sync_identifier == sync_id:
                            results.append((start_time, carb))
                            break
        
        # Also include carbs not tracked in all_carb_versions (backward compatibility)
        for start_time, carb in self.events.items():
            if not carb.sync_identifier:
                event_input_time = self.events_input.get(carb, start_time)
                if event_input_time <= time:
                    results.append((start_time, carb))
        
        return results

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