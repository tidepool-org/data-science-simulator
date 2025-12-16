__author__ = "Cameron Summers"

"""
Classes structures for various types of data used for simulation.
"""

import copy
import datetime

import numpy as np


class Measure(object):
    """
    Base class for values that have units.
    """

    def __init__(self, value, units):

        self.value = value
        self.units = units

    def __repr__(self):

        return "{} {}".format(self.value, self.units)

    def __add__(self, other):

        if self.units == other.units:
            return Measure(self.value + other.value, self.units)
        else:
            raise ValueError("Cannot add measures of different units.")

    def __eq__(self, other):

        return self.value == other.value and self.units == other.units

    def __hash__(self):
        return hash((self.value, self.units))

    def get_value(self):
        return self.value

    def get_units(self):
        return self.units


class MeasureRange(object):
    """
    Base class for values that have minimums and maximums
    """

    def __init__(self, min_value, max_value, units):
        self.min_value = min_value
        self.max_value = max_value
        self.units = units

    def get_value(self):

        return self.min_value, self.max_value


class BasalRate(Measure):
    """
    A rate of insulin delivered in even pulses over a time period.
    """

    def __init__(self, value, units):
        super().__init__(value, units)

    def get_bolus_schedule(self, start_time, end_time):
        """
        Get a list of times and boluses that would actualize the basal rate.
        """
        raise NotImplementedError

    def get_insulin_in_interval(self, minutes_delta=5):

        # TODO: make this configurable?
        divisor = (
            60 / minutes_delta
        )  # assumes units are U/hr => 12 pulse/hr 60 min/hr / 5 min/pulse
        return self.value / divisor


class TempBasal(BasalRate):
    """
    A basal rate that expires after a duration.
    """

    def __init__(self, time, value, duration_minutes, units):
        super().__init__(value, units)

        self.start_time = copy.deepcopy(time)
        self.scheduled_duration_minutes = duration_minutes
        self.scheduled_end_time = self.start_time + datetime.timedelta(minutes=duration_minutes)
        self.actual_end_time = None
        self.actual_duration_minutes = 0

        self.active = True
        self.delivered_units = 0

    def __str__(self):
        this_str = "None"
        if self.active:
            this_str = "{} {}".format(self.value, self.scheduled_duration_minutes)

        return this_str

    def __repr__(self):
        return "{} {}min".format(super().__repr__(), self.scheduled_duration_minutes)

    def __eq__(self, other):

        return  self.start_time == other.start_time and \
                self.value == other.value and \
                self.units == other.units and \
                self.scheduled_duration_minutes == other.scheduled_duration_minutes

    def __hash__(self):
        return hash((self.value, self.units, self.scheduled_duration_minutes))

    def get_end_time(self):
        """
        Return the expected end time unless the temp basal was cut short, then
        return the actual end time.

        Returns
        -------
        datetime.datetime
        """

        end_time = self.scheduled_end_time
        if self.actual_end_time is not None:
            end_time = self.actual_end_time

        return end_time

    def get_minutes_remaining(self, time):
        time_elapsed = time - self.start_time
        minutes_elapsed = time_elapsed.total_seconds() / 60.0
        minutes_remaining = self.scheduled_duration_minutes - minutes_elapsed
        return minutes_remaining

    def is_active(self, time):
        """
        Determine if the temp basal is active at given time.

        Parameters
        ----------
        time: datetime
            The current time

        Returns
        -------
        bool
            If the temp basal is active
        """
        minutes_passed = (time - self.start_time).total_seconds() / 60.0

        if minutes_passed >= self.scheduled_duration_minutes:
            self.active = False

        return self.active


class Bolus(Measure):
    """
    A bolus delivered by a pump
    """
    def __init__(self, value, units):
        super().__init__(value, units)


class ManualBolus(Bolus):
    """
    A Bolus that is delivered manually, e.g. via injection
    """
    def __init__(self, value, units):
        super().__init__(value, units)

class HeartRate(Measure):
    """
    Heart Rate
    """
    def __init__(self, value, units):
        super().__init__(value, units)


# Carb operation types (mirrors Loop's Operation enum)
class CarbOperation:
    """Operation types for carb entries, mirroring Loop's Operation enum."""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"


class Carb(Measure):
    """
    A carb with an expected absorption duration, entry time tracking, and version control.
    
    This class models carbohydrate intake for diabetes simulation, supporting:
    1. Separation between when a carb entry is made in Loop (entry_time) and when
       the carbs are actually consumed (start_time, tracked via the event timeline)
    2. Version-based edit tracking mirroring Loop's supercession model
    
    Attributes
    ----------
    value : float
        Amount of carbohydrates in grams
    units : str
        Unit of measurement (typically "g")
    duration_minutes : int
        Expected absorption duration in minutes
    entry_time : datetime, optional
        When the user entered this carb in Loop (mirrors Loop's date/userCreatedDate)
    sync_identifier : str, optional
        Unique ID linking all versions of the same logical carb entry
    sync_version : int
        Version number (0 = original, incremented on each edit)
    user_created_date : datetime, optional
        When the entry was originally created (preserved across edits)
    user_updated_date : datetime, optional
        When the entry was last edited (None if never edited)
    superceded_date : datetime, optional
        When this entry was replaced by a newer version (None if active)
    operation : str
        One of: 'create', 'update', 'delete' (from CarbOperation)
    
    Notes
    -----
    Loop maintains two distinct timestamps for carb entries:
    - startDate: When carbs are/were consumed (physiological time)
    - date/userCreatedDate: When the entry was created in the app
    
    Loop uses a version-based supercession model for editing:
    - syncIdentifier links all versions of the same entry
    - syncVersion is incremented on each edit
    - supercededDate marks when an entry was replaced
    - Only entries with supercededDate=None and operation!='delete' are active
    """

    def __init__(self, value, units, duration_minutes, entry_time=None,
                 sync_identifier=None, sync_version=0, user_created_date=None,
                 user_updated_date=None, superceded_date=None, operation=CarbOperation.CREATE):
        """
        Initialize a Carb object.
        
        Parameters
        ----------
        value : float
            Amount of carbohydrates
        units : str
            Unit of measurement (e.g., "g" for grams)
        duration_minutes : int or float
            Expected absorption duration in minutes
        entry_time : datetime, optional
            When the carb entry was made in Loop. If None, entry time is assumed
            to be the same as consumption time (tracked in the event timeline).
        sync_identifier : str, optional
            Unique ID for this carb entry. If None, one will be generated.
        sync_version : int, optional
            Version number (default 0 for original entry)
        user_created_date : datetime, optional
            When entry was originally created. If None, uses entry_time.
        user_updated_date : datetime, optional
            When entry was last edited. None for original entries.
        superceded_date : datetime, optional
            When this version was replaced. None if this is the active version.
        operation : str, optional
            Operation type: 'create', 'update', or 'delete' (default 'create')
        """
        super().__init__(value, units)
        self.duration_minutes = int(duration_minutes)
        self.entry_time = entry_time
        
        # Version control attributes (mirrors Loop's CarbStore model)
        self.sync_identifier = sync_identifier
        self.sync_version = sync_version
        self.user_created_date = user_created_date if user_created_date else entry_time
        self.user_updated_date = user_updated_date
        self.superceded_date = superceded_date
        self.operation = operation

    def __repr__(self):
        parts = [f"Carb({self.value} {self.units}, duration={self.duration_minutes}min"]
        if self.sync_identifier:
            parts.append(f", id={self.sync_identifier[:8]}...")
        parts.append(f", v{self.sync_version}")
        if self.operation != CarbOperation.CREATE:
            parts.append(f", op={self.operation}")
        if self.superceded_date:
            parts.append(f", superceded")
        parts.append(")")
        return "".join(parts)

    def get_duration(self):
        """
        Get the expected absorption duration.
        
        Returns
        -------
        int
            Absorption duration in minutes
        """
        return self.duration_minutes
    
    def get_entry_time(self):
        """
        Get the time when this carb entry was made in Loop.
        
        Returns
        -------
        datetime or None
            The entry time, or None if not specified (meaning entry time
            equals consumption time)
        """
        return self.entry_time
    
    def has_separate_entry_time(self):
        """
        Check if this carb has a separate entry time from consumption time.
        
        Returns
        -------
        bool
            True if entry_time is explicitly set, False otherwise
        """
        return self.entry_time is not None
    
    # Version control methods
    
    def get_sync_identifier(self):
        """Get the unique identifier linking all versions of this carb entry."""
        return self.sync_identifier
    
    def get_sync_version(self):
        """Get the version number (0 = original, increments on edit)."""
        return self.sync_version
    
    def get_user_created_date(self):
        """Get when this entry was originally created (preserved across edits)."""
        return self.user_created_date
    
    def get_user_updated_date(self):
        """Get when this entry was last edited (None if never edited)."""
        return self.user_updated_date
    
    def get_superceded_date(self):
        """Get when this version was replaced (None if active)."""
        return self.superceded_date
    
    def get_operation(self):
        """Get the operation type: 'create', 'update', or 'delete'."""
        return self.operation
    
    def is_active(self):
        """
        Check if this carb entry version is currently active.
        
        An entry is active if:
        - It has not been superceded (superceded_date is None)
        - It has not been deleted (operation != 'delete')
        
        Returns
        -------
        bool
            True if this is the active version of the carb entry
        """
        return self.superceded_date is None and self.operation != CarbOperation.DELETE
    
    def is_visible_at_time(self, query_time):
        """
        Check if this carb entry is visible to Loop at a given time.
        
        An entry is visible if:
        - The entry_time has passed (Loop knows about it)
        - It hasn't been superceded yet at query_time
        - It hasn't been deleted yet at query_time
        
        Parameters
        ----------
        query_time : datetime
            The time to check visibility
            
        Returns
        -------
        bool
            True if Loop would see this carb entry at query_time
        """
        # Entry must have been made
        if self.entry_time and query_time < self.entry_time:
            return False
        
        # Check if superceded before query_time
        if self.superceded_date and query_time >= self.superceded_date:
            return False
        
        # Check if deleted
        if self.operation == CarbOperation.DELETE:
            return False
        
        return True
    
    def create_edited_version(self, edit_time, new_value=None, new_start_time=None,
                               new_duration=None, operation=CarbOperation.UPDATE):
        """
        Create a new version of this carb entry representing an edit.
        
        This method implements Loop's edit flow:
        1. The new version gets the same sync_identifier
        2. sync_version is incremented
        3. user_created_date is preserved from original
        4. user_updated_date is set to edit_time
        
        Parameters
        ----------
        edit_time : datetime
            When the edit was made
        new_value : float, optional
            New carb value. If None, inherits from this version.
        new_start_time : datetime, optional
            New consumption time. If None, inherits from this version.
        new_duration : int, optional
            New absorption duration. If None, inherits from this version.
        operation : str, optional
            Operation type (default 'update', use 'delete' for deletions)
            
        Returns
        -------
        Carb
            A new Carb object representing the edited version
        """
        return Carb(
            value=new_value if new_value is not None else self.value,
            units=self.units,
            duration_minutes=new_duration if new_duration is not None else self.duration_minutes,
            entry_time=edit_time,
            sync_identifier=self.sync_identifier,
            sync_version=self.sync_version + 1,
            user_created_date=self.user_created_date,
            user_updated_date=edit_time,
            superceded_date=None,  # New version is active
            operation=operation
        )


class CarbInsulinRatio(Measure):
    """
    Carb-Insulin Ratio
    """

    def __init__(self, value, units):
        super().__init__(value, units)

    def calculate_bolus(self, carb):
        """
        Convenience bolus calculator.

        Parameters
        ----------
        carb: Carb
            Carbs to be ingested

        Returns
        -------
        float
            Insulin required for the carbs
        """
        # TODO: do a units check
        return carb.value / self.value


class InsulinSensitivityFactor(Measure):
    """
    Insulin Sensitivity Factor
    """

    def __init__(self, value, units):
        super().__init__(value, units)

class GlucoseSensitivityFactor(Measure):
    """
    Glucose Sensitivity Factor
    """

    def __init__(self, value, units):
        super().__init__(value, units)

class BasalBloodGlucose(Measure):
    """
    Basal Blood Glucose
    """

    def __init__(self, value, units):
        super().__init__(value, units)

class InsulinProductionRate(Measure):
    """
    Insulin Production Rate
    """

    def __init__(self, value, units):
        super().__init__(value, units)

class TargetRange(MeasureRange):
    """
    Target range
    """

    def __init__(self, min_value, max_value, units):
        super().__init__(min_value, max_value, units)


class BloodGlucose(Measure):
    """
    Blood glucose
    """

    def __init__(self, value, units):
        super().__init__(value, units)


class GlucoseTrace(object):
    """
    Basic encapsulation of a trace with associated datetimes.

    TODO: Utilize pandas series more here for time operations
    TODO: make bg an BloodGlucose obj instead of int
    """

    def __init__(self, datetimes=None, values=None):

        self.datetimes = []
        if datetimes is not None:
            self.datetimes = datetimes

        self.bg_values = []
        if values is not None:
            self.bg_values = values

    def __iter__(self):
        for dt, bg_val in zip(self.datetimes, self.bg_values):
            yield dt, bg_val

    def get_last(self):
        """
        Get most recent value.

        Returns
        -------
        (datetime, int)
        """

        return self.datetimes[-1], self.bg_values[-1]

    def append(self, date, bg):
        """
        Add a new value

        Parameters
        ----------
        date: datetime
        bg: int

        Returns
        -------

        """

        self.datetimes.append(date)
        self.bg_values.append(bg)

    def get_loop_inputs(self, time=None, num_hours_history=None):
        """
        Get two numpy arrays for dates and values, used for Loop input.

        Optionally only get values in recent history.
        """
        loop_bg_values = []
        loop_bg_datetimes = []

        if time is not None:
            for dt, bg in zip(self.datetimes, self.bg_values):
                time_since_bg = (time - dt).total_seconds() / 3600.0

                if bg is not None and time_since_bg < num_hours_history:
                    processed_bg = max(40, min(400, float(np.round(bg))))
                    loop_bg_datetimes.append(dt)
                    loop_bg_values.append(processed_bg)
        else:
            loop_bg_values = [max(40, min(400, float(np.round(bg)))) for bg in self.bg_values]
            loop_bg_datetimes = self.datetimes

        return loop_bg_datetimes, loop_bg_values
    
class PhysicalActivity(object):
    """
    Physical activity with an activity name, duration, and optional expected heart rate
    """
    def __init__(self, activity='', duration=0, expected_hr=None):
        """
        Parameters
        ----------
        activity : str
            Type of activity (e.g., "walking", "running", "cycling")
        duration : int
            Duration in minutes
        expected_hr : float, optional
            Expected heart rate during this activity in bpm
            If None, will use a default based on activity type
        """
        self.activity = activity
        self.duration = duration
        self.expected_hr = expected_hr
        # Maintain backward compatibility - 'value' property for activity name
        self.value = activity
        
    def __repr__(self):
        return f"PhysicalActivity({self.activity}, duration={self.duration}min, hr={self.expected_hr}bpm)"
        
class HeartRateTrace(object):
    def __init__(self, datetimes=None, values=None):
        self.datetimes = []
        if datetimes is not None:
            self.datetimes = datetimes
        self.hr_values = []
        if values is not None:
            self.hr_values = values
    def __iter__(self):
        for dt, hr_val in zip(self.datetimes, self.hr_values):
            yield dt, hr_val
    def get_last(self):
        """
        Get most recent value
        Returns
        -------
        (datetime, int)
        """
        return self.datetimes[-1], self.hr_values[-1]
    
    def get_heart_rate(self, dt):
        """
        get heart rate at the given time
        """
        # Defensive check for empty trace
        if len(self.datetimes) == 0 or len(self.hr_values) == 0:
            return 0
        
        idx = np.searchsorted(self.datetimes, dt, side='right')
        
        # Bound check to prevent negative indexing issues
        if idx == 0:
            result_hr = self.hr_values[0]
        else:
            result_hr = self.hr_values[idx - 1]
        
        return result_hr
    def append(self, date, hr):
        """
        Add a new value
        Parameters
        ----------
        date: datetime
        hr: int
        Returns
        -------
        """
        self.datetimes.append(date)
        self.hr_values.append(hr)
