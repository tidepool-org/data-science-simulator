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


class Carb(Measure):
    """
    A carb with an expected absorption duration.
    """

    def __init__(self, value, units, duration_minutes):
        super().__init__(value, units)

        self.duration_minutes = int(duration_minutes)

    def get_duration(self):
        return self.duration_minutes


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
    Physical activity with an activity name and duration
    """
    def __init__(self, activity='', duration=0):
        self.activity = activity
        self.duration = duration
        
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
        idx = np.searchsorted(self.datetimes, dt, side='right')
        return self.hr_values[idx - 1]
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
        self.bg_values.append(hr)
