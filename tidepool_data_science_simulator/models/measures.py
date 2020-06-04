__author__ = "Cameron Summers"

"""
Classes structures for various types of data used for simulation.
"""

import copy


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


class MeasureRange(object):
    """
    Base class for values that have minimums and maximums
    """

    def __init__(self, min_value, max_value, units):
        self.min_value = min_value
        self.max_value = max_value
        self.units = units


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
        self.duration_minutes = duration_minutes
        self.active = True

    def __str__(self):
        this_str = "None"
        if self.active:
            this_str = "{} {}".format(self.value, self.duration_minutes)

        return this_str

    def __repr__(self):
        return "{} {}min".format(super().__repr__(), self.duration_minutes)

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

        if minutes_passed >= self.duration_minutes:
            self.active = False

        return self.active


class Bolus(Measure):
    """
    A bolus
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

    def get_loop_inputs(self):
        """
        Get two numpy arrays for dates and values, used for Loop input
        """
        loop_bg_values = [max(40, min(400, round(bg))) for bg in self.bg_values]
        return self.datetimes, loop_bg_values
