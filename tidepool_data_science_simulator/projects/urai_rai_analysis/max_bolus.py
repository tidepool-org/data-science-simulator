"""Per-patient max bolus and settings flags for the URAI-RAI generator (TRSET-77 AC 13, 14).

Everything is computed in exact rational arithmetic (``fractions.Fraction``) built from the shortest decimal text of
each value, so a value that lands exactly on the delivery-increment grid (0.15 or 0.30 U) is never pushed down a step
by binary floating-point error.
"""
from dataclasses import dataclass
from fractions import Fraction
from typing import List, Sequence

from tidepool_data_science_simulator.projects.urai_rai_analysis.config import MaxBolusConfig
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import Schedule

MINUTES_PER_DAY = 24 * 60


@dataclass(frozen=True)
class SettingsFlags:
    """Descriptive per-patient flags. They never exclude a patient."""
    tdd_estimate_u: float
    max_bolus_u: float
    max_bolus_clamped: bool
    meal80_fits: bool


def exact(value: float) -> Fraction:
    """The exact decimal a float was written as (``0.9`` is 9/10, not the nearest binary fraction)."""
    return Fraction(repr(float(value)))


def start_minutes(start_times: Sequence[str]) -> List[Fraction]:
    """Minutes after midnight of each ``H:MM[:SS]`` start time.

    Raises
    ------
    ValueError
        If a time does not parse, the first is not 00:00, or the times are not strictly increasing within a day.
    """
    minutes = []
    for text in start_times:
        parts = text.strip().split(":")
        if len(parts) not in (2, 3) or not all(p.isdigit() for p in parts):
            raise ValueError(f"start time {text!r} is not H:MM[:SS]")
        seconds = int(parts[2]) if len(parts) == 3 else 0
        minutes.append(int(parts[0]) * 60 + int(parts[1]) + Fraction(seconds, 60))
    if minutes[0] != 0:
        raise ValueError(f"schedule starts at {start_times[0]!r}, not 00:00")
    if any(later <= earlier for earlier, later in zip(minutes, minutes[1:])) or minutes[-1] >= MINUTES_PER_DAY:
        raise ValueError("schedule start times are not strictly increasing within one day")
    return minutes


def daily_basal_u(basal: Schedule) -> Fraction:
    """Scheduled basal over one day in units: each rate times the hours it is in force."""
    starts = start_minutes(basal.start_times)
    ends = starts[1:] + [Fraction(MINUTES_PER_DAY)]
    return sum((exact(rate) * (end - start) / 60 for rate, start, end in zip(basal.values, starts, ends)), Fraction(0))


def floor_to_increment(value: Fraction, step: Fraction) -> Fraction:
    """Round ``value`` down to a multiple of ``step``; exact, so on-grid values are returned unchanged."""
    return (value // step) * step


def settings_flags(basal: Schedule, first_carb_ratio: float, largest_meal_g: float,
                   config: MaxBolusConfig) -> SettingsFlags:
    """Per-patient max bolus (clamped, then rounded down to the increment) and the settings flags.

    ``meal80_fits`` is true when the rounded max bolus covers the largest configured meal at the first CIR entry.
    """
    tdd = exact(config.tdd_basal_multiplier) * daily_basal_u(basal)
    uncapped = exact(config.fraction_of_tdd) * tdd
    ceiling = exact(config.ceiling_u)
    step = exact(config.increment_u)
    rounded = floor_to_increment(min(uncapped, ceiling), step)
    return SettingsFlags(
        tdd_estimate_u=float(tdd),
        max_bolus_u=float(rounded),
        max_bolus_clamped=uncapped > ceiling,
        meal80_fits=rounded >= exact(largest_meal_g) / exact(first_carb_ratio),
    )


def controller_max_bolus(flags: SettingsFlags, config: MaxBolusConfig) -> float:
    """The max bolus the controller gets: the static comparator value when set, else the per-patient value."""
    return config.static_u if config.static_u is not None else flags.max_bolus_u
