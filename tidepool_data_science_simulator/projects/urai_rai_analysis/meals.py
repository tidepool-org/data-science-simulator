"""Random meal schedules for one patient draw (TRSET-74)."""
import datetime
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from tidepool_data_science_simulator.projects.urai_rai_analysis.config import MealConfig

MAX_ATTEMPTS = 2000


@dataclass(frozen=True)
class Meal:
    """One meal. ``time`` is on the 5-minute grid; ``absorption_min`` is written to the carb entry ``duration``."""
    time: datetime.datetime
    carbs_g: int
    absorption_min: int


@dataclass(frozen=True)
class MealSchedule:
    """All meals of a draw, sorted by time, and whether the draw includes a seconds pair."""
    meals: Tuple[Meal, ...]
    has_seconds_pair: bool


def _minutes(clock: str) -> int:
    hours, minutes = clock.split(":")
    return int(hours) * 60 + int(minutes)


def _day_windows(run_start: datetime.datetime, run_end: datetime.datetime, cfg: MealConfig):
    """(window_start, window_end) per calendar day, clipped to the run; run end is exclusive."""
    windows = []
    day = run_start.replace(hour=0, minute=0, second=0, microsecond=0)
    last_slot = run_end - datetime.timedelta(minutes=cfg.grid_min)
    while day < run_end:
        lo = max(day + datetime.timedelta(minutes=_minutes(cfg.window_start)), run_start)
        hi = min(day + datetime.timedelta(minutes=_minutes(cfg.window_end)), last_slot)
        windows.append((lo, hi))
        day += datetime.timedelta(days=1)
    return windows


def _spaced(slots: np.ndarray, min_gap: int) -> bool:
    return bool(np.all(np.diff(np.sort(slots)) >= min_gap))


def _sample_day(rng, lo_min: int, hi_min: int, k: int, pair: bool, cfg: MealConfig) -> List[int]:
    """Meal times (minutes from the run start) for one day; ``k`` includes the second meal of a pair."""
    grid = np.arange(lo_min, hi_min + 1, cfg.grid_min)
    n_primary = k - 1 if pair else k
    gaps = np.arange(cfg.seconds_pair_gap_min[0], cfg.seconds_pair_gap_min[1] + 1, cfg.grid_min)
    for _ in range(MAX_ATTEMPTS):
        if len(grid) < n_primary:
            break
        primary = rng.choice(grid, size=n_primary, replace=False)
        if not _spaced(primary, cfg.min_spacing_min):
            continue
        if not pair:
            return sorted(primary.tolist())
        anchor = int(rng.choice(primary))
        second = anchor + int(rng.choice(gaps))
        others = primary[primary != anchor]
        if second > hi_min or (len(others) and np.min(np.abs(others - second)) < cfg.min_spacing_min):
            continue
        return sorted(primary.tolist() + [second])
    raise ValueError(f"Could not place {k} meals (pair={pair}) in window of minutes {lo_min}-{hi_min} under the rules")


def generate_meals(seed: int, cfg: MealConfig, run_start: datetime.datetime,
                   duration_hours: float) -> MealSchedule:
    """Generate the meal schedule for one patient draw.

    Per calendar day, 2-4 meals fall in the day window, clipped to the run. Meals are at least ``min_spacing_min``
    apart, except that with probability ``seconds_pair_prob`` one day gets a seconds pair whose second meal follows
    the first by 5-55 min and counts toward that day's total.

    Raises
    ------
    ValueError
        If the rules cannot be satisfied for a day, or meals on adjacent days end up closer than the spacing.
    """
    rng = np.random.default_rng(seed)
    run_end = run_start + datetime.timedelta(hours=duration_hours)
    windows = _day_windows(run_start, run_end, cfg)
    has_pair = bool(rng.random() < cfg.seconds_pair_prob)
    pair_day = int(rng.integers(len(windows))) if has_pair else -1

    meals: List[Meal] = []
    for day_index, (lo, hi) in enumerate(windows):
        k = int(rng.integers(cfg.meals_per_day[0], cfg.meals_per_day[1] + 1))
        lo_min = int((lo - run_start).total_seconds() // 60)
        hi_min = int((hi - run_start).total_seconds() // 60)
        for minute in _sample_day(rng, lo_min, hi_min, k, day_index == pair_day, cfg):
            meals.append(Meal(
                time=run_start + datetime.timedelta(minutes=minute),
                carbs_g=int(rng.integers(cfg.carbs_g[0], cfg.carbs_g[1] + 1)),
                absorption_min=int(rng.choice(cfg.absorption_times_min)),
            ))
    meals.sort(key=lambda m: m.time)
    for first, second in zip(meals, meals[1:]):
        gap_min = (second.time - first.time).total_seconds() / 60
        if first.time.date() != second.time.date() and gap_min < cfg.min_spacing_min:
            raise ValueError(f"Meals on adjacent days are {gap_min:g} min apart, under {cfg.min_spacing_min}")
    return MealSchedule(meals=tuple(meals), has_seconds_pair=has_pair)
