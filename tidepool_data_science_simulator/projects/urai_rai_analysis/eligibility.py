"""Eligibility rules E1-E6 for the URAI-RAI generator (TRSET-77 AC 2, 3, 5, 6).

Every pool file is read once. A patient is either excluded by the first failing rule (E1 -> E6, with a detail string) or
eligible, in which case the parsed record and the settings flags are kept. Nothing here writes to disk.
"""
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from tidepool_data_science_simulator.projects.urai_rai_analysis.config import RunConfig
from tidepool_data_science_simulator.projects.urai_rai_analysis.max_bolus import (
    SettingsFlags, settings_flags, start_minutes)
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import (
    GLUCOSE_ROW, PatientRecord, Schedule, TargetSchedule, find_glucose_seed, list_pool_files, non_empty, read_cells)

logger = logging.getLogger(__name__)

RULES = ("E1", "E2", "E3", "E4", "E5", "E6")


@dataclass(frozen=True)
class Exclusion:
    """A patient excluded before sampling: the first failing rule and what failed."""
    patient_id: str
    rule: str
    detail: str


@dataclass(frozen=True)
class EligiblePatient:
    """A patient that passed E1-E6, with the settings flags and whether the default suspend threshold was used."""
    record: PatientRecord
    path: str
    default_suspend: bool
    flags: SettingsFlags


@dataclass(frozen=True)
class PoolEvaluation:
    """Result of the whole-pool eligibility pass."""
    eligible: Tuple[EligiblePatient, ...]
    exclusions: Tuple[Exclusion, ...]


def _numbers(tokens: Sequence[str]) -> Optional[Tuple[float, ...]]:
    """The tokens as floats, or None if there are none or any is not a finite number."""
    try:
        values = tuple(float(t) for t in tokens)
    except ValueError:
        return None
    return values if values and all(v == v and abs(v) != float("inf") for v in values) else None


def _read_schedule(rows: Dict[str, List[str]], label: str, times_name: str, values_name: str,
                   problems: List[str]) -> Optional[Schedule]:
    times, values = rows.get(times_name, []), _numbers(rows.get(values_name, []))
    if not times or values is None:
        problems.append(f"{label} missing or non-numeric")
    elif len(times) != len(values):
        problems.append(f"{label} has {len(times)} start times but {len(values)} values")
    else:
        return Schedule(tuple(times), values)
    return None


def _read_target(rows: Dict[str, List[str]], problems: List[str]) -> Optional[TargetSchedule]:
    times = rows.get("target_range_start_times", [])
    lower = _numbers(rows.get("target_range_minimum_values", []))
    upper = _numbers(rows.get("target_range_maximum_values", []))
    if not times or lower is None or upper is None:
        problems.append("target range missing or non-numeric")
    elif not len(times) == len(lower) == len(upper):
        problems.append("target range start times, minimums and maximums differ in length")
    else:
        return TargetSchedule(tuple(times), lower, upper)
    return None


def _age(rows: Dict[str, List[str]], path: str) -> float:
    values = _numbers(rows.get("age", []))
    if values is None or len(values) != 1:
        raise ValueError(f"{path}: age is missing or not a single number, so eligibility cannot be decided")
    return values[0]


def _out_of_range(label: str, values: Sequence[float], bounds: Tuple[float, float]) -> Optional[str]:
    for value in values:
        if not bounds[0] <= value <= bounds[1]:
            return f"{label} {value:g} outside {bounds[0]:g}-{bounds[1]:g}"
    return None


def evaluate_patient(path: str, config: RunConfig) -> Union[EligiblePatient, Exclusion]:
    """Apply E1-E6 to one raw patient file.

    Raises
    ------
    ValueError
        If the file repeats a setting row it should not, or its age is missing (E1 cannot be decided).
    """
    patient_id = os.path.splitext(os.path.basename(path))[0]
    cells = read_cells(path)
    rows = {name: non_empty(row) for name, row in cells.items()}
    rails = config.guardrails

    age = _age(rows, path)
    if age < config.min_age:
        return Exclusion(patient_id, "E1", f"age {age:g} below {config.min_age:g}")

    seed = find_glucose_seed(cells.get(GLUCOSE_ROW, []), config.glucose_seed_points)
    if seed is None:
        return Exclusion(patient_id, "E2", f"no run of {config.glucose_seed_points} consecutive glucose values")
    seed_values = [float(token) for token in seed]
    if _out_of_range("glucose", seed_values, rails.glucose_seed):
        return Exclusion(patient_id, "E3", "selected glucose run has a " + _out_of_range("glucose", seed_values,
                                                                                         rails.glucose_seed))

    problems: List[str] = []
    basal = _read_schedule(rows, "basal rate", "basal_rate_start_times", "basal_rate_values", problems)
    carb_ratio = _read_schedule(rows, "carb ratio", "carb_ratio_start_times", "carb_ratio_values", problems)
    isf = _read_schedule(rows, "insulin sensitivity", "sensitivity_ratio_start_times", "sensitivity_ratio_values",
                         problems)
    target = _read_target(rows, problems)
    suspend = _numbers(rows.get("suspend_threshold", []))
    default_suspend = False
    if suspend is None or len(suspend) != 1:
        if config.default_suspend_threshold is None:
            problems.append("suspend threshold missing or non-numeric")
        suspend, default_suspend = (config.default_suspend_threshold,), True
    for label, schedule in (("basal rate", basal), ("carb ratio", carb_ratio), ("insulin sensitivity", isf),
                            ("target range", target)):
        if schedule is not None:
            try:
                start_minutes(schedule.start_times)
            except ValueError as error:
                problems.append(f"{label}: {error}")
    if problems:
        return Exclusion(patient_id, "E4", "; ".join(problems))
    suspend_value = suspend[0]

    violations = [v for v in (
        _out_of_range("basal rate", basal.values, rails.basal_rate),
        _out_of_range("carb ratio", carb_ratio.values, rails.carb_ratio),
        _out_of_range("insulin sensitivity", isf.values, rails.isf),
        _out_of_range("target minimum", target.lower_values, rails.target),
        _out_of_range("target maximum", target.upper_values, rails.target),
        _out_of_range("suspend threshold", (suspend_value,), rails.suspend_threshold),
    ) if v]
    if violations:
        return Exclusion(patient_id, "E5", "; ".join(violations))

    relations = []
    if any(low > high for low, high in zip(target.lower_values, target.upper_values)):
        relations.append("target minimum above target maximum")
    if any(suspend_value > low for low in target.lower_values):
        relations.append(f"suspend threshold {suspend_value:g} above a target minimum")
    if relations:
        return Exclusion(patient_id, "E6", "; ".join(relations))

    record = PatientRecord(
        patient_id=patient_id, age=age, suspend_threshold=suspend_value, basal=basal, carb_ratio=carb_ratio, isf=isf,
        target_range=target, glucose_tokens=seed,
        actual_basal=_numbers(rows.get("actual_basal_rates", [])),
        actual_carb_ratio=_numbers(rows.get("actual_carb_ratios", [])),
        actual_isf=_numbers(rows.get("actual_sensitivity_ratios", [])),
    )
    flags = settings_flags(basal, carb_ratio.values[0], max(config.meals.carbs_g), config.max_bolus)
    return EligiblePatient(record=record, path=path, default_suspend=default_suspend, flags=flags)


def evaluate_pool(pool_dir: str, config: RunConfig) -> PoolEvaluation:
    """Apply E1-E6 to every patient in the pool. Reads only; never writes to the pool."""
    eligible, exclusions = [], []
    for path in list_pool_files(pool_dir):
        result = evaluate_patient(path, config)
        (exclusions if isinstance(result, Exclusion) else eligible).append(result)
    counts = {rule: sum(e.rule == rule for e in exclusions) for rule in RULES}
    logger.info("Eligibility: %d eligible, excluded by rule %s", len(eligible), counts)
    return PoolEvaluation(tuple(eligible), tuple(exclusions))


def check_pump_matches_model(patient: EligiblePatient) -> None:
    """Hard error unless the pump and patient-model schedules agree; the study uses no settings variance.

    Raises
    ------
    ValueError
        If a patient-model (``actual_*``) schedule is missing or differs from the pump schedule.
    """
    record = patient.record
    for name, pump_values, actual in (
            ("actual_basal_rates", record.basal.values, record.actual_basal),
            ("actual_carb_ratios", record.carb_ratio.values, record.actual_carb_ratio),
            ("actual_sensitivity_ratios", record.isf.values, record.actual_isf)):
        if actual is None:
            raise ValueError(f"{patient.path}: {name} is missing, so pump and patient-model settings cannot be compared")
        if actual != pump_values:
            raise ValueError(f"{patient.path}: {name} {actual} differs from the pump setting {pump_values}")


def eligible_table(evaluation: PoolEvaluation) -> pd.DataFrame:
    """Eligible patients as columns patient_id, age, isf (first schedule entry), path, for cohort filtering."""
    return pd.DataFrame.from_records(
        [{"patient_id": p.record.patient_id, "age": p.record.age, "isf": p.record.isf_for_cohorts, "path": p.path}
         for p in evaluation.eligible],
        columns=["patient_id", "age", "isf", "path"])
