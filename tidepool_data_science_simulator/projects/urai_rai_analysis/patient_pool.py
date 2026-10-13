"""Read-only access to the raw-format virtual-patient CSV pool (TRSET-74).

Raw file layout: one row per setting, columns ``setting_name, settings, 0, 1, 2, ...``. Depending on the setting, the
value(s) sit in ``settings`` (scalars such as ``age``) or start in column ``0`` (schedules and series). Both layouts
are read with one rule: the tokens of a row are its non-empty cells from ``settings`` onward, in column order.
"""
import csv
import logging
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import pandas as pd

logger = logging.getLogger(__name__)

csv.field_size_limit(2 ** 31 - 1)

GLUCOSE_ROW = "actual_blood_glucose"


@dataclass(frozen=True)
class Schedule:
    """A setting schedule exactly as written in the source file (start times are the source strings)."""
    start_times: Tuple[str, ...]
    values: Tuple[float, ...]


@dataclass(frozen=True)
class TargetSchedule:
    """Target-range schedule from ``target_range_minimum_values`` / ``target_range_maximum_values``."""
    start_times: Tuple[str, ...]
    lower_values: Tuple[float, ...]
    upper_values: Tuple[float, ...]


@dataclass(frozen=True)
class PatientRecord:
    """One virtual patient's settings and glucose seed, all values unchanged from the source file."""
    patient_id: str
    age: float
    suspend_threshold: float
    basal: Schedule
    carb_ratio: Schedule
    isf: Schedule
    target_range: TargetSchedule
    glucose_tokens: Tuple[str, ...]  # source text of the first n actual_blood_glucose values, in order

    @property
    def isf_for_cohorts(self) -> float:
        """The value used for ISF cohort filters: the first (00:00) schedule entry."""
        return self.isf.values[0]


def read_tokens(path: str) -> Dict[str, List[str]]:
    """Return {setting_name: non-empty cells from ``settings`` onward, in order} for a raw patient CSV.

    Raises
    ------
    ValueError
        If a setting_name appears twice.
    """
    rows: Dict[str, List[str]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            name = row[0]
            if name in rows:
                raise ValueError(f"{path}: duplicate setting_name {name!r}")
            rows[name] = [cell for cell in row[1:] if cell.strip() != ""]
    return rows


def _require(rows: Dict[str, List[str]], name: str, path: str) -> List[str]:
    tokens = rows.get(name)
    if not tokens:
        raise ValueError(f"{path}: required setting {name!r} is missing or empty")
    return tokens


def _scalar(rows: Dict[str, List[str]], name: str, path: str) -> float:
    tokens = _require(rows, name, path)
    if len(tokens) != 1:
        raise ValueError(f"{path}: expected one value for {name!r}, found {len(tokens)}")
    return float(tokens[0])


def _schedule(rows: Dict[str, List[str]], times_name: str, values_name: str, path: str) -> Schedule:
    times = _require(rows, times_name, path)
    values = [float(v) for v in _require(rows, values_name, path)]
    if len(times) != len(values):
        raise ValueError(f"{path}: {times_name} has {len(times)} entries but {values_name} has {len(values)}")
    return Schedule(tuple(times), tuple(values))


def parse_patient(path: str, n_glucose: int) -> PatientRecord:
    """Parse one raw patient CSV into a PatientRecord.

    The pump (``*_values``) and patient-model (``actual_*``) settings must agree; the study uses no settings variance.

    Raises
    ------
    ValueError
        If a required setting is missing, schedule lengths disagree, pump and actual values differ, or fewer than
        ``n_glucose`` glucose values are present.
    """
    rows = read_tokens(path)
    basal = _schedule(rows, "basal_rate_start_times", "basal_rate_values", path)
    carb_ratio = _schedule(rows, "carb_ratio_start_times", "carb_ratio_values", path)
    isf = _schedule(rows, "sensitivity_ratio_start_times", "sensitivity_ratio_values", path)
    for schedule, actual_name in ((basal, "actual_basal_rates"), (carb_ratio, "actual_carb_ratios"),
                                  (isf, "actual_sensitivity_ratios")):
        actual = tuple(float(v) for v in _require(rows, actual_name, path))
        if actual != schedule.values:
            raise ValueError(f"{path}: {actual_name} {actual} differs from the pump setting {schedule.values}")

    target_times = _require(rows, "target_range_start_times", path)
    lower = tuple(float(v) for v in _require(rows, "target_range_minimum_values", path))
    upper = tuple(float(v) for v in _require(rows, "target_range_maximum_values", path))
    if not len(target_times) == len(lower) == len(upper):
        raise ValueError(f"{path}: target range start times, minimums and maximums differ in length")

    glucose = _require(rows, GLUCOSE_ROW, path)
    if len(glucose) < n_glucose:
        raise ValueError(f"{path}: {GLUCOSE_ROW} has {len(glucose)} values, need {n_glucose}")

    return PatientRecord(
        patient_id=os.path.splitext(os.path.basename(path))[0],
        age=_scalar(rows, "age", path),
        suspend_threshold=_scalar(rows, "suspend_threshold", path),
        basal=basal, carb_ratio=carb_ratio, isf=isf,
        target_range=TargetSchedule(tuple(target_times), lower, upper),
        glucose_tokens=tuple(glucose[:n_glucose]),
    )


def list_pool_files(pool_dir: str) -> List[str]:
    """Sorted CSV paths in the pool directory.

    Raises
    ------
    FileNotFoundError
        If the directory is missing or holds no CSV files.
    """
    if not os.path.isdir(pool_dir):
        raise FileNotFoundError(f"Patient pool directory not found: {pool_dir}")
    files = sorted(os.path.join(pool_dir, f) for f in os.listdir(pool_dir) if f.endswith(".csv"))
    if not files:
        raise FileNotFoundError(f"Patient pool directory has no CSV files: {pool_dir}")
    return files


def scan_pool(pool_dir: str) -> pd.DataFrame:
    """Read only the cohort-relevant fields of every patient: columns patient_id, age, isf, path.

    Only reads the pool; never writes to it.
    """
    records = []
    for path in list_pool_files(pool_dir):
        rows = read_tokens(path)
        isf = _schedule(rows, "sensitivity_ratio_start_times", "sensitivity_ratio_values", path)
        records.append({
            "patient_id": os.path.splitext(os.path.basename(path))[0],
            "age": _scalar(rows, "age", path),
            "isf": isf.values[0],
            "path": path,
        })
    return pd.DataFrame.from_records(records)
