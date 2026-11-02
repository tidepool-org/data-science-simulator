"""Read-only access to the raw-format virtual-patient CSV pool (TRSET-74).

Raw file layout: one row per setting, columns ``setting_name, settings, 0, 1, 2, ...``. Depending on the setting, the
value(s) sit in ``settings`` (scalars such as ``age``) or start in column ``0`` (schedules and series). Both layouts
are read with one rule: the tokens of a row are its non-empty cells from ``settings`` onward, in column order.
The glucose row is the exception (TRSET-77): it is read positionally, so a blank cell is a missing value and the seed
is the first run of consecutive values with none missing.
"""
import csv
import logging
import math
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

csv.field_size_limit(2 ** 31 - 1)

GLUCOSE_ROW = "actual_blood_glucose"

# Event rows the pool writes once per event (one row per carb / dose). The generator does not use them; their tokens
# are concatenated in file order. Any other repeated setting_name is still an error.
REPEATED_EVENT_ROWS = frozenset(
    {"carb_dates", "actual_carbs", "dose_start_times", "dose_end_times", "dose_types", "dose_values"}
)


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
    """One eligible virtual patient's settings and glucose seed, all values unchanged from the source file.

    ``actual_*`` are the patient-model schedules (``None`` when the file has no such row); they are only used by the
    pump-versus-model consistency check.
    """
    patient_id: str
    age: float
    suspend_threshold: float
    basal: Schedule
    carb_ratio: Schedule
    isf: Schedule
    target_range: TargetSchedule
    glucose_tokens: Tuple[str, ...]  # source text of the selected run of glucose values, in order
    actual_basal: Optional[Tuple[float, ...]] = None
    actual_carb_ratio: Optional[Tuple[float, ...]] = None
    actual_isf: Optional[Tuple[float, ...]] = None

    @property
    def isf_for_cohorts(self) -> float:
        """The value used for ISF cohort filters: the first (00:00) schedule entry."""
        return self.isf.values[0]


def read_cells(path: str) -> Dict[str, List[str]]:
    """Return {setting_name: every cell from ``settings`` onward, blanks kept, in column order} for a raw patient CSV.

    Rows the pool writes once per event (``REPEATED_EVENT_ROWS``) are concatenated in file order.

    Raises
    ------
    ValueError
        If a setting_name outside ``REPEATED_EVENT_ROWS`` appears twice.
    """
    rows: Dict[str, List[str]] = {}
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader)  # header
        for row in reader:
            if not row:
                continue
            name, cells = row[0], row[1:]
            if name in rows:
                if name not in REPEATED_EVENT_ROWS:
                    raise ValueError(f"{path}: duplicate setting_name {name!r}")
                rows[name].extend(cells)
            else:
                rows[name] = cells
    return rows


def non_empty(cells: List[str]) -> List[str]:
    """The tokens of a row: its non-empty cells, in order."""
    return [cell for cell in cells if cell.strip() != ""]


def read_tokens(path: str) -> Dict[str, List[str]]:
    """Return {setting_name: non-empty cells from ``settings`` onward, in order} for a raw patient CSV.

    Raises
    ------
    ValueError
        If a setting_name outside ``REPEATED_EVENT_ROWS`` appears twice.
    """
    return {name: non_empty(cells) for name, cells in read_cells(path).items()}


def _is_number(cell: str) -> bool:
    try:
        return math.isfinite(float(cell))
    except ValueError:
        return False


def find_glucose_seed(cells: List[str], n_values: int) -> Optional[Tuple[str, ...]]:
    """The first run of ``n_values`` consecutive numeric cells, in column order, as source text; None if there is none.

    A blank or non-numeric cell is a missing value and ends a run.
    """
    run_start, run_length = 0, 0
    for i, cell in enumerate(cells):
        if not _is_number(cell):
            run_start, run_length = i + 1, 0
            continue
        run_length += 1
        if run_length == n_values:
            return tuple(cell.strip() for cell in cells[run_start:i + 1])
    return None


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
