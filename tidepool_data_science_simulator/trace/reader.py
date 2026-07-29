"""Trace-tier reader (TRSET-21).

Parse a single simulator run's ``<sim_id>.tsv`` output into a structured,
typed :class:`SimulationTrace` exposing the history time series that back a
Loop-home-screen-style visualization: true/sensor BG, IOB, basal, boluses,
carbs, and COB.

This is a *read-only consumer* of the TSV contract defined by
``models.simulation.Simulation.get_results_df()`` and written by
``utils.save_df()`` (tab-separated, with the ``time`` index as the first
column). It selects columns *by name*, so it tolerates the variable trailing
``loop_prediction_abs_error_*`` columns that appear on only some rows.

Deliberately excluded: any prediction/forecast series. The TSV stores only a
per-timestamp scalar endpoint (``loop_final_glucose_pred`` = the last value of
Loop's forecast per step), not the full forecast array, so a forward-projecting
forecast line is not reconstructable from current output. Exposing it would
require persisting the full ``predicted_glucose_values`` array at the source --
its own follow-up ticket, not built here.
"""
import os
from dataclasses import dataclass, fields

import pandas as pd

# The time axis every series aligns to. Written by save_df() as the first
# (index) column of the TSV.
TIME_COLUMN = "time"

# Maps each first-class trace field to the TSV column it is sourced from. The
# field names are the SimulationTrace attributes; the values are the
# get_results_df() column names. Every one of these columns must be present in
# the file (AC #6); their *cells* may be empty (AC #4).
FIELD_TO_COLUMN = {
    # BG
    "bg": "bg",
    "bg_sensor": "bg_sensor",
    # IOB
    "iob": "iob",
    # Basal
    "sbr": "sbr",
    "temp_basal": "temp_basal",
    # Boluses
    "true_bolus": "true_bolus",
    "reported_bolus": "reported_bolus",
    # Carbs
    "true_carb_value": "true_carb_value",
    "reported_carb_value": "reported_carb_value",
    # COB -- may be sparse or entirely empty on non-active-Loop stages (AC #4).
    "loop_cob": "loop_cob",
}


class TraceReadError(ValueError):
    """Raised when a ``<sim_id>.tsv`` cannot be read or is missing the contract.

    Subclasses ``ValueError`` so callers may catch either. Carries an
    informative message naming the file and the specific problem -- never a
    silent failure (workflow section 4 / AC #6).
    """


@dataclass(frozen=True)
class SimulationTrace:
    """Typed trace of one simulator run's history series.

    All series share ``time`` as their index. Missing/empty cells are ``NaN``
    (never fabricated or forward-filled). No prediction/forecast field is
    exposed -- see the module docstring.
    """

    sim_id: str
    time: pd.DatetimeIndex
    # BG
    bg: pd.Series
    bg_sensor: pd.Series
    # IOB
    iob: pd.Series
    # Basal
    sbr: pd.Series
    temp_basal: pd.Series
    # Boluses
    true_bolus: pd.Series
    reported_bolus: pd.Series
    # Carbs
    true_carb_value: pd.Series
    reported_carb_value: pd.Series
    # COB
    loop_cob: pd.Series


def read_trace(path) -> SimulationTrace:
    """Read one ``<sim_id>.tsv`` into a :class:`SimulationTrace`.

    Parameters
    ----------
    path : str or os.PathLike
        Path to a single simulator run's tab-separated ``.tsv`` output.

    Returns
    -------
    SimulationTrace
        The run's history series, all aligned to the parsed ``time`` axis.

    Raises
    ------
    TraceReadError
        If the file is missing/unreadable/empty, lacks the ``time`` column, or
        is missing any expected series column (AC #6).
    """
    path = os.fspath(path)

    if not os.path.isfile(path):
        raise TraceReadError("Trace file does not exist: {}".format(path))

    try:
        df = pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError as e:
        raise TraceReadError("Trace file is empty: {}".format(path)) from e
    except Exception as e:  # unreadable/malformed -- surface, never swallow
        raise TraceReadError(
            "Could not read trace file {}: {}".format(path, e)
        ) from e

    if df.empty:
        raise TraceReadError("Trace file has no data rows: {}".format(path))

    if TIME_COLUMN not in df.columns:
        raise TraceReadError(
            "Trace file {} is missing the required '{}' column".format(
                path, TIME_COLUMN
            )
        )

    missing = [
        column
        for column in FIELD_TO_COLUMN.values()
        if column not in df.columns
    ]
    if missing:
        raise TraceReadError(
            "Trace file {} is missing expected columns: {}".format(
                path, ", ".join(missing)
            )
        )

    # Parse the time axis and align every series to it. errors="raise" so a
    # non-datetime time column fails explicitly rather than silently.
    try:
        time_index = pd.DatetimeIndex(pd.to_datetime(df[TIME_COLUMN]))
    except Exception as e:
        raise TraceReadError(
            "Trace file {} has an unparseable '{}' column: {}".format(
                path, TIME_COLUMN, e
            )
        ) from e
    time_index.name = TIME_COLUMN

    # Column selection into aligned series -- no fillna/forward-fill, so empty
    # cells remain NaN (AC #4). sim_id is a convenience derived from the stem.
    series = {
        field: pd.Series(
            df[column].to_numpy(), index=time_index, name=field
        )
        for field, column in FIELD_TO_COLUMN.items()
    }
    sim_id = os.path.splitext(os.path.basename(path))[0]

    return SimulationTrace(sim_id=sim_id, time=time_index, **series)


# Field-name/dataclass consistency guard: the only non-series fields are
# ``sim_id`` and ``time``; every other SimulationTrace field must have a
# column mapping. Fails loudly at import if the two drift out of sync.
_series_fields = {f.name for f in fields(SimulationTrace)} - {"sim_id", "time"}
assert _series_fields == set(FIELD_TO_COLUMN), (
    "SimulationTrace series fields and FIELD_TO_COLUMN are out of sync: "
    "{}".format(_series_fields.symmetric_difference(set(FIELD_TO_COLUMN)))
)
