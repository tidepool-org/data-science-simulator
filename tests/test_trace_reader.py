"""Unit tests for the TRSET-21 trace-tier reader (isolation).

Exercises the reader against synthetic TSVs written to disk -- validates
alignment, NaN preservation, tolerance of the variable trailing
``loop_prediction_abs_error_*`` columns, and every explicit error path.

The integration test (``test_trace_reader_integration.py``) covers the real
``loop_risk_v2_0`` boundary; here we keep inputs synthetic and self-contained.
"""
import numpy as np
import pandas as pd
import pytest

from tidepool_data_science_simulator.trace.reader import (
    FIELD_TO_COLUMN,
    TIME_COLUMN,
    SimulationTrace,
    TraceReadError,
    read_trace,
)

SERIES_COLUMNS = list(FIELD_TO_COLUMN.values())


def _write_tsv(df, path):
    """Write a df to a tab-separated .tsv the way save_df() does (index first)."""
    df.to_csv(path, sep="\t", index=False)
    return str(path)


def _base_frame():
    """A minimal, contract-complete 3-row frame with a datetime time column."""
    times = pd.to_datetime(
        ["2019-08-15T12:00:00Z", "2019-08-15T12:05:00Z", "2019-08-15T12:10:00Z"]
    )
    data = {TIME_COLUMN: times}
    for i, col in enumerate(SERIES_COLUMNS):
        data[col] = [float(i), float(i) + 1.0, float(i) + 2.0]
    return pd.DataFrame(data)


def test_reads_all_first_class_fields(tmp_path):
    path = _write_tsv(_base_frame(), tmp_path / "sim-abc.tsv")

    trace = read_trace(path)

    assert isinstance(trace, SimulationTrace)
    assert trace.sim_id == "sim-abc"
    assert isinstance(trace.time, pd.DatetimeIndex)
    assert len(trace.time) == 3
    # Every series is present, aligned to the time axis, correct length.
    for field in FIELD_TO_COLUMN:
        series = getattr(trace, field)
        assert isinstance(series, pd.Series)
        assert series.index.equals(trace.time)
        assert len(series) == 3


def test_series_values_match_source_columns(tmp_path):
    df = _base_frame()
    path = _write_tsv(df, tmp_path / "sim-values.tsv")

    trace = read_trace(path)

    for field, column in FIELD_TO_COLUMN.items():
        np.testing.assert_array_equal(
            getattr(trace, field).to_numpy(), df[column].to_numpy()
        )


def test_time_is_parsed_to_datetime(tmp_path):
    path = _write_tsv(_base_frame(), tmp_path / "sim-time.tsv")

    trace = read_trace(path)

    assert pd.api.types.is_datetime64_any_dtype(trace.time)
    assert trace.time[0] == pd.Timestamp("2019-08-15T12:00:00Z")


def test_empty_cells_preserved_as_nan_not_forward_filled(tmp_path):
    df = _base_frame()
    # Sparse loop_cob: only the middle row has a value (AC #4).
    df["loop_cob"] = [np.nan, 7.5, np.nan]
    path = _write_tsv(df, tmp_path / "sim-sparse.tsv")

    trace = read_trace(path)

    assert np.isnan(trace.loop_cob.iloc[0])
    assert trace.loop_cob.iloc[1] == 7.5
    assert np.isnan(trace.loop_cob.iloc[2])  # not forward-filled from row 1


def test_entirely_empty_loop_cob_tolerated(tmp_path):
    df = _base_frame()
    df["loop_cob"] = [np.nan, np.nan, np.nan]  # non-active-Loop stage
    path = _write_tsv(df, tmp_path / "sim-nocob.tsv")

    trace = read_trace(path)  # must not raise (AC #4)

    assert trace.loop_cob.isna().all()


def test_tolerates_variable_abs_error_columns(tmp_path):
    df = _base_frame()
    # These trailing columns appear only on some rows in real output; the reader
    # targets named columns and must ignore them (Known Constraints).
    df["loop_prediction_abs_error_30_ago_true"] = [np.nan, 4.0, np.nan]
    df["loop_prediction_abs_error_30_ago_sensor"] = [np.nan, 5.0, np.nan]
    path = _write_tsv(df, tmp_path / "sim-abserr.tsv")

    trace = read_trace(path)

    assert len(trace.time) == 3
    # No forecast/abs-error field leaks onto the trace object (AC #5).
    assert not any("abs_error" in f for f in FIELD_TO_COLUMN)


def test_no_prediction_field_exposed():
    # AC #5: the dataclass carries no prediction/forecast field of any kind.
    field_names = {f for f in FIELD_TO_COLUMN}
    assert not any(
        key in name for name in field_names for key in ("pred", "forecast")
    )


def test_missing_expected_column_raises(tmp_path):
    df = _base_frame().drop(columns=["iob"])
    path = _write_tsv(df, tmp_path / "sim-missing.tsv")

    with pytest.raises(TraceReadError, match="missing expected columns.*iob"):
        read_trace(path)


def test_missing_time_column_raises(tmp_path):
    df = _base_frame().drop(columns=[TIME_COLUMN])
    path = _write_tsv(df, tmp_path / "sim-notime.tsv")

    with pytest.raises(TraceReadError, match="missing the required 'time' column"):
        read_trace(path)


def test_nonexistent_file_raises(tmp_path):
    with pytest.raises(TraceReadError, match="does not exist"):
        read_trace(tmp_path / "nope.tsv")


def test_empty_file_raises(tmp_path):
    path = tmp_path / "sim-empty.tsv"
    path.write_text("")

    with pytest.raises(TraceReadError, match="empty"):
        read_trace(str(path))


def test_header_only_file_raises(tmp_path):
    path = tmp_path / "sim-headeronly.tsv"
    header = "\t".join([TIME_COLUMN] + SERIES_COLUMNS) + "\n"
    path.write_text(header)

    with pytest.raises(TraceReadError, match="no data rows"):
        read_trace(str(path))


def test_unparseable_time_column_raises(tmp_path):
    df = _base_frame()
    df[TIME_COLUMN] = ["not-a-date", "also-bad", "still-bad"]
    path = _write_tsv(df, tmp_path / "sim-badtime.tsv")

    with pytest.raises(TraceReadError, match="unparseable 'time' column"):
        read_trace(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
