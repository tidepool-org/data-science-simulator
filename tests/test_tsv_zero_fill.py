"""TRSET-53: fill blank per-step columns in the results TSV only.

Unit tests for ``format_results_for_file`` / ``save_df(format_for_file=True)``
plus a file-boundary integration test on the TLR-000-swift suite (real Swift
dylib, skipped when unavailable).
"""
import datetime
import os

import numpy as np
import pandas as pd
import pytest

from tidepool_data_science_simulator.utils import (
    EVENT_COLUMNS,
    LOOP_COLUMNS,
    LOOP_PLACEHOLDER,
    format_results_for_file,
    save_df,
)

ALL_COLUMNS = EVENT_COLUMNS + LOOP_COLUMNS


def _frame(active=(1, 1), values=None):
    """Frame with all 18 columns blank (NaN) except optional per-column values."""
    n = len(active)
    df = pd.DataFrame({"active": list(active)})
    for col in ALL_COLUMNS:
        df[col] = np.nan
    for col, vals in (values or {}).items():
        df[col] = vals
    return df


def test_counts():
    assert len(EVENT_COLUMNS) == 6 and len(LOOP_COLUMNS) == 12
    assert not set(EVENT_COLUMNS) & set(LOOP_COLUMNS)


@pytest.mark.parametrize("col", EVENT_COLUMNS)
def test_event_blank_becomes_placeholder_real_value_and_zero_kept(col):
    out = format_results_for_file(_frame(active=(1, 1, 1), values={col: [np.nan, 2.5, 0.0]}))
    assert out[col].iloc[0] == LOOP_PLACEHOLDER
    assert out[col].iloc[1] == 2.5
    assert out[col].iloc[2] == 0.0 and out[col].iloc[2] != LOOP_PLACEHOLDER


@pytest.mark.parametrize("col", LOOP_COLUMNS)
def test_loop_blank_becomes_placeholder_never_zero(col):
    out = format_results_for_file(_frame())
    assert (out[col] == LOOP_PLACEHOLDER).all()


@pytest.mark.parametrize("col", LOOP_COLUMNS)
def test_real_zero_kept_for_loop_columns(col):
    out = format_results_for_file(_frame(values={col: [0.0, None]}))
    assert out[col].iloc[0] == 0.0 and out[col].iloc[0] != LOOP_PLACEHOLDER
    assert out[col].iloc[1] == LOOP_PLACEHOLDER


def test_presim_rows_blank_in_all_columns():
    out = format_results_for_file(_frame(active=(0, 1)))
    assert out.loc[0, ALL_COLUMNS].isna().all()
    assert not out.loc[1, ALL_COLUMNS].isna().any()


def test_failed_extraction_row_is_placeholder_not_zero():
    out = format_results_for_file(_frame(values={"true_bolus": [1.0, 1.0]}))
    assert (out.loc[:, LOOP_COLUMNS] == LOOP_PLACEHOLDER).all().all()
    assert not (out.loc[:, ALL_COLUMNS] == 0).any().any()


def test_input_not_mutated():
    df = _frame()
    before = df.copy()
    format_results_for_file(df)
    pd.testing.assert_frame_equal(df, before)


def test_save_df_default_does_not_fill(tmp_path):
    save_df(_frame(), "plain", str(tmp_path))
    raw = pd.read_csv(tmp_path / "plain.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert (raw[ALL_COLUMNS] == "").all().all()


def test_save_df_format_for_file_fills(tmp_path):
    save_df(_frame(active=(0, 1)), "filled", str(tmp_path), format_for_file=True)
    raw = pd.read_csv(tmp_path / "filled.tsv", sep="\t", dtype=str, keep_default_na=False)
    assert (raw.loc[0, ALL_COLUMNS] == "").all()
    assert (raw.loc[1, ALL_COLUMNS] == LOOP_PLACEHOLDER).all()


# --- Integration: file boundary ------------------------------------------------

try:
    import loop_to_python_api.api  # noqa: F401
    _DYLIB = True
except Exception:  # pragma: no cover - env-dependent
    _DYLIB = False


@pytest.mark.skipif(not _DYLIB, reason="Swift dylib not available")
def test_written_tsv_has_no_blanks_on_active_rows(tmp_path):
    from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
    from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController

    config = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/test/TLR-000-swift",
        "Simulation-Configuration-TLR-000-base_median_profile_v1.json",
    )
    io_dir = tmp_path / "io"
    io_dir.mkdir()
    sims = ScenarioParserV2(path_to_json_config=config).get_sims()
    saw_loop = saw_nonloop = False
    for sim_id, sim in sims.items():
        if hasattr(sim.controller, "loop_algo_io_dir"):
            sim.controller.loop_algo_io_dir = str(io_dir)
        sim.run(early_stop_datetime=sim.start_time + datetime.timedelta(hours=1))
        save_df(sim.get_results_df(), sim_id, str(tmp_path), format_for_file=True)
        raw = pd.read_csv(tmp_path / f"{sim_id}.tsv", sep="\t", dtype=str, keep_default_na=False)
        active = raw["active"].isin(["1", "1.0", "True"])
        assert active.any()
        # (a) no blanks on active rows
        assert (raw.loc[active, ALL_COLUMNS] != "").all().all()
        # (c) pre-sim rows blank
        assert (raw.loc[~active, ALL_COLUMNS] == "").all().all()
        if isinstance(sim.controller, SwiftLoopController):
            saw_loop = True
        else:
            saw_nonloop = True
            # (b) non-Loop stage: placeholder in every Loop column, never 0
            assert (raw.loc[active, LOOP_COLUMNS] == LOOP_PLACEHOLDER).all().all()
    assert saw_loop and saw_nonloop
