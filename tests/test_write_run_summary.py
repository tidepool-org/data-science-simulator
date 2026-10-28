"""
Integration tests for write_run_summary (TRSET-55).

Each run must leave exactly one Risk_Results_*.csv and one metadata.json, with the
metadata timestamp matching the CSV filename, and a failed write must raise.
"""

import glob
import json
import os

import pandas as pd
import pytest

from tidepool_data_science_simulator.projects.risk.loop_risk_v2_0 import (
    write_run_summary,
)


@pytest.fixture
def results():
    return [
        pd.DataFrame({"sim_id": ["a", "b"], "lbgi": [1.0, 2.0]}),
        pd.DataFrame({"sim_id": ["c"], "lbgi": [3.0]}),
    ]


def test_one_csv_and_one_metadata(tmp_path, results):
    write_run_summary(results, str(tmp_path), {})

    assert len(glob.glob(str(tmp_path / "Risk_Results_*.csv"))) == 1
    assert len(glob.glob(str(tmp_path / "metadata.json"))) == 1


def test_metadata_timestamp_matches_csv_name(tmp_path, results):
    csv_path = write_run_summary(results, str(tmp_path), {})

    with open(tmp_path / "metadata.json") as fh:
        timestamp = json.load(fh)["timestamp"]
    assert os.path.basename(csv_path) == f"Risk_Results_{timestamp}.csv"


def test_csv_content_is_concat_of_inputs(tmp_path, results):
    csv_path = write_run_summary(results, str(tmp_path), {})

    written = pd.read_csv(csv_path, index_col=0)
    expected = pd.concat(results)
    expected.index = written.index
    pd.testing.assert_frame_equal(written, expected)


def test_failure_raises_logs_and_writes_nothing(tmp_path, caplog):
    with caplog.at_level("ERROR"):
        with pytest.raises(Exception):
            write_run_summary([], str(tmp_path), {})

    assert "Failed to write run summary" in caplog.text
    assert os.listdir(tmp_path) == []
