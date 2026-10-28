"""
run_simulations failure handling (TRSET-80).

Each sim is handled on its own; after every sim has finished and been joined,
SimulationRunError is raised if any failed. Sims are in-process fakes.
"""

import glob
import os

import numpy as np
import pandas as pd
import pytest

from tidepool_data_science_simulator.run import run_simulations, SimulationRunError


class _Queue:
    def __init__(self, df):
        self.df = df

    def get(self):
        return self.df


class _Sim:
    def __init__(self, fault=None):
        n = 288
        cols = {
            "active": np.ones(n),
            "bg": 110.0 + 20.0 * np.sin(np.linspace(0, 6, n)),
            "bg_sensor": 112.0 + 20.0 * np.sin(np.linspace(0, 6, n)),
            "iob": np.full(n, 1.0),
            "sbr": np.full(n, 0.8),
            "delivered_basal_insulin": np.full(n, 0.07),
            "reported_bolus": np.zeros(n),
            "randint": np.arange(n),
        }
        if fault == "metrics":
            del cols["iob"]
        self.fault = fault
        self.queue = _Queue(pd.DataFrame(cols))
        self.joined = False

    def start(self):
        if self.fault == "run":
            raise RuntimeError("boom")

    def join(self):
        self.joined = True

    def close(self):
        pass

    def get_info_stateless(self):
        if self.fault == "save":
            raise OSError("disk full")
        return {}


def test_success_returns_results_and_writes_summary_csv(tmp_path):
    full_results, summary = run_simulations({"a": _Sim(), "b": _Sim()}, str(tmp_path), num_procs=2, name="x")

    assert set(full_results) == {"a", "b"}
    assert list(summary.index) == ["a", "b"]
    assert os.path.exists(tmp_path / "summary_results_x.csv")


@pytest.mark.parametrize("fault", ["run", "metrics", "save"])
def test_failed_sim_raises_after_batch_with_other_sims_done(tmp_path, fault):
    good_first, bad, good_last = _Sim(), _Sim(fault), _Sim()

    with pytest.raises(SimulationRunError) as exc_info:
        run_simulations({"a": good_first, "bad": bad, "c": good_last}, str(tmp_path), num_procs=3, name="x")

    error = exc_info.value
    assert [(f["sim_id"], f["stage"]) for f in error.failures] == [("bad", fault)]
    assert error.failures[0]["exception_type"] and error.failures[0]["traceback"]
    assert set(error.full_results) == {"a", "c"}
    assert list(error.summary_results.index) == ["a", "c"]
    assert not error.summary_results.isna().any().any()
    assert good_first.joined and good_last.joined


def test_incomplete_summary_csv_is_not_written(tmp_path):
    with pytest.raises(SimulationRunError):
        run_simulations({"a": _Sim(), "bad": _Sim("run")}, str(tmp_path), num_procs=2, name="x")

    assert glob.glob(str(tmp_path / "summary_results_*.csv")) == []


def test_every_sim_failing_carries_empty_results(tmp_path):
    with pytest.raises(SimulationRunError) as exc_info:
        run_simulations({"a": _Sim("run"), "b": _Sim("run")}, str(tmp_path), num_procs=2)

    assert len(exc_info.value.failures) == 2
    assert exc_info.value.full_results == {}
    assert exc_info.value.summary_results.empty
