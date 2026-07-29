"""End-to-end integration test for the TRSET-21 trace-tier reader.

Runs a real ``loop_risk_v2_0`` scenario config through the actual parser and the
real Swift ``.dylib`` (no mocks), writes each run's ``get_results_df()`` to a
genuine tab-separated ``<sim_id>.tsv`` on disk (the same format ``save_df()``
produces), then reads it back with ``read_trace()`` and asserts the resulting
:class:`SimulationTrace` matches the file at the boundary -- not mocked.

The harness config is the ``TLR-000-swift`` median suite, whose stages exercise
both Loop-active (SwiftLoopController) and Loop-inactive (DoNothingController)
paths, so the reader is validated on both populated and empty ``loop_cob``.

Run under the arm64 conda env ``tidepool-data-science-simulator`` (the one that
matches the committed .dylib). The module skips cleanly when the dylib isn't
importable, so the suite stays portable.
"""
import datetime
import os

import numpy as np
import pandas as pd
import pytest

# Importing the api module loads the .dylib at import time (ctypes.CDLL); an
# incompatible/missing dylib raises OSError, not ImportError -- catch broadly.
try:
    import loop_to_python_api.api  # noqa: F401
    _DYLIB_AVAILABLE = True
    _DYLIB_ERR = ""
except Exception as e:  # pragma: no cover - env-dependent
    _DYLIB_AVAILABLE = False
    _DYLIB_ERR = repr(e)

pytestmark = pytest.mark.skipif(
    not _DYLIB_AVAILABLE,
    reason=f"loop_to_python_api / .dylib not available in this env: {_DYLIB_ERR}",
)

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.trace.reader import (
    FIELD_TO_COLUMN,
    SimulationTrace,
    read_trace,
)

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(
    REPO_ROOT,
    "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/test/TLR-000-swift",
    "Simulation-Configuration-TLR-000-base_median_profile_v1.json",
)

RUN_HOURS = 1  # early-stop keeps the run near the ~1-min budget


@pytest.fixture(scope="module")
def trace_fixtures(tmp_path_factory):
    """Run the suite once, write each sim to a real .tsv, and return per-sim
    ``(tsv_path, source_df, trace)`` triples keyed by sim_id."""
    io_dir = str(tmp_path_factory.mktemp("loop_algo_io"))
    tsv_dir = str(tmp_path_factory.mktemp("trace_tsv"))
    parser = ScenarioParserV2(path_to_json_config=CONFIG)
    sims = parser.get_sims()

    fixtures = {}
    for sim_id, sim in sims.items():
        if hasattr(sim.controller, "loop_algo_io_dir"):
            sim.controller.loop_algo_io_dir = io_dir
        stop = sim.start_time + datetime.timedelta(hours=RUN_HOURS)
        sim.run(early_stop_datetime=stop)

        df = sim.get_results_df()  # time is the index, matching save_df()
        tsv_path = os.path.join(tsv_dir, "{}.tsv".format(sim_id))
        df.to_csv(tsv_path, sep="\t")  # exactly what utils.save_df() writes

        fixtures[sim_id] = (tsv_path, df, read_trace(tsv_path))
    assert fixtures, "no sims produced by the harness config"
    return fixtures


def test_returns_simulation_trace_for_each_run(trace_fixtures):
    for sim_id, (tsv_path, _df, trace) in trace_fixtures.items():
        assert isinstance(trace, SimulationTrace)
        # sim_id is derived from the file stem.
        assert trace.sim_id == os.path.splitext(os.path.basename(tsv_path))[0]


def test_time_axis_matches_file(trace_fixtures):
    for _sim_id, (_tsv_path, df, trace) in trace_fixtures.items():
        assert isinstance(trace.time, pd.DatetimeIndex)
        assert len(trace.time) == len(df)
        expected = pd.DatetimeIndex(pd.to_datetime(df.index))
        assert trace.time.equals(expected)


def test_every_series_matches_source_column_at_the_boundary(trace_fixtures):
    """The core AC #8 assertion: each trace series equals the on-disk column."""
    for _sim_id, (tsv_path, _df, trace) in trace_fixtures.items():
        # Re-read the file independently so the comparison is against the file,
        # not the in-memory df the trace was built from.
        on_disk = pd.read_csv(tsv_path, sep="\t")
        assert len(trace.time) == len(on_disk)
        for field, column in FIELD_TO_COLUMN.items():
            np.testing.assert_allclose(
                getattr(trace, field).to_numpy(dtype=float),
                on_disk[column].to_numpy(dtype=float),
                equal_nan=True,
                err_msg="series '{}' diverged from column '{}'".format(field, column),
            )


def test_core_series_populated(trace_fixtures):
    """BG/IOB/basal are populated on every run regardless of controller."""
    for _sim_id, (_tsv_path, _df, trace) in trace_fixtures.items():
        assert trace.bg.notna().all()
        assert trace.iob.notna().any()
        assert trace.sbr.notna().any()


def test_loop_cob_tolerated_whether_populated_or_empty(trace_fixtures):
    """AC #4: reader tolerates loop_cob on both active and inactive stages."""
    populated_somewhere = False
    for _sim_id, (_tsv_path, _df, trace) in trace_fixtures.items():
        # Never raises regardless of sparsity; presence varies by stage.
        assert len(trace.loop_cob) == len(trace.time)
        if trace.loop_cob.notna().any():
            populated_somewhere = True
    # The median suite includes an active-Loop stage, so COB shows up somewhere.
    assert populated_somewhere, "expected loop_cob populated on the active-Loop stage"


def test_no_prediction_field_on_trace(trace_fixtures):
    """AC #5: no prediction/forecast attribute is exposed on the trace object."""
    _sim_id, (_tsv_path, _df, trace) = next(iter(trace_fixtures.items()))
    attrs = set(vars(trace))
    assert not any(k in a for a in attrs for k in ("pred", "forecast"))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
