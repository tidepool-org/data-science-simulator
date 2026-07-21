"""
Phase 3 Streamlit MVP integration tests.

Per the approved plan (CodeBot_Base/Phase 3 - Streamlit MVP Integration Test
Plan.md): drives the real streamlit_app.py via streamlit.testing.v1.AppTest,
against real scenario_configs/ data, no mocks in the exercised path.
Assertions target the structured result objects / GUI state at the
select-config -> run -> render boundary, not internal function calls.

Real, non-PHI fixtures used:
- Happy path: a single real directory from
  scenario_configs/tidepool_risk_v2/loop_risk_v2_0/test/ (TLR-QAE-482-test).
- Multi-TLR / cancel: NOT the whole "test" collection -- it has a pre-existing
  broken config (TLR-000-base's t2_resistant profile references a reusable
  file under reusable/simulations/versions/, a subdirectory ScenarioParserV2's
  load_pointer doesn't search for "simulations" references; found while
  writing this suite, flagged separately, not fixed here). A dedicated
  2-directory fixture collection (both copies of the known-good
  TLR-QAE-482-test config) is used instead.
- Warnings / errors / no-data: no real config in the library happens to
  trigger these (confirmed by grep before writing this), so a temp sibling
  collection is created under the same library root (so reference
  resolution to reusable/ works exactly as it does for real collections)
  with one deliberately-crafted TLR-* directory each.

All synthetic fixture collections are created fresh and torn down after the
module's tests run.
"""

import json
import os
import shutil
import time

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY_ROOT = os.path.join(REPO_ROOT, "scenario_configs", "tidepool_risk_v2", "loop_risk_v2_0")
REAL_COLLECTION = "test"  # scenario_configs/.../loop_risk_v2_0/test -- 8 real TLR-* dirs

FIXTURES_COLLECTION_NAME = "_pytest_phase3_integration_fixtures"
FIXTURES_COLLECTION_DIR = os.path.join(LIBRARY_ROOT, FIXTURES_COLLECTION_NAME)

MULTI_COLLECTION_NAME = "_pytest_phase3_integration_multi"
MULTI_COLLECTION_DIR = os.path.join(LIBRARY_ROOT, MULTI_COLLECTION_NAME)

_BASE_CONFIG_PATH = os.path.join(
    LIBRARY_ROOT, REAL_COLLECTION, "TLR-QAE-482-test",
    "Simulation-Configuration-TLR-QAE-482-test_median_v1.json",
)


@pytest.fixture(scope="module", autouse=True)
def synthetic_fixtures():
    """Create the warn/error/no-data TLR-* dirs as a sibling collection of
    real ones (so _find_pointer_object_dir resolves the real reusable/ dir
    two levels up, exactly as for any real collection), torn down after."""
    with open(_BASE_CONFIG_PATH) as fh:
        base_config = json.load(fh)

    warn_dir = os.path.join(FIXTURES_COLLECTION_DIR, "TLR-WARN-TEST")
    err_dir = os.path.join(FIXTURES_COLLECTION_DIR, "TLR-ERR-TEST")
    nodata_dir = os.path.join(FIXTURES_COLLECTION_DIR, "TLR-NODATA-TEST")
    os.makedirs(warn_dir, exist_ok=True)
    os.makedirs(err_dir, exist_ok=True)
    os.makedirs(nodata_dir, exist_ok=True)

    warn_config = json.loads(json.dumps(base_config))
    warn_config["override_config"][1]["controller"]["max_active_insulin_multiplier"] = 3.0
    with open(os.path.join(warn_dir, "Simulation-Configuration-TLR-WARN-test_median_v1.json"), "w") as fh:
        json.dump(warn_config, fh, indent=2)

    err_config = json.loads(json.dumps(base_config))
    err_config["base_config"] = "reusable.simulations.this_file_does_not_exist_xyz"
    with open(os.path.join(err_dir, "Simulation-Configuration-TLR-ERR-test_median_v1.json"), "w") as fh:
        json.dump(err_config, fh, indent=2)

    # Filename deliberately doesn't start with "Simulation-Configuration-TLR" --
    # run_simulations names its output CSV "summary_results_<scenario_json_name>.csv",
    # and build_assessment only globs "summary_results_Simulation-Configuration-TLR*.csv",
    # so this real naming-convention mismatch reproduces the no-usable-data path for real
    # (this is exactly the kind of silent-drop risk the design doc's Phase 2 findings flagged).
    nodata_config = json.loads(json.dumps(base_config))
    with open(os.path.join(nodata_dir, "unconventional_name_scenario.json"), "w") as fh:
        json.dump(nodata_config, fh, indent=2)

    # Dedicated clean 2-directory collection for multi-TLR/cancel -- deliberately
    # NOT the real "test" collection, which has an unrelated pre-existing broken
    # config (see module docstring) that would fail every "Run all" on it.
    # cancel_event is only rechecked once per scenario FILE (gui_runner.py's loop),
    # while progress only fires once per completed DIRECTORY -- so each directory
    # gets 3 duplicate scenario files (not 1), giving a real multi-second window
    # during the second directory's processing for a cancel to land after the
    # first directory's progress has already fired, instead of a race decided in
    # a single Python statement gap.
    multi_dir_a = os.path.join(MULTI_COLLECTION_DIR, "TLR-MULTI-A")
    multi_dir_b = os.path.join(MULTI_COLLECTION_DIR, "TLR-MULTI-B")
    os.makedirs(multi_dir_a, exist_ok=True)
    os.makedirs(multi_dir_b, exist_ok=True)
    for d, name in [(multi_dir_a, "TLR-MULTI-A"), (multi_dir_b, "TLR-MULTI-B")]:
        for i in range(3):
            config = json.loads(json.dumps(base_config))
            with open(os.path.join(d, f"Simulation-Configuration-{name}_median_v1_copy{i}.json"), "w") as fh:
                json.dump(config, fh, indent=2)

    yield

    shutil.rmtree(FIXTURES_COLLECTION_DIR, ignore_errors=True)
    shutil.rmtree(MULTI_COLLECTION_DIR, ignore_errors=True)


def _select_collection(at, collection_name):
    at.selectbox[0].select(collection_name).run()


def _select_single_directory(at, exact_dir_name):
    at.radio[0].set_value("One specific directory").run()
    tlr_selectbox = [sb for sb in at.selectbox if sb.label == "TLR-* directory"][0]
    tlr_selectbox.select(exact_dir_name).run()


def _click_run_and_wait(at, timeout=180):
    at.button[0].click().run()
    thread = at.session_state["run_thread"]
    assert thread is not None, "Run assessment did not start a background thread"
    thread.join(timeout=timeout)
    assert not thread.is_alive(), f"Run did not complete within {timeout}s"
    at.run()


# ---------------------------------------------------------------------------
# Case 1: happy path
# ---------------------------------------------------------------------------

def test_happy_path_all_three_stages_reach_the_gui_intact():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    _select_collection(at, REAL_COLLECTION)
    _select_single_directory(at, "TLR-QAE-482-test")

    assert not at.button[0].disabled, "Run should not be blocked -- this config has no validation errors"
    _click_run_and_wait(at)

    assert not at.exception
    result = at.session_state["run_result"]
    assert result is not None and result.cancelled is False
    assert [r.risk_dir_name for r in result.risk_dir_results] == ["TLR-QAE-482-test"]

    assessment = result.risk_dir_results[0].assessment
    assert assessment is not None
    assert set(assessment.stages.keys()) == {"pre", "no_loop", "post"}
    for stage_result in assessment.stages.values():
        assert stage_result.n_sims is not None  # the silent-drop-detection hook is populated

    # TWI-0006 constraint: both pre and no_loop reach the GUI and are selectable --
    # never auto-collapsed to one pre-mitigation figure.
    premit_selectbox = [sb for sb in at.selectbox if "pre-mitigation" in sb.label.lower()][0]
    assert set(premit_selectbox.options) == {"Pre-mitigation", "No Loop"}
    assert premit_selectbox.value is None


# ---------------------------------------------------------------------------
# Case 2: warnings surfaced pre-run, do not block
# ---------------------------------------------------------------------------

def test_warnings_surfaced_before_run_and_do_not_block_it():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    _select_collection(at, FIXTURES_COLLECTION_NAME)
    _select_single_directory(at, "TLR-WARN-TEST")

    assert not at.exception
    assert any("warning" in expander.label.lower() for expander in at.expander)
    assert not at.button[0].disabled


# ---------------------------------------------------------------------------
# Case 3: errors block the run
# ---------------------------------------------------------------------------

def test_errors_block_the_run():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    _select_collection(at, FIXTURES_COLLECTION_NAME)
    _select_single_directory(at, "TLR-ERR-TEST")

    assert not at.exception
    assert len(at.error) >= 1
    assert any("validation error" in e.value.lower() for e in at.error)
    assert at.button[0].disabled


# ---------------------------------------------------------------------------
# Case 5: no-usable-data surfaced explicitly, not dropped
# ---------------------------------------------------------------------------

def test_no_usable_data_surfaced_explicitly():
    at = AppTest.from_file("streamlit_app.py", default_timeout=30)
    at.run()
    _select_collection(at, FIXTURES_COLLECTION_NAME)
    _select_single_directory(at, "TLR-NODATA-TEST")

    assert not at.button[0].disabled
    _click_run_and_wait(at, timeout=60)

    assert not at.exception
    result = at.session_state["run_result"]
    assert len(result.risk_dir_results) == 1
    assert result.risk_dir_results[0].risk_dir_name == "TLR-NODATA-TEST"
    assert result.risk_dir_results[0].assessment is None
    assert any("no usable data" in w.value.lower() for w in at.warning)


# ---------------------------------------------------------------------------
# Case 6: multi-TLR aggregation
# ---------------------------------------------------------------------------

def test_multi_tlr_aggregation_yields_one_assessment_per_directory():
    at = AppTest.from_file("streamlit_app.py", default_timeout=60)
    at.run()
    _select_collection(at, MULTI_COLLECTION_NAME)
    # scope_choice defaults to "All directories in this collection" -- no
    # second selectbox needed; target_risk_dir stays None.

    assert not at.button[0].disabled
    _click_run_and_wait(at, timeout=60)

    assert not at.exception
    result = at.session_state["run_result"]
    assert result.cancelled is False
    # order is filesystem-dependent (build_risk_sim_generator doesn't sort) -- compare as a set
    assert {r.risk_dir_name for r in result.risk_dir_results} == {"TLR-MULTI-A", "TLR-MULTI-B"}
    for r in result.risk_dir_results:
        assert r.assessment is not None


# ---------------------------------------------------------------------------
# Case 4: cancel mid-run
# ---------------------------------------------------------------------------

def test_cancel_mid_run_stops_before_completing_every_directory():
    at = AppTest.from_file("streamlit_app.py", default_timeout=60)
    at.run()
    _select_collection(at, MULTI_COLLECTION_NAME)
    # "All directories" scope -- 2 real dirs, enough headroom to cancel between them.

    at.button[0].click().run()
    thread = at.session_state["run_thread"]
    cancel_event = at.session_state["cancel_event"]
    assert thread is not None and cancel_event is not None

    # Wait for the first directory to finish (real progress, not a fixed sleep guess)
    # before cancelling, so this exercises a genuine "some done, more pending" cancellation.
    deadline = time.time() + 30
    while at.session_state["progress"] is None and time.time() < deadline:
        time.sleep(0.2)
    assert at.session_state["progress"] is not None, "No directory completed within 30s -- cannot test a mid-run cancel"

    cancel_event.set()
    thread.join(timeout=30)
    assert not thread.is_alive()
    at.run()

    assert not at.exception
    result = at.session_state["run_result"]
    assert result.cancelled is True
    assert len(result.risk_dir_results) == 1, (
        f"expected exactly 1 of 2 directories to complete before cancel, got {len(result.risk_dir_results)}"
    )
    # No partial/corrupt result: the completed entry has a real assessment, not something half-written.
    assert result.risk_dir_results[0].assessment is not None
