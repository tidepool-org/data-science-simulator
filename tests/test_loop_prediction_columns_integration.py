"""End-to-end integration test for TRSET-24/TRSET-25 Loop prediction/effect/COB columns.

Runs a real ``loop_risk_v2_0`` scenario config through the actual parser and the
real Swift ``.dylib`` (no mocks) and asserts on the resulting ``get_results_df()``
dataframe at the system boundary. The harness config is the ``TLR-000-swift``
median suite, whose three stages exercise both Loop-active (SwiftLoopController)
and Loop-inactive (DoNothingController) paths in a single run.

TRSET-24 left ``loop_final_insulin_effect`` / ``loop_final_carb_effect`` /
``loop_final_momentum_effect`` / ``loop_final_rc_effect`` /
``loop_recommended_bolus_value`` unbacked (D1/D5 below); TRSET-25 backs all
five, so this module now asserts they populate on active Loop steps instead of
asserting they stay empty.

Run under the arm64 conda env ``tidepool-data-science-simulator`` (the one that
matches the committed .dylib). The module skips cleanly when the dylib isn't
importable, so the suite stays portable.
"""
import datetime
import os

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
from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG = os.path.join(
    REPO_ROOT,
    "scenario_configs/tidepool_risk_v2/loop_risk_v2_0/test/TLR-000-swift",
    "Simulation-Configuration-TLR-000-base_median_profile_v1.json",
)

RUN_HOURS = 1  # early-stop keeps the run near the ~1-min budget

# TRSET-25: these were unbacked under TRSET-24 (D1/D5) -- now sourced from the
# new getPredictionEffects bridge call (the four effect columns) and from the
# existing recommendation object's automatic/manual precedence (bolus value).
EFFECT_COLUMNS = [
    "loop_final_insulin_effect",
    "loop_final_carb_effect",
    "loop_final_momentum_effect",
    "loop_final_rc_effect",
]

# Columns the TRSET-24 feature populates from the Swift prediction API.
PREDICTION_COLUMNS = [
    "loop_final_glucose_pred",
    "loop_final_counteraction_effect",
    "loop_cob",
]


@pytest.fixture(scope="module")
def sim_results(tmp_path_factory):
    """Run the three-stage suite once (early-stopped) and return {sim_id: results_df}."""
    io_dir = str(tmp_path_factory.mktemp("loop_algo_io"))
    parser = ScenarioParserV2(path_to_json_config=CONFIG)
    sims = parser.get_sims()

    results = {}
    for sim_id, sim in sims.items():
        if hasattr(sim.controller, "loop_algo_io_dir"):
            sim.controller.loop_algo_io_dir = io_dir
        stop = sim.start_time + datetime.timedelta(hours=RUN_HOURS)
        sim.run(early_stop_datetime=stop)
        results[sim_id] = (sim, sim.get_results_df())
    return results


def _loop_sim(sim_results):
    """The post-Loop (SwiftLoopController) stage."""
    for sim_id, (sim, df) in sim_results.items():
        if isinstance(sim.controller, SwiftLoopController) and "post-Loop" in sim_id:
            return df
    pytest.fail("No post-Loop SwiftLoopController stage found in the suite")


def _nonloop_sim(sim_results):
    """The no-Loop (DoNothingController) stage."""
    for sim_id, (sim, df) in sim_results.items():
        if not isinstance(sim.controller, SwiftLoopController):
            return df
    pytest.fail("No non-Swift (DoNothing) stage found in the suite")


# --- AC#1 / AC#5: the TRSET-21 blocker is cleared ----------------------------

def test_glucose_pred_populated_on_active_loop_steps(sim_results):
    df = _loop_sim(sim_results)
    active = df[df["active"] == 1]
    assert len(active) > 0
    pred = active["loop_final_glucose_pred"]
    # Every active Loop step has a predicted glucose value...
    assert pred.notna().all(), "loop_final_glucose_pred has gaps on active Loop steps"
    # ...and they are physiologically plausible mg/dL values.
    assert ((pred >= 10) & (pred <= 600)).all()


# --- AC#2: counteraction + COB populate --------------------------------------

def test_cob_and_counteraction_populated_on_active_loop_steps(sim_results):
    df = _loop_sim(sim_results)
    active = df[df["active"] == 1]
    assert active["loop_final_counteraction_effect"].notna().all()
    # COB is populated on every active step; 0.0 is legitimate, so also require
    # that carbs are genuinely on board at some step (config carries carb doses).
    assert active["loop_cob"].notna().all()
    assert (active["loop_cob"] > 0).any(), "expected COB > 0 at some active step"


# --- AC#3: no fabrication where Loop isn't automating ------------------------

def test_no_prediction_data_on_donothing_stage(sim_results):
    df = _nonloop_sim(sim_results)
    for col in PREDICTION_COLUMNS + EFFECT_COLUMNS + ["loop_recommended_bolus_value"]:
        assert df[col].isna().all(), f"{col} fabricated on a non-Loop stage"


# --- TRSET-25 AC#19: effect columns + bolus value populate on active Loop steps

def test_effect_columns_populated_on_active_loop_steps(sim_results):
    df = _loop_sim(sim_results)
    active = df[df["active"] == 1]
    for col in EFFECT_COLUMNS:
        assert active[col].notna().all(), f"{col} has gaps on active Loop steps"


def test_bolus_value_matches_automatic_or_manual_precedence(sim_results):
    """Per-row check: automatic wins when present (0.0 counts), else manual,
    else None. The TLR-000-swift harness runs tempBasal-only mode, so it never
    actually exercises a populated bolus -- the precedence unit tests in
    tests/test_loop_effect_columns.py cover the populated cases directly."""
    df = _loop_sim(sim_results)
    active = df[df["active"] == 1]
    for _, row in active.iterrows():
        auto = row["loop_automatic_bolus_rec"]
        manual = row["loop_manual_bolus_rec"]
        expected = auto if pd.notna(auto) else (manual if pd.notna(manual) else None)
        actual = row["loop_recommended_bolus_value"]
        if expected is None:
            assert pd.isna(actual)
        else:
            assert actual == expected


# --- AC#6: existing populated columns unchanged ------------------------------

def test_existing_columns_still_populated(sim_results):
    df = _loop_sim(sim_results)
    active = df[df["active"] == 1]
    # Recommendation-backed temp basal columns still fill on active Swift steps.
    assert active["loop_recommended_temp_basal_value"].notna().all()
    assert active["loop_temp_basal_duration_sec"].notna().all()
    # Core patient columns unaffected.
    assert df["bg"].notna().all()
    assert active["iob"].notna().all()


# --- D2: loop_prediction_abs_error_* stays out of scope ----------------------

def test_abs_error_columns_remain_out_of_scope(sim_results):
    # Storing predictions on a new ControllerState field (not the recommendation
    # dict) must NOT quietly activate the horizon abs-error columns.
    df = _loop_sim(sim_results)
    abs_err_cols = [c for c in df.columns if c.startswith("loop_prediction_abs_error")]
    for col in abs_err_cols:
        assert df[col].isna().all(), f"{col} activated -- D2 scope boundary breached"
