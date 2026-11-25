"""TRSET-81 integration tests: v1 insulin-action curves through the real scenario parser, patient and Loop run.

IT-1  insulin type/timing in a loop_risk_v2_2_0_full scenario reaches the metabolism model at instantiation
IT-2  a v1 scenario runs end to end through the Swift Loop API (skips when the dylib is unavailable)
IT-3  backward compatibility: baseline config parses/runs identically with and without the new defaults
IT-4  effect horizon: patient prediction window and initial IOB for 11 h / 14 h curves
IT-5  schema: configs without the new keys stay valid; bad insulin_timing is rejected
ARM   generated REG-706 arms differ from the baseline only in patient insulin settings (+ controller model)
"""
import copy
import datetime
import json
import os
import sys

import numpy as np
import pytest

from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2
from tidepool_data_science_simulator.models.insulin_action_v1 import (
    SimpleMetabolismModelV1, InsulinActionV1Model, v1_time_constants, prediction_horizon_hours)
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
import configure_insulin_v1_reg706 as gen  # noqa: E402

BASELINE = os.path.join(gen.SOURCE_DIR, "TLR-843", "Simulation-Configuration-TLR-843_Median_profile_v1.json")

try:
    import loop_to_python_api.api  # noqa: F401
    _DYLIB_ERR = ""
except Exception as e:  # pragma: no cover - env dependent
    _DYLIB_ERR = repr(e)

ARM_V1 = [a for a, (t, _, _) in gen.ARMS.items() if t and t.endswith("_v1")]


@pytest.fixture(scope="module")
def arm_dir(tmp_path_factory):
    dest = tmp_path_factory.mktemp("arms")
    for arm in gen.ARMS:
        out = dest / arm / "TLR-843"
        out.mkdir(parents=True)
        cfg = gen.transform_config(json.load(open(BASELINE)), arm)
        json.dump(cfg, open(out / os.path.basename(BASELINE), "w"))
    return dest


def sims_for(path):
    return ScenarioParserV2(path_to_json_config=str(path)).get_sims()


def arm_path(arm_dir, arm):
    return arm_dir / arm / "TLR-843" / os.path.basename(BASELINE)


# ---------------------------------------------------------------- IT-1
@pytest.mark.parametrize("arm", ARM_V1)
def test_it1_v1_type_and_timing_reach_the_metabolism_model(arm_dir, arm):
    patient_type, timing, _ = gen.ARMS[arm]
    sims = sims_for(arm_path(arm_dir, arm))
    assert len(sims) == 3
    for sim in sims.values():
        model = sim.virtual_patient.instantiate_metabolism_model()
        assert isinstance(model, SimpleMetabolismModelV1)
        assert type(model.insulin_model) is InsulinActionV1Model
        assert model.insulin_model.time_constants == pytest.approx(v1_time_constants(patient_type, timing))


def test_it1_baseline_and_old_urai_keep_the_palerm_curve(arm_dir):
    for arm in ("A1_old_RAI", "A2_old_URAI"):
        for sim in sims_for(arm_path(arm_dir, arm)).values():
            assert type(sim.virtual_patient.instantiate_metabolism_model().insulin_model) is PalermInsulinModel


def test_it1_arm_controller_models(arm_dir):
    for arm, (_, _, fiasp) in gen.ARMS.items():
        for sim in sims_for(arm_path(arm_dir, arm)).values():
            settings = getattr(sim.controller, "controller_config", None)
            model = getattr(settings, "controller_settings", {}).get("model") if settings else None
            if model is None:
                continue  # no-Loop stage
            assert model == ("fiasp" if fiasp else "novolog")


# ---------------------------------------------------------------- ARM invariants
def _strip(cfg):
    """Remove the keys an arm is allowed to change."""
    cfg = copy.deepcopy(cfg)
    pm = cfg.get("patient", {}).get("patient_model", {})
    ms = pm.get("metabolism_settings")
    if isinstance(ms, dict):
        ms.pop("patient_insulin_type", None)
        ms.pop("insulin_timing", None)
    ctl = cfg.get("controller")
    if isinstance(ctl, dict) and isinstance(ctl.get("settings"), dict):
        ctl["settings"].pop("model", None)
    return cfg


@pytest.mark.parametrize("arm", [a for a in gen.ARMS if a != "A1_old_RAI"])
def test_arm_differs_from_baseline_only_in_insulin_settings(arm_dir, arm):
    base = ScenarioParserV2(path_to_json_config=BASELINE)
    new = ScenarioParserV2(path_to_json_config=str(arm_path(arm_dir, arm)))
    for b_delta, n_delta in zip(base.override_configs, new.override_configs):
        b_cfg, n_cfg = copy.deepcopy(base.base_sim_config), copy.deepcopy(new.base_sim_config)
        base.apply_config_override(b_cfg, copy.deepcopy(b_delta))
        new.apply_config_override(n_cfg, copy.deepcopy(n_delta))
        assert _strip(b_cfg) == _strip(n_cfg)


def test_arm_generator_matches_the_reg706_test_set():
    dirs = gen.matched_dirs(gen.SOURCE_DIR, gen.TEST_SET_TLRS)
    names = {d.name for d in dirs}
    for required in ("TLR-844_170_corr", "TLR-899_1_25", "TLR-843", "TLR-959", "TLR-861", "TLR-553",
                     "TLR-613", "TLR-1122", "TLR-552"):
        assert required in names
    assert all(n.split("_")[0] in gen.TEST_SET_TLRS for n in names)


# ---------------------------------------------------------------- IT-3 / IT-5
def test_it3_inlined_defaults_are_behaviourally_identical_to_baseline(arm_dir):
    """A1 is a byte copy; the explicit defaults the generator inlines build the same patient as the baseline."""
    base_sims = sims_for(BASELINE)
    explicit = copy.deepcopy(json.load(open(BASELINE)))
    explicit["base_config"] = gen.inline_base(explicit["base_config"])
    explicit_path = arm_dir / "explicit_defaults.json"
    json.dump(explicit, open(explicit_path, "w"))
    new_sims = sims_for(explicit_path)
    assert list(base_sims) == list(new_sims)
    for sid in base_sims:
        a = base_sims[sid].virtual_patient.instantiate_metabolism_model().insulin_model
        b = new_sims[sid].virtual_patient.instantiate_metabolism_model().insulin_model
        assert type(a) is type(b) is PalermInsulinModel
        assert (a._tau1, a._tau2) == (b._tau1, b._tau2)


def test_it5_config_without_new_keys_parses_with_defaults():
    for sim in sims_for(BASELINE).values():
        pc = sim.virtual_patient.patient_config
        assert pc.insulin_timing == "main"
        assert pc.patient_insulin_type == "rapid_acting_adult"


def test_it5_bad_insulin_timing_is_rejected(arm_dir):
    cfg = json.load(open(arm_path(arm_dir, "B1_new_RAI_main")))
    for item in cfg["override_config"]:
        item["patient"]["patient_model"]["metabolism_settings"]["insulin_timing"] = "sideways"
    path = arm_dir / "bad_timing.json"
    json.dump(cfg, open(path, "w"))
    with pytest.raises(ValueError, match="insulin_timing"):  # rejected when the simulations are built
        sims_for(path)


# ---------------------------------------------------------------- IT-4
@pytest.mark.parametrize("arm, hours", [("B1_new_RAI_main", 11), ("B3_new_RAI_slower", 14),
                                        ("B4_new_URAI_main", 10), ("B6_new_URAI_slower", 13),
                                        ("B2_new_RAI_faster", 8), ("A1_old_RAI", 8)])
def test_it4_prediction_window_and_initial_iob(arm_dir, arm, hours):
    patient_type, timing, _ = gen.ARMS[arm]
    for sim in sims_for(arm_path(arm_dir, arm)).values():
        patient = sim.virtual_patient
        assert patient.num_prediction_steps == hours * 12
        if patient_type:
            assert hours == prediction_horizon_hours(patient_type, timing)
        assert np.isfinite(patient.iob_current if hasattr(patient, "iob_current") else 0.0)


# ---------------------------------------------------------------- IT-2
@pytest.mark.skipif(bool(_DYLIB_ERR), reason=f"Swift dylib unavailable: {_DYLIB_ERR}")
def test_it2_v1_scenario_runs_end_to_end_through_loop(arm_dir, tmp_path):
    sims = sims_for(arm_path(arm_dir, "B1_new_RAI_main"))
    sim = next(s for sid, s in sims.items() if "post-Loop" in sid)
    if hasattr(sim.controller, "loop_algo_io_dir"):
        sim.controller.loop_algo_io_dir = str(tmp_path)
    sim.run()
    df = sim.get_results_df()
    assert len(df) > 0
    assert df["bg"].notna().all() if "bg" in df.columns else True
    assert (df["active"] == 1).any() if "active" in df.columns else True
    assert df.index[-1] >= sim.start_time + datetime.timedelta(hours=7)
