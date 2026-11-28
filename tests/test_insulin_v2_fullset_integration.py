"""
TRSET-84 integration tests: generate the three arms from 10 real risks of loop_risk_v2_2_0_full into a temp clone of
the scenario tree, and check verification by difference, failure modes, determinism and a Loop run.

Subset (one variant directory per risk), by role:
  deviation        TLR-1053 (fiasp controller), TLR-791 (rapid_acting_child controller)
  preset repoint   TLR-554 (12 override items repoint patient metabolism), TLR-1120 (same, on an activity-preset base)
  guardrail        TLR-899_1_125, TLR-1049_18
  plain            TLR-1005, TLR-1011 (no override touches patient metabolism or controller settings)
  structural       TLR-959 (suspension file in the chain), TLR-889_corr (sensitive500 swift base)

The end-to-end run needs the Loop Algorithm API and is local only: set TRSET84_E2E=1.
"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.insulin_v2_fullset import generate, manifest as mf, translation  # noqa: E402
from scripts.insulin_v2_fullset.resolve import (DEFAULT_BASELINE, NO_CONTROLLER, SCENARIO_ROOT, UNSET,  # noqa: E402
                                                resolve_config_file)

SUBSET = ["TLR-1053", "TLR-791", "TLR-554", "TLR-1120", "TLR-899_1_125", "TLR-1049_18",
          "TLR-1005", "TLR-1011", "TLR-959", "TLR-889_corr"]
ARMS = list(translation.ARMS)
QUIET = dict(exclude=[], sha="testsha", dirty=False, log=lambda *a, **k: None)


def make_world(root, tlrs):
    scen = Path(root) / "scenario"
    scen.mkdir(parents=True)
    try:
        subprocess.run(["cp", "-Rc", str(SCENARIO_ROOT / "reusable"), str(scen / "reusable")], check=True, capture_output=True)
    except Exception:
        shutil.copytree(SCENARIO_ROOT / "reusable", scen / "reusable")
    # The repo's reusable/ folder holds the committed generator output. Strip it (the committed manifests list every
    # file the generator created) so the sandbox starts from the baseline state the generator is meant to run on.
    for manifest in sorted(mf.MANIFEST_DIR.glob("*.json")):
        for entry in json.loads(manifest.read_text())["reusable_files"]:
            (scen / entry["path"]).unlink(missing_ok=True)
    base = scen / "loop_risk_v2_0" / "loop_risk_v2_2_0_full"
    for t in tlrs:
        shutil.copytree(DEFAULT_BASELINE / t, base / t)
    return scen, base


def tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(Path(root).rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def gen(scen, base, tlrs=None, **kw):
    return generate.run(scen, base, scen / "loop_risk_v2_0", tlrs=tlrs, **{**QUIET, **kw})


@pytest.fixture(scope="module")
def world(tmp_path_factory):
    scen, base = make_world(tmp_path_factory.mktemp("full"), SUBSET)
    reusable_before = {p: p.read_bytes() for p in (scen / "reusable").rglob("*") if p.is_file()}
    base_hash = tree_hash(base)
    gen(scen, base)
    return {"scen": scen, "base": base, "out": scen / "loop_risk_v2_0", "reusable_before": reusable_before,
            "base_hash": base_hash}


def sims(world, arm_or_base):
    """{(tlr, file): resolved sim records} for the baseline subset or an arm."""
    root = world["base"] if arm_or_base == "baseline" else world["out"] / arm_or_base
    return {(d.name, c.name): resolve_config_file(c, world["scen"])
            for d in sorted(root.iterdir()) if d.is_dir() for c in sorted(d.glob("*.json"))}


def expected_model(arm, baseline_model):
    if arm == "loop_risk_v2_rai_main":
        return baseline_model
    return "fiasp" if baseline_model == "rapid_acting_adult" else baseline_model


EXPECTED_PATIENT = {"loop_risk_v2_rai_main": ("rapid_acting_v1", "main"),
                    "loop_risk_v2_urai_main": ("ultra_rapid_v1", "main"),
                    "loop_risk_v2_urai_old": ("fiasp", UNSET)}


# ---------------------------------------------------------------- layout, scope, read-only
def test_three_arm_directories_each_hold_all_ten_risks_and_a_manifest(world):
    for arm in ARMS:
        assert sorted(p.name for p in (world["out"] / arm).iterdir() if p.is_dir()) == sorted(SUBSET)
        assert (mf.manifest_path(world["out"] / "_manifests", arm)).is_file()


def test_baseline_and_existing_reusable_files_are_unchanged(world):
    assert tree_hash(world["base"]) == world["base_hash"]
    for p, content in world["reusable_before"].items():
        assert p.read_bytes() == content, p


def test_no_algorithm_io_files_are_generated(world):
    assert not list(world["scen"].rglob("loop_algo_*_*.json"))


# ---------------------------------------------------------------- verification by difference (independent check)
@pytest.mark.parametrize("arm", ARMS)
def test_every_generated_sim_differs_from_baseline_only_in_the_three_fields(world, arm):
    base, gen_ = sims(world, "baseline"), sims(world, arm)
    assert base.keys() == gen_.keys()
    ptype, timing = EXPECTED_PATIENT[arm]
    n = 0
    for key in base:
        assert [r["sim_id"] for r in base[key]] == [r["sim_id"] for r in gen_[key]]
        for b, g in zip(base[key], gen_[key]):
            n += 1
            assert (g["patient_insulin_type"], g["insulin_timing"]) == (ptype, timing), (arm, key, g["sim_id"])
            assert g["controller_model"] == expected_model(arm, b["controller_model"]), (arm, key, g["sim_id"])
            br, gr = copy.deepcopy(b["resolved"]), copy.deepcopy(g["resolved"])
            for r in (br, gr):
                ms = r["patient"]["patient_model"]["metabolism_settings"]
                ms.pop("patient_insulin_type", None), ms.pop("insulin_timing", None)
                if r.get("controller"):
                    r["controller"]["settings"].pop("model", None)
            assert br == gr, (arm, key, g["sim_id"])   # includes the whole pump side
    assert n >= 100


@pytest.mark.parametrize("arm", ARMS)
def test_pump_side_metabolism_is_untouched(world, arm):
    base, gen_ = sims(world, "baseline"), sims(world, arm)
    for key in base:
        for b, g in zip(base[key], gen_[key]):
            assert b["resolved"]["patient"]["pump"] == g["resolved"]["patient"]["pump"]


def test_deviation_sims_keep_their_baseline_controller(world):
    base = sims(world, "baseline")
    for arm in ARMS:
        gen_ = sims(world, arm)
        for key, recs in base.items():
            if key[0] in ("TLR-1053", "TLR-791"):
                for b, g in zip(recs, gen_[key]):
                    if b["controller_model"] in ("fiasp", "rapid_acting_child"):
                        assert g["controller_model"] == b["controller_model"], (arm, key, b["sim_id"])
    models = {r["controller_model"] for recs in base.values() for r in recs}
    assert {"fiasp", "rapid_acting_child"} <= models    # the subset really contains both deviations


def test_guardrail_override_keeps_fiasp_through_the_leaf_merge(world):
    base, gen_ = sims(world, "baseline"), sims(world, "loop_risk_v2_urai_main")
    checked = 0
    for key in base:
        if key[0] in ("TLR-899_1_125", "TLR-1049_18"):
            for b, g in zip(base[key], gen_[key]):
                if b["override_sets_controller_settings"] and b["controller_model"] == "rapid_acting_adult":
                    assert g["controller_model"] == "fiasp"
                    for k in ("suspend_threshold", "max_basal_rate", "max_bolus"):
                        assert g["resolved"]["controller"]["settings"][k] == b["resolved"]["controller"]["settings"][k]
                    checked += 1
    assert checked >= 2


def test_preset_repoint_does_not_reset_the_insulin_type(world):
    base, gen_ = sims(world, "baseline"), sims(world, "loop_risk_v2_rai_main")
    repointed = [(k, i) for k, recs in base.items() if k[0] in ("TLR-554", "TLR-1120")
                 for i, r in enumerate(recs) if r["override_sets_patient_metabolism"]]
    assert len(repointed) >= 20
    for k, i in repointed:
        assert gen_[k][i]["patient_insulin_type"] == "rapid_acting_v1" and gen_[k][i]["insulin_timing"] == "main"


def test_generated_bases_and_profiles_contain_insulin_timing_explicitly(world):
    for arm in ("loop_risk_v2_rai_main", "loop_risk_v2_urai_main"):
        files = [json.loads(Path(p["path"] if False else world["scen"] / p["path"]).read_text())
                 for p in json.loads((mf.manifest_path(world["out"] / "_manifests", arm)).read_text())["reusable_files"]
                 if "metabolism_settings" in p["path"]]
        assert files and all(f.get("insulin_timing") == "main" for f in files)


# ---------------------------------------------------------------- names, exclude, manifest
def test_new_reusable_names_do_not_collide_with_existing_files(world):
    for arm in ARMS:
        for entry in json.loads((mf.manifest_path(world["out"] / "_manifests", arm)).read_text())["reusable_files"]:
            p = Path(entry["path"])
            assert p.stem.endswith(("_" + arm.replace("loop_risk_v2_", ""),)) or "_" + arm.replace("loop_risk_v2_", "") in p.stem
            assert (world["scen"] / p) not in world["reusable_before"]


def test_manifest_records_provenance_and_every_created_reusable_file(world):
    for arm in ARMS:
        m = json.loads((mf.manifest_path(world["out"] / "_manifests", arm)).read_text())
        assert m["generator_commit"] == "testsha" and m["generator_dirty"] is False
        assert m["baseline_tree_hash"] == mf.tree_hash([world["base"] / t for t in sorted(SUBSET)])
        assert m["translation_table_hash"] == translation.table_hash(translation.load_table())
        assert m["exclude_list"] == [] and sorted(m["tlr_dirs"]) == sorted(SUBSET)
        assert m["reusable_files"]
        for e in m["reusable_files"]:
            assert hashlib.sha256((world["scen"] / e["path"]).read_bytes()).hexdigest() == e["sha256"]


def test_excluded_tlr_is_absent_from_every_arm(tmp_path):
    scen, base = make_world(tmp_path, ["TLR-1005", "TLR-549"])
    generate.run(scen, base, scen / "loop_risk_v2_0", **{**QUIET, "exclude": ["TLR-549"]})
    for arm in ARMS:
        assert [p.name for p in (scen / "loop_risk_v2_0" / arm).iterdir() if p.is_dir()] == ["TLR-1005"]


# ---------------------------------------------------------------- failure modes
def test_unmapped_pair_fails_and_writes_nothing(tmp_path):
    scen, base = make_world(tmp_path, ["TLR-791", "TLR-1005"])
    table = translation.load_table()
    table["rows"] = [r for r in table["rows"] if r["key"] != "rapid_acting_adult | rapid_acting_child"]
    before = tree_hash(scen)
    with pytest.raises(generate.UnmappedPairError, match="rapid_acting_child"):
        generate.run(scen, base, scen / "loop_risk_v2_0", table=table, **QUIET)
    assert tree_hash(scen) == before
    assert not any((scen / "loop_risk_v2_0" / a).exists() for a in ARMS)


def test_unresolved_table_row_fails_and_writes_nothing(tmp_path):
    scen, base = make_world(tmp_path, ["TLR-1005"])
    table = translation.load_table()
    table["rows"][0]["status"] = "NEEDS_REVIEW"
    before = tree_hash(scen)
    with pytest.raises(generate.UnmappedPairError):
        generate.run(scen, base, scen / "loop_risk_v2_0", table=table, **QUIET)
    assert tree_hash(scen) == before


def test_overwriting_a_differing_file_needs_force(tmp_path):
    scen, base = make_world(tmp_path, ["TLR-1005"])
    gen(scen, base)
    victim = sorted(p for p in (scen / "reusable").rglob("*_rai_main.json"))[0]
    good = victim.read_text()
    victim.write_text(good + "\n ")
    with pytest.raises(generate.OverwriteError):
        gen(scen, base)
    assert victim.read_text() == good + "\n "        # untouched by the failed run
    gen(scen, base, force=True)
    assert victim.read_text() == good


def test_name_collision_in_the_search_path_fails_and_writes_nothing(tmp_path):
    scen, base = make_world(tmp_path, ["TLR-1005"])
    (scen / "reusable/metabolism_settings/presets/median_v1_rai_main.json").write_text("{}")  # shadows the profile copy
    before = tree_hash(scen)
    with pytest.raises(generate.GenerationError, match="share a name"):
        gen(scen, base)
    assert tree_hash(scen) == before


# ---------------------------------------------------------------- determinism
def test_second_run_is_byte_identical_and_writes_nothing(tmp_path):
    scen, base = make_world(tmp_path, ["TLR-1005", "TLR-1053"])
    logs = []
    generate.run(scen, base, scen / "loop_risk_v2_0", **{**QUIET, "log": logs.append})
    first = tree_hash(scen)
    logs.clear()
    generate.run(scen, base, scen / "loop_risk_v2_0", **{**QUIET, "log": logs.append})
    assert tree_hash(scen) == first
    assert any("wrote 0 files" in str(m) for m in logs)


# ---------------------------------------------------------------- end to end (local only)
try:
    import loop_to_python_api.api  # noqa: F401
    _DYLIB_ERR = ""
except Exception as e:  # pragma: no cover - env dependent
    _DYLIB_ERR = repr(e)


@pytest.mark.skipif(os.environ.get("TRSET84_E2E") != "1" or bool(_DYLIB_ERR),
                    reason="local only: set TRSET84_E2E=1 with the Loop Algorithm API available " + _DYLIB_ERR)
@pytest.mark.parametrize("arm", ARMS)
def test_one_sim_per_arm_runs_to_completion_through_loop(world, arm, tmp_path):
    import datetime
    from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

    # TLR-1053 is the deviation sim (controller fiasp) that runs through Loop. The baseline TLR-791 sims
    # (controller rapid_acting_child) crash the Swift library even unmodified (Trace/BPT trap), so they
    # cannot serve as the end-to-end sim.
    cfg = next((world["out"] / arm / "TLR-1053").glob("*Adolescent_profile.json"))
    sims_ = ScenarioParserV2(str(cfg), pointer_object_dir=str(world["scen"])).get_sims()
    sim = next(s for sid, s in sims_.items() if "post-Loop" in sid)
    if hasattr(sim.controller, "loop_algo_io_dir"):
        sim.controller.loop_algo_io_dir = str(tmp_path)
    sim.run()
    df = sim.get_results_df()
    assert len(df) > 0
    assert df.index[-1] >= sim.start_time + datetime.timedelta(hours=7)
