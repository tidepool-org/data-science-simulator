"""TRSET-84 Phase 1 unit tests: read-only inventory and translation table (synthetic mini-baseline)."""
import hashlib
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.insulin_v2_fullset import inventory, resolve, translation  # noqa: E402
from scripts.insulin_v2_fullset.resolve import NO_CONTROLLER, UNSET  # noqa: E402


def _write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _tree_hash(root):
    h = hashlib.sha256()
    for p in sorted(root.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(root)).encode())
            h.update(p.read_bytes())
    return h.hexdigest()


@pytest.fixture
def world(tmp_path):
    """A scenario root (pointer_object_dir) and a baseline dir with four TLR dirs plus one excluded dir."""
    root = tmp_path / "scenario"
    r = root / "reusable"
    _write(r / "metabolism_settings/profiles/prof_plain.json", {"patient_insulin_type": "rapid_acting_adult", "x": 1})
    _write(r / "metabolism_settings/profiles/prof_noset.json", {"x": 1})
    _write(r / "metabolism_settings/presets/preset_fiasp.json", {"patient_insulin_type": "fiasp"})
    _write(r / "loop_settings/ctrl_plain.json", {"model": "rapid_acting_adult", "max_basal_rate": 5})
    _write(r / "loop_settings/ctrl_fiasp.json", {"model": "fiasp", "max_basal_rate": 5})
    _write(r / "mitigations/guardrail.json", {"max_basal_rate": 3})
    _write(r / "simulations/base/base_plain.json", _base("prof_plain", "ctrl_plain"))
    _write(r / "simulations/base/base_noset.json", _base("prof_noset", "ctrl_fiasp"))

    base = root / "baseline"
    _cfg(base / "TLR-1_plain", "reusable.simulations.base_plain", [{"sim_id": "s1"}])
    _cfg(base / "TLR-2_repoint", "reusable.simulations.base_plain", [
        {"sim_id": "s2", "patient": {"patient_model": {"metabolism_settings": "reusable.metabolism_settings.preset_fiasp"}}}])
    _cfg(base / "TLR-3_guardrail", "reusable.simulations.base_noset", [
        {"sim_id": "s3", "controller": {"settings": "reusable.mitigations.guardrail"}}])
    _cfg(base / "TLR-4_nocontroller", "reusable.simulations.base_plain", [{"sim_id": "s4", "controller": None}])
    _cfg(base / "TLR-799_excluded", "reusable.simulations.base_plain", [{"sim_id": "s5"}])
    return root, base


def _base(profile, ctrl):
    return {
        "sim_id": "base", "duration_hours": 8.0,
        "patient": {"pump": {"metabolism_settings": f"reusable.metabolism_settings.{profile}"},
                    "patient_model": {"metabolism_settings": f"reusable.metabolism_settings.{profile}"}},
        "controller": {"id": "swift", "settings": f"reusable.loop_settings.{ctrl}"},
    }


def _cfg(d, base_pointer, overrides):
    _write(d / "cfg.json", {"metadata": {"risk-id": d.name}, "base_config": base_pointer, "override_config": overrides})


def _run(world):
    root, base = world
    return resolve.resolve_baseline(base, ["TLR-799_excluded"], pointer_object_dir=root)


def test_pairs_for_plain_repoint_guardrail_and_nocontroller(world):
    sims, errors = _run(world)
    assert errors == []
    by_id = {s["sim_id"]: s for s in sims}
    assert (by_id["s1"]["patient_insulin_type"], by_id["s1"]["controller_model"]) == ("rapid_acting_adult", "rapid_acting_adult")
    # preset repoint resets the patient insulin type through the leaf merge
    assert by_id["s2"]["patient_insulin_type"] == "fiasp"
    assert by_id["s2"]["override_sets_patient_metabolism"] is True
    # guardrail override keeps the base-level controller model; patient type is unset in this profile
    assert by_id["s3"]["controller_model"] == "fiasp"
    assert by_id["s3"]["patient_insulin_type"] == UNSET
    assert by_id["s3"]["override_sets_controller_settings"] is True
    assert by_id["s4"]["controller_model"] == NO_CONTROLLER


def test_unset_insulin_timing_is_reported_not_defaulted(world):
    sims, _ = _run(world)
    assert {s["insulin_timing"] for s in sims} == {UNSET}


def test_exclude_list_is_honoured(world):
    sims, _ = _run(world)
    assert "TLR-799_excluded" not in {s["tlr"] for s in sims}
    assert len(sims) == 4


def test_exclude_entry_matching_no_directory_fails_loudly(world):
    root, base = world
    with pytest.raises(ValueError, match="TLR_799_excluded"):
        resolve.resolve_baseline(base, ["TLR_799_excluded"], pointer_object_dir=root)


def test_unresolvable_config_is_reported_and_does_not_abort(world):
    root, base = world
    _cfg(base / "TLR-6_broken", "reusable.simulations.does_not_exist", [{"sim_id": "s6"}])
    sims, errors = resolve.resolve_baseline(base, ["TLR-799_excluded"], pointer_object_dir=root)
    assert len(sims) == 4
    assert [e["tlr"] for e in errors] == ["TLR-6_broken"]
    assert "does_not_exist" in errors[0]["error"]


def test_inventory_groups_pairs_with_tlrs_and_sim_ids(world):
    sims, errors = _run(world)
    report = inventory.build_report(sims, errors, ["TLR-799_excluded"])
    keys = {p["key"] for p in report["pairs"]}
    assert "rapid_acting_adult | rapid_acting_adult" in keys
    plain = next(p for p in report["pairs"] if p["key"] == "rapid_acting_adult | rapid_acting_adult")
    assert plain["tlrs"] == ["TLR-1_plain"] and plain["sim_ids"] == ["TLR-1_plain/s1"]
    assert report["n_sims"] == 4 and report["errors"] == []


def test_inventory_is_read_only(world, tmp_path):
    root, base = world
    before = _tree_hash(root)
    sims, errors = resolve.resolve_baseline(base, ["TLR-799_excluded"], pointer_object_dir=root)
    inventory.build_report(sims, errors, [])
    assert _tree_hash(root) == before


def test_nominal_rows_cover_all_three_arms():
    t = translation.build_table([("rapid_acting_adult", "rapid_acting_adult")])
    row = t["rows"][0]
    assert row["status"] == "NOMINAL" and set(row["arms"]) == set(translation.ARMS) and len(row["arms"]) == 3
    a = row["arms"]
    assert set(a) == {"loop_risk_v2_rai_main", "loop_risk_v2_urai_main", "loop_risk_v2_urai_old"}
    assert a["loop_risk_v2_rai_main"] == {"patient_insulin_type": "rapid_acting_v1", "insulin_timing": "main",
                                          "controller_model": translation.CONTROLLER_UNCHANGED}
    assert a["loop_risk_v2_urai_main"] == {"patient_insulin_type": "ultra_rapid_v1", "insulin_timing": "main", "controller_model": "fiasp"}
    assert a["loop_risk_v2_urai_old"] == {"patient_insulin_type": "fiasp", "insulin_timing": None, "controller_model": "fiasp"}


def test_no_controller_rows_never_set_a_controller():
    t = translation.build_table([("rapid_acting_adult", NO_CONTROLLER)])
    assert all(c["controller_model"] == translation.CONTROLLER_UNCHANGED for c in t["rows"][0]["arms"].values())


def test_non_plain_pairs_need_review_and_block_phase_2():
    t = translation.build_table([("fiasp", "fiasp"), ("rapid_acting_adult", "fiasp"), (UNSET, "rapid_acting_adult")])
    assert {r["status"] for r in t["rows"]} == {"NEEDS_REVIEW"}
    assert all(r["arms"] == {} for r in t["rows"])
    assert len(translation.unresolved_rows(t)) == 3


def test_rebuild_keeps_hand_filled_rows():
    first = translation.build_table([("fiasp", "fiasp")])
    first["rows"][0].update(status="FILLED", arms=translation.nominal_arms("fiasp"))
    again = translation.build_table([("fiasp", "fiasp"), ("rapid_acting_adult", "rapid_acting_adult")], existing=first)
    kept = next(r for r in again["rows"] if r["key"] == "fiasp | fiasp")
    assert kept["status"] == "FILLED"
    assert translation.unresolved_rows(again) == []


def test_table_hash_is_stable():
    t = translation.build_table([("rapid_acting_adult", "rapid_acting_adult")])
    assert translation.table_hash(t) == translation.table_hash(json.loads(json.dumps(t)))
