"""TRSET-84 Phase 2 unit tests: span-aware JSON edits, pointer location, registry, verification helpers."""
import json
import os
import sys

import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

from scripts.insulin_v2_fullset import jsonspans as js  # noqa: E402
from scripts.insulin_v2_fullset import locate, rewrite, verify  # noqa: E402
from scripts.insulin_v2_fullset.resolve import DEFAULT_BASELINE, SCENARIO_ROOT, UNSET  # noqa: E402


# ---------------------------------------------------------------- jsonspans
def test_parse_spans_and_replace_preserves_every_other_byte():
    text = '{\n  "a": 1,\n  "p": {"q": "reusable.x.y",   "r": [1, 2]},\n"z":   "keep"\n}'
    root = js.parse(text)
    node = js.get(root, ["p", "q"])
    out = js.apply(text, [js.replace_edit(node, "reusable.x.y_arm")])
    assert out == text.replace('"reusable.x.y"', '"reusable.x.y_arm"')
    assert json.loads(out)["p"]["q"] == "reusable.x.y_arm"


def test_set_member_inserts_with_multiline_indent():
    text = '{\n  "a": 1,\n  "patient_insulin_type": "rapid_acting_adult"\n}'
    root = js.parse(text)
    out = js.apply(text, [js.set_member_edit(text, root, "insulin_timing", "main")])
    assert out == '{\n  "a": 1,\n  "patient_insulin_type": "rapid_acting_adult",\n  "insulin_timing": "main"\n}'


def test_set_member_matches_unindented_last_key_and_inline_objects():
    text = '{\n"a": 1,\n"patient_insulin_type": "x"\n}'
    out = js.apply(text, [js.set_member_edit(text, js.parse(text), "t", "main")])
    assert out == '{\n"a": 1,\n"patient_insulin_type": "x",\n"t": "main"\n}'
    inline = '{"sim_id": "s", "k": 1}'
    out = js.apply(inline, [js.set_member_edit(inline, js.parse(inline), "n", 2)])
    assert out == '{"sim_id": "s", "k": 1, "n": 2}'
    assert js.apply("{}", [js.set_member_edit("{}", js.parse("{}"), "n", 2)]) == '{"n": 2}'


def test_set_member_replaces_existing_value_in_place():
    text = '{"patient_insulin_type": "old", "z": 1}'
    out = js.apply(text, [js.set_member_edit(text, js.parse(text), "patient_insulin_type", "new")])
    assert out == '{"patient_insulin_type": "new", "z": 1}'


def test_parser_handles_escapes_null_and_numbers():
    text = '{"s": "a\\"b", "n": null, "x": -1.5e3, "l": [], "o": {}}'
    root = js.parse(text)
    assert js.get(root, ["s"]).value(text) == 'a"b'
    assert js.get(root, ["n"]).value(text) is None and js.get(root, ["x"]).value(text) == -1500.0


# ---------------------------------------------------------------- locate
def test_locate_agrees_with_the_parser_for_every_baseline_pointer():
    """Guards drift between locate.py and ScenarioParserV2.load_pointer on all pointers in the baseline."""
    from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

    parser = ScenarioParserV2()
    parser.pointer_object_dir = str(SCENARIO_ROOT)
    pointers = set()

    def walk(o):
        if isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
        elif isinstance(o, str) and parser.is_config_file_pointer(o) and "." in o:
            pointers.add(o)

    for p in DEFAULT_BASELINE.glob("*/*.json"):
        walk(json.loads(p.read_text()))
    checked = 0
    for ptr in sorted(pointers):
        found = locate.locate(ptr, SCENARIO_ROOT)
        if found is None:  # csv pointers (glucose traces) are not handled by the generator
            continue
        assert json.loads(found.read_text()) == parser.load_pointer(ptr), ptr
        checked += 1
    assert checked > 100


def test_new_pointer_keeps_the_folder_prefix():
    assert locate.new_pointer("reusable.simulations.custom_presets.preset_80_median_2_0_v1", "x_rai_main") == \
        "reusable.simulations.custom_presets.x_rai_main"


# ---------------------------------------------------------------- registry collisions
def _root(tmp_path):
    d = tmp_path / "reusable/metabolism_settings"
    (d / "profiles").mkdir(parents=True)
    (d / "presets").mkdir()
    (d / "profiles/median_v1.json").write_text('{"patient_insulin_type": "rapid_acting_adult"}')
    return tmp_path


def test_registry_rejects_a_name_that_exists_elsewhere_in_the_search_path(tmp_path):
    root = _root(tmp_path)
    (root / "reusable/metabolism_settings/presets/median_v1_rai_main.json").write_text("{}")
    reg = rewrite.Registry(root)
    src = root / "reusable/metabolism_settings/profiles/median_v1.json"
    with pytest.raises(rewrite.GenerationError, match="share a name"):
        reg.add("loop_risk_v2_rai_main", "reusable.metabolism_settings.median_v1", src, "median_v1_rai_main", "{}")


def test_registry_allows_the_same_file_to_be_regenerated(tmp_path):
    root = _root(tmp_path)
    src = root / "reusable/metabolism_settings/profiles/median_v1.json"
    (src.parent / "median_v1_rai_main.json").write_text("{}")  # left by an earlier identical run
    reg = rewrite.Registry(root)
    ptr = reg.add("loop_risk_v2_rai_main", "reusable.metabolism_settings.median_v1", src, "median_v1_rai_main", "{}")
    assert ptr == "reusable.metabolism_settings.median_v1_rai_main"


def test_profile_pointer_changes_only_type_and_timing(tmp_path):
    root = _root(tmp_path)
    src = root / "reusable/metabolism_settings/profiles/median_v1.json"
    src.write_text('{\n  "basal_rate": {"values": [0.5]},\n  "patient_insulin_type": "rapid_acting_adult"\n}')
    reg = rewrite.Registry(root)
    ptr = rewrite.profile_pointer(reg, "loop_risk_v2_urai_main", "reusable.metabolism_settings.median_v1",
                                  "ultra_rapid_v1", "main")
    text = next(iter(reg.files.values()))
    assert ptr.endswith("median_v1_urai_main")
    assert text == ('{\n  "basal_rate": {"values": [0.5]},\n  "patient_insulin_type": "ultra_rapid_v1",\n'
                    '  "insulin_timing": "main"\n}')


def test_profile_pointer_with_null_timing_writes_no_timing_key(tmp_path):
    root = _root(tmp_path)
    reg = rewrite.Registry(root)
    rewrite.profile_pointer(reg, "loop_risk_v2_urai_old", "reusable.metabolism_settings.median_v1", "fiasp", None)
    assert json.loads(next(iter(reg.files.values()))) == {"patient_insulin_type": "fiasp"}


def test_settings_pointer_reuses_fiasp_v2_when_only_the_model_differs():
    reg = rewrite.Registry(SCENARIO_ROOT)
    ptr = rewrite.settings_pointer(reg, "loop_risk_v2_urai_main", "reusable.loop_settings.2_0_v1", "fiasp")
    assert ptr == "reusable.loop_settings.fiasp_v2" and reg.files == {}


def test_fiasp_v2_has_the_guardrail_keys_mitigations_target():
    keys = json.loads(locate.locate("reusable.loop_settings.fiasp_v2", SCENARIO_ROOT).read_text())
    assert {"suspend_threshold", "max_basal_rate", "max_bolus"} <= set(keys)


# ---------------------------------------------------------------- verification helpers
def test_timing_rule_for_na_targets():
    assert verify.timing_ok(None, UNSET, "fiasp")
    assert verify.timing_ok(None, "main", "rapid_acting_adult")      # same behaviour as absent for non-v1 types
    assert not verify.timing_ok(None, "main", "ultra_rapid_v1")
    assert not verify.timing_ok(None, "slower", "rapid_acting_adult")
    assert verify.timing_ok("main", "main", "rapid_acting_v1") and not verify.timing_ok("main", UNSET, "rapid_acting_v1")


def test_non_v1_type_with_main_timing_is_behaviourally_identical_to_absent():
    """Basis of the timing rule: SimpleMetabolismModelV1 treats 'main' as the default for old types."""
    from tidepool_data_science_simulator.models.insulin_action_v1 import SimpleMetabolismModelV1
    import inspect
    assert inspect.signature(SimpleMetabolismModelV1.__init__).parameters["insulin_timing"].default == "main"


def _rec(sim_id="s", ptype="rapid_acting_adult", timing=UNSET, model="rapid_acting_adult", extra=0):
    return {"sim_id": sim_id, "patient_insulin_type": ptype, "insulin_timing": timing, "controller_model": model,
            "resolved": {"x": extra, "patient": {"patient_model": {"metabolism_settings": {
                "patient_insulin_type": ptype, "insulin_timing": timing, "isf": 50}}},
                "controller": {"settings": {"model": model, "max_bolus": 3}}}}


def test_compare_passes_when_only_the_three_fields_change_to_target():
    base = _rec()
    gen = _rec(ptype="ultra_rapid_v1", timing="main", model="fiasp")
    assert verify.compare(base, gen, ("ultra_rapid_v1", "main", "fiasp")) == []


def test_compare_reports_hard_difference_outside_the_three_fields():
    problems = verify.compare(_rec(), _rec(ptype="ultra_rapid_v1", timing="main", model="fiasp", extra=1),
                              ("ultra_rapid_v1", "main", "fiasp"))
    assert [p["kind"] for p in problems] == ["hard"]


def test_compare_classifies_patient_and_model_mismatches_as_correctable():
    problems = verify.compare(_rec(), _rec(), ("ultra_rapid_v1", "main", "fiasp"))
    assert {p["kind"] for p in problems} == {"patient", "model"}


def test_pump_side_difference_is_hard():
    base, gen = _rec(), _rec(ptype="ultra_rapid_v1", timing="main", model="fiasp")
    gen["resolved"]["patient"]["pump"] = {"metabolism_settings": {"patient_insulin_type": "ultra_rapid_v1"}}
    assert [p["kind"] for p in verify.compare(base, gen, ("ultra_rapid_v1", "main", "fiasp"))] == ["hard"]


def test_first_difference_treats_nan_as_equal_but_still_sees_real_differences():
    nan = float("nan")
    assert verify.first_difference({"g": [1.0, nan]}, {"g": [1.0, float("nan")]}) is None
    assert verify.first_difference({"g": [1.0, nan]}, {"g": [1.0, 2.0]}) is not None
