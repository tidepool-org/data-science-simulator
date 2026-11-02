"""Unit tests for TRSET-77: the URAI-RAI generator aligned with the Data Analysis Plan (synthetic patients only)."""
import dataclasses
import hashlib
import json
import os
from fractions import Fraction

import pandas as pd
import pytest

from urai_rai_support import MAX_BOLUS, config_dict, make_scenario_root, write_raw_patient
from tidepool_data_science_simulator.projects.urai_rai_analysis import generate as generate_mod
from tidepool_data_science_simulator.projects.urai_rai_analysis.cohorts import apply_filter, cohort_order, draw_cohort
from tidepool_data_science_simulator.projects.urai_rai_analysis.config import MaxBolusConfig, parse_run_config
from tidepool_data_science_simulator.projects.urai_rai_analysis.eligibility import (
    Exclusion, check_pump_matches_model, eligible_table, evaluate_patient, evaluate_pool)
from tidepool_data_science_simulator.projects.urai_rai_analysis.generate import generate_run
from tidepool_data_science_simulator.projects.urai_rai_analysis.max_bolus import (
    exact, floor_to_increment, settings_flags)
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import Schedule, find_glucose_seed
from tidepool_data_science_simulator.projects.urai_rai_analysis.replacement import (
    generate_replacement_run, read_replacements)

ADULT = {"name": "adult", "filter": {"age_min": 18, "age_max": 60}}
N = 30  # glucose seed points used where the window position matters


def _cfg(cohorts=(ADULT,), **overrides):
    overrides.setdefault("glucose_seed", {"points": N})
    return parse_run_config(config_dict(list(cohorts), **overrides))


def _write(tmp_path, name="icgm_user_1", **kwargs):
    path = str(tmp_path / f"{name}.csv")
    write_raw_patient(path, **{"age": 30, **kwargs})
    return path


def _remove_row(path, row_name):
    text = open(path).read().replace(f"\n{row_name},", f"\n{row_name}X,")
    open(path, "w").write(text)


def _evaluate(tmp_path, cfg=None, **kwargs):
    return evaluate_patient(_write(tmp_path, **kwargs), cfg or _cfg())


# ---- config ------------------------------------------------------------------------------------------------------

def test_guardrails_and_max_bolus_blocks_are_required():
    for key in ("guardrails", "max_bolus"):
        raw = config_dict([ADULT])
        del raw[key]
        with pytest.raises(ValueError, match=key):
            parse_run_config(raw)


def test_the_static_controller_max_bolus_key_is_rejected():
    with pytest.raises(ValueError, match="controller.max_bolus is replaced"):
        parse_run_config(config_dict([ADULT], controller={"max_bolus": 20}))


# ---- E1: age ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("age,excluded", [(5, True), (5.99, True), (6, False)])
def test_e1_age_floor(tmp_path, age, excluded):
    result = _evaluate(tmp_path, age=age)
    assert isinstance(result, Exclusion) is excluded
    if excluded:
        assert result.rule == "E1"


def test_a_missing_age_is_a_hard_error_not_an_exclusion(tmp_path):
    path = _write(tmp_path)
    _remove_row(path, "age")
    with pytest.raises(ValueError, match="age is missing"):
        evaluate_patient(path, _cfg())


# ---- E2: glucose seed, read positionally -------------------------------------------------------------------------

def _glucose(length=70, gap_at=None):
    values = [str(100 + i) for i in range(length)]
    if gap_at is not None:
        values[gap_at] = ""
    return values


@pytest.mark.parametrize("gap_at", [0, 15, N - 1])
def test_a_gap_in_the_window_moves_the_seed_to_the_first_gap_free_run(tmp_path, gap_at):
    result = _evaluate(tmp_path, glucose=_glucose(gap_at=gap_at))
    assert result.record.glucose_tokens == tuple(str(100 + i) for i in range(gap_at + 1, gap_at + 1 + N))


def test_a_gap_after_the_window_does_not_matter(tmp_path):
    result = _evaluate(tmp_path, glucose=_glucose(gap_at=N))
    assert result.record.glucose_tokens == tuple(str(100 + i) for i in range(N))


def test_a_run_of_exactly_the_seed_length_is_enough_and_one_short_is_e2(tmp_path):
    assert not isinstance(_evaluate(tmp_path, glucose=_glucose(N)), Exclusion)
    short = _evaluate(tmp_path, name="icgm_user_2", glucose=_glucose(N - 1))
    assert (short.rule, short.patient_id) == ("E2", "icgm_user_2")


def test_two_short_runs_around_a_gap_are_e2(tmp_path):
    assert _evaluate(tmp_path, glucose=_glucose(2 * N - 2, gap_at=N - 1)).rule == "E2"


def test_a_non_numeric_cell_counts_as_missing():
    assert find_glucose_seed(["1", "x", "3"], 2) is None
    assert find_glucose_seed(["1", "nan", "3", "4"], 2) == ("3", "4")


# ---- E3: seed values in range -------------------------------------------------------------------------------------

@pytest.mark.parametrize("value,excluded", [("19", True), ("20", False), ("600", False), ("601", True)])
def test_e3_range_bounds(tmp_path, value, excluded):
    glucose = _glucose(N)
    glucose[5] = value
    assert isinstance(_evaluate(tmp_path, glucose=glucose), Exclusion) is excluded


def test_e3_does_not_search_for_a_later_clean_run(tmp_path):
    glucose = _glucose(70)       # one unbroken run, so the first 30 values are the seed
    glucose[5] = "19"
    assert _evaluate(tmp_path, glucose=glucose).rule == "E3"


# ---- E4: missing or non-numeric settings --------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"basal": (("00:00:00",), ("abc",))},
    {"carb_ratio": (("00:00:00",), ("",))},
    {"isf": ("n/a",)},
    {"target": (("00:00:00",), ("x",), (120.0,))},
    {"basal": (("00:00:00", "06:00:00"), (0.9,))},                     # start times and values differ in length
    {"basal": (("06:00:00",), (0.9,))},                                # does not start at 00:00
    {"basal": (("00:00:00", "00:00:00"), (0.9, 1.0))},                 # not increasing
])
def test_e4_missing_or_unusable_settings(tmp_path, kwargs):
    result = _evaluate(tmp_path, **kwargs)
    assert isinstance(result, Exclusion) and result.rule == "E4"


def test_a_missing_suspend_threshold_is_e4_without_a_default(tmp_path):
    path = _write(tmp_path)
    _remove_row(path, "suspend_threshold")
    assert evaluate_patient(path, _cfg()).rule == "E4"


def test_a_default_suspend_threshold_keeps_the_patient_and_is_flagged(tmp_path):
    missing = _write(tmp_path, name="icgm_user_1")
    _remove_row(missing, "suspend_threshold")
    own = _write(tmp_path, name="icgm_user_2", suspend_threshold=75.0)
    cfg = _cfg(default_suspend_threshold=67)
    kept, other = evaluate_patient(missing, cfg), evaluate_patient(own, cfg)
    assert (kept.default_suspend, kept.record.suspend_threshold) == (True, 67)
    assert (other.default_suspend, other.record.suspend_threshold) == (False, 75.0)


def test_a_default_suspend_threshold_outside_its_guardrail_is_e5(tmp_path):
    path = _write(tmp_path)
    _remove_row(path, "suspend_threshold")
    assert evaluate_patient(path, _cfg(default_suspend_threshold=60)).rule == "E5"


# ---- E5: absolute guardrails, each at its bound and one step outside ----------------------------------------------

def _basal(rate):
    return {"basal": (("00:00:00",), (rate,))}


@pytest.mark.parametrize("kwargs,rule", [
    (_basal(0.05), None), (_basal(0.04), "E5"), (_basal(30), None), (_basal(31), "E5"),
    ({"carb_ratio": (("00:00:00",), (2.0,))}, None), ({"carb_ratio": (("00:00:00",), (1.0,))}, "E5"),
    ({"carb_ratio": (("00:00:00",), (150.0,))}, None), ({"carb_ratio": (("00:00:00",), (151.0,))}, "E5"),
    ({"isf": (10.0,)}, None), ({"isf": (9.0,)}, "E5"), ({"isf": (500.0,)}, None), ({"isf": (501.0,)}, "E5"),
    ({"target": (("00:00:00",), (87.0,), (120.0,))}, None), ({"target": (("00:00:00",), (86.0,), (120.0,))}, "E5"),
    ({"target": (("00:00:00",), (100.0,), (180.0,))}, None), ({"target": (("00:00:00",), (100.0,), (181.0,))}, "E5"),
    ({"suspend_threshold": 67.0}, None), ({"suspend_threshold": 66.0}, "E5"),
    ({"suspend_threshold": 110.0, "target": (("00:00:00",), (120.0,), (150.0,))}, None),
    ({"suspend_threshold": 111.0, "target": (("00:00:00",), (120.0,), (150.0,))}, "E5"),
])
def test_e5_guardrail_bounds(tmp_path, kwargs, rule):
    result = _evaluate(tmp_path, **kwargs)
    assert (result.rule if isinstance(result, Exclusion) else None) == rule


def test_e5_checks_every_schedule_segment(tmp_path):
    result = _evaluate(tmp_path, basal=(("00:00:00", "12:00:00"), (0.9, 40.0)))
    assert result.rule == "E5"


# ---- E6: settings inconsistent with each other -------------------------------------------------------------------

@pytest.mark.parametrize("kwargs", [
    {"target": (("00:00:00",), (130.0,), (120.0,))},                                 # minimum above maximum
    {"suspend_threshold": 100.0, "target": (("00:00:00",), (90.0,), (120.0,))},      # suspend above target minimum
    {"suspend_threshold": 95.0, "target": (("00:00:00", "12:00:00"), (100.0, 90.0), (120.0, 120.0))},   # one segment
])
def test_e6_relationships(tmp_path, kwargs):
    assert _evaluate(tmp_path, **kwargs).rule == "E6"


def test_the_first_failing_rule_is_reported(tmp_path):
    glucose = _glucose(N)
    glucose[0] = "19"
    assert _evaluate(tmp_path, age=5, glucose=glucose, **_basal(99)).rule == "E1"
    assert _evaluate(tmp_path, glucose=glucose, **_basal(99)).rule == "E3"
    assert _evaluate(tmp_path, carb_ratio=(("00:00:00",), ("",)), **_basal(99)).rule == "E4"


# ---- S3 / S20: pump versus patient-model settings -----------------------------------------------------------------

def _pool(tmp_path, specs):
    pool = tmp_path / "pool"
    pool.mkdir(exist_ok=True)
    for name, kwargs in specs:
        write_raw_patient(str(pool / f"{name}.csv"), **{"age": 30, **kwargs})
    return str(pool)


def _break_actual_basal(pool, name):
    path = os.path.join(pool, f"{name}.csv")
    text = open(path).read().replace("actual_basal_rates,,0.9", "actual_basal_rates,,0.95")
    open(path, "w").write(text)


def _written(root):
    return [d for d in ("loop_risk_v2_0", os.path.join("reusable", "glucose", "urai_rai"))
            if os.path.exists(os.path.join(root, d))]


def test_a_mismatch_in_an_eligible_patient_raises_and_writes_nothing(tmp_path):
    pool = _pool(tmp_path, [("icgm_user_1", {})])
    _break_actual_basal(pool, "icgm_user_1")
    root = make_scenario_root(tmp_path)
    with pytest.raises(ValueError, match="differs from the pump setting"):
        generate_run(config_dict([ADULT], default_n=1), pool_dir=pool, scenario_root=root)
    assert _written(root) == []


def test_a_mismatch_in_an_excluded_patient_is_not_checked(tmp_path):
    pool = _pool(tmp_path, [("icgm_user_1", {}), ("icgm_user_2", {"age": 5})])
    _break_actual_basal(pool, "icgm_user_2")
    result = generate_run(config_dict([ADULT], default_n=1), pool_dir=pool, scenario_root=make_scenario_root(tmp_path))
    assert result.exclusions["rule"].tolist() == ["E1"]


def test_a_missing_patient_model_row_is_a_hard_error(tmp_path):
    path = _write(tmp_path)
    _remove_row(path, "actual_basal_rates")
    with pytest.raises(ValueError, match="actual_basal_rates is missing"):
        check_pump_matches_model(evaluate_patient(path, _cfg()))


# ---- nothing written until every check is done --------------------------------------------------------------------

def test_a_meal_rule_failure_stops_the_run_with_nothing_written(tmp_path, monkeypatch):
    pool = _pool(tmp_path, [(f"icgm_user_{i}", {}) for i in range(1, 4)])
    root = make_scenario_root(tmp_path)
    calls = []

    def failing_meals(*args, **kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise ValueError("cannot place meals")
        return generate_mod_meals(*args, **kwargs)

    generate_mod_meals = generate_mod.generate_meals
    monkeypatch.setattr(generate_mod, "generate_meals", failing_meals)
    with pytest.raises(ValueError, match="cannot place meals"):
        generate_run(config_dict([ADULT], default_n=3), pool_dir=pool, scenario_root=root)
    assert _written(root) == []


# ---- cohorts: ISF filter, seeded ordering ------------------------------------------------------------------------

def test_the_isf_cohort_filter_uses_the_first_schedule_entry(tmp_path):
    pool = _pool(tmp_path, [
        ("icgm_user_1", {"isf": (120.0, 40.0), "isf_times": ("00:00:00", "12:00:00")}),
        ("icgm_user_2", {"isf": (40.0, 120.0), "isf_times": ("00:00:00", "12:00:00")}),
    ])
    table = eligible_table(evaluate_pool(pool, _cfg()))
    cohort = _cfg([{"name": "isf_high", "filter": {"isf_min": 100}}]).cohorts[0]
    assert apply_filter(table, cohort)["patient_id"].tolist() == ["icgm_user_1"]


def _table(n):
    return pd.DataFrame({"patient_id": [f"p{i:02d}" for i in range(n)], "age": [30] * n, "isf": [50] * n,
                         "path": ["x"] * n})


def test_the_seeded_order_is_reproducible_and_the_cohort_is_its_first_n():
    table, cfg = _table(20), _cfg([{**ADULT, "n": 5}])
    order = cohort_order(table, cfg.cohorts[0], cfg.master_seed)
    assert order == cohort_order(table, cfg.cohorts[0], cfg.master_seed)
    assert sorted(order) == table["patient_id"].tolist()
    assert order != cohort_order(table, cfg.cohorts[0], cfg.master_seed + 1)
    assert draw_cohort(table, cfg.cohorts[0], cfg).patient_ids == order[:5]


# ---- max bolus (AC 13) and flags (AC 14) --------------------------------------------------------------------------

CFG = MaxBolusConfig(**MAX_BOLUS)


def _flags(basal_times, basal_values, cir=10.0, meal=80, cfg=CFG):
    return settings_flags(Schedule(tuple(basal_times), tuple(basal_values)), cir, meal, cfg)


def test_tdd_integrates_a_single_entry_schedule():
    flags = _flags(["00:00:00"], [0.9])
    assert (flags.tdd_estimate_u, flags.max_bolus_u) == (43.2, 21.6)


def test_tdd_integrates_a_multi_entry_schedule_by_hours_in_force():
    flags = _flags(["00:00:00", "06:30:00", "18:00:00"], [0.8, 1.15, 0.95])   # 5.2 + 13.225 + 5.7 = 24.125 U
    assert flags.tdd_estimate_u == 48.25
    assert flags.max_bolus_u == 24.1                                           # 24.125 rounded down to 0.05 U


@pytest.mark.parametrize("rate,clamped,expected", [(1.24, False, 29.75), (1.25, False, 30.0), (1.26, True, 30.0)])
def test_the_30_u_clamp_below_at_and_above(rate, clamped, expected):
    flags = _flags(["00:00:00"], [rate])
    assert (flags.max_bolus_clamped, flags.max_bolus_u) == (clamped, expected)


@pytest.mark.parametrize("cir,fits", [(8.0, True), (7.9, False), (8.1, True)])
def test_meal80_fits_at_cir_times_tdd_of_160(cir, fits):
    # 10 U/day basal, so TDD 20 U and a max bolus of 10 U: an 80 g meal fits when CIR >= 8 (CIR x TDD >= 160)
    assert _flags(["00:00:00", "10:00:00"], [1.0, 0.0], cir=cir).meal80_fits is fits


@pytest.mark.parametrize("value,expected", [("0.51", "0.5"), ("0.59", "0.55"), ("0.15", "0.15"), ("0.30", "0.3"),
                                            ("30.00", "30")])
def test_rounding_goes_down_and_on_grid_values_are_unchanged(value, expected):
    assert floor_to_increment(exact(float(value)), exact(0.05)) == exact(float(expected))


@pytest.mark.parametrize("rate,expected", [(0.00625, 0.15), (0.0125, 0.3)])
def test_a_value_exactly_on_the_grid_is_not_pushed_down_a_step(rate, expected):
    assert _flags(["00:00:00"], [rate]).max_bolus_u == expected


def test_the_clamp_is_applied_before_the_rounding():
    capped = MaxBolusConfig(0.5, 2, 0.52, 0.05)
    flags = _flags(["00:00:00"], [0.0375], cfg=capped)       # uncapped 0.9 U, clamped to 0.52, rounded down to 0.50
    assert (flags.max_bolus_u, flags.max_bolus_clamped) == (0.5, True)


# ---- generated runs: layout, records, pairs, comparator ----------------------------------------------------------

def _run(tmp_path, n_patients=3, cohorts=(ADULT,), run_name="unit", pool=None, root=None, **overrides):
    pool = pool or _pool(tmp_path, [(f"icgm_user_{i}", {}) for i in range(1, n_patients + 1)])
    root = root or make_scenario_root(tmp_path)
    overrides.setdefault("default_n", min(n_patients, 3))
    cfg = config_dict(list(cohorts), run_name=run_name, **overrides)
    return generate_run(cfg, pool_dir=pool, scenario_root=root), pool, root


def test_layout_is_one_tlr_folder_per_patient_with_records_beside_the_manifest(tmp_path):
    result, _, _ = _run(tmp_path, cohorts=(ADULT, {"name": "adult_again", "filter": {"age_min": 18, "age_max": 60}}))
    folders = sorted(os.listdir(result.run_dir))
    patient_dirs = [f for f in folders if f.startswith("TLR-000_")]
    assert patient_dirs == [f"TLR-000_icgm_user_{i}" for i in (1, 2, 3)]
    assert all("TLR-" in d for d in patient_dirs)                       # the risk runner's discovery filter
    assert sorted(set(folders) - set(patient_dirs)) == ["eligible_pool.csv", "exclusions.csv", "manifest.csv",
                                                        "run_config.json"]
    for d in patient_dirs:
        files = os.listdir(os.path.join(result.run_dir, d))
        assert len(files) == 4 and all(f.startswith("Simulation-Configuration-TLR-") and f.endswith(".json")
                                       for f in files)               # two cohorts x two arms, nothing else
    assert result.patient_dirs() == [os.path.join(result.run_dir, d) for d in patient_dirs]


def test_manifest_and_eligible_pool_carry_the_new_columns(tmp_path):
    pool = _pool(tmp_path, [("icgm_user_1", {}), ("icgm_user_2", {"age": 5}), ("icgm_user_3", {})])
    missing = os.path.join(pool, "icgm_user_3.csv")
    _remove_row(missing, "suspend_threshold")
    result, _, _ = _run(tmp_path, pool=pool, default_suspend_threshold=67, default_n=2)
    manifest = result.manifest.set_index("patient_id")
    assert manifest["default_suspend"].astype(str).groupby(level=0).first().to_dict() == {
        "icgm_user_1": "False", "icgm_user_3": "True"}
    assert set(manifest["max_bolus_mode"]) == {"per_patient"}
    assert {"tdd_estimate_u", "max_bolus_u", "max_bolus_clamped", "meal80_fits"} <= set(manifest.columns)
    assert result.exclusions.to_dict("records")[0]["rule"] == "E1"
    eligible = pd.read_csv(os.path.join(result.run_dir, "eligible_pool.csv")).set_index("patient_id")
    assert eligible["default_suspend"].astype(str).to_dict() == {"icgm_user_1": "False", "icgm_user_3": "True"}
    assert eligible.loc["icgm_user_1", "starting_glucose"] == 100.0


def test_both_arms_of_a_pair_have_the_same_max_bolus_and_flags(tmp_path):
    result, _, _ = _run(tmp_path)
    for _, pair in result.manifest.groupby("pair_id"):
        assert pair[["tdd_estimate_u", "max_bolus_u", "max_bolus_clamped", "meal80_fits"]].nunique().max() == 1
        settings = [json.loads(s) for s in pair["controller_settings"]]
        assert settings[0]["max_bolus"] == settings[1]["max_bolus"] == pair["max_bolus_u"].iloc[0]


def _normalised(text, run_name):
    return text.replace(run_name, "<run_name>")


def test_the_static_comparator_differs_only_in_max_bolus_mode_and_run_name_paths(tmp_path):
    pool = _pool(tmp_path, [(f"icgm_user_{i}", {}) for i in range(1, 4)])
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
    per_patient, _, _ = _run(tmp_path / "a", pool=pool, run_name="main")
    comparator, _, _ = _run(tmp_path / "b", pool=pool, run_name="static30",
                            max_bolus={**MAX_BOLUS, "static_u": 30})

    assert set(comparator.manifest["max_bolus_mode"]) == {"static"}
    cols = [c for c in per_patient.manifest.columns if c not in ("max_bolus_mode", "controller_settings")]
    pd.testing.assert_frame_equal(per_patient.manifest[cols], comparator.manifest[cols])      # incl. the flags
    for a, b in zip(per_patient.manifest["controller_settings"], comparator.manifest["controller_settings"]):
        sa, sb = json.loads(a), json.loads(b)
        assert sb["max_bolus"] == 30 and {k: v for k, v in sa.items() if k != "max_bolus"} == {
            k: v for k, v in sb.items() if k != "max_bolus"}
    for row in per_patient.manifest.itertuples():
        a = json.load(open(per_patient.scenario_path(row.patient_id, row.file_name)))
        b = json.load(open(comparator.scenario_path(row.patient_id, row.file_name)))
        assert b["override_config"][0]["controller"]["settings"]["max_bolus"] == 30
        a["override_config"][0]["controller"]["settings"]["max_bolus"] = 30
        assert _normalised(json.dumps(a, sort_keys=True), "main") == _normalised(json.dumps(b, sort_keys=True),
                                                                                 "static30")
    for name in ("eligible_pool.csv", "exclusions.csv"):
        assert open(os.path.join(per_patient.run_dir, name)).read() == open(os.path.join(comparator.run_dir, name)).read()


# ---- replacement runs --------------------------------------------------------------------------------------------

def _tree_hash(path):
    digest = hashlib.sha256()
    for folder, _, files in sorted(os.walk(path)):
        for name in sorted(files):
            digest.update(os.path.join(folder, name).encode())
            digest.update(open(os.path.join(folder, name), "rb").read())
    return digest.hexdigest()


@pytest.fixture
def original(tmp_path):
    pool = _pool(tmp_path, [(f"icgm_user_{i}", {}) for i in range(1, 9)])
    result, _, root = _run(tmp_path, n_patients=8, pool=pool, run_name="orig")
    cfg = parse_run_config(config_dict([ADULT], run_name="orig", default_n=3, glucose_seed={"points": 137}))
    order = cohort_order(eligible_table(evaluate_pool(pool, cfg)), cfg.cohorts[0], cfg.master_seed)
    return result, pool, root, order


def test_a_replacement_is_the_next_patient_in_the_seeded_order_and_the_original_is_untouched(original):
    result, pool, root, order = original
    before = _tree_hash(result.run_dir)
    pair = f"adult_{order[0]}"
    new = generate_replacement_run(result.run_dir, [(pair, "run failed")], "repl", pool, root)
    assert _tree_hash(result.run_dir) == before
    assert set(new.manifest["patient_id"]) == {order[3]}
    assert set(new.manifest["replaces_pair_id"]) == {pair} and set(new.manifest["replacement_reason"]) == {"run failed"}
    assert set(result.manifest["derived_seed"]).isdisjoint(set(new.manifest["derived_seed"]))
    assert sorted(os.listdir(new.run_dir)) == ["TLR-000_" + order[3], "eligible_pool.csv", "exclusions.csv",
                                               "manifest.csv", "run_config.json"]
    assert json.load(open(os.path.join(new.run_dir, "run_config.json")))["run_name"] == "repl"


def test_replacements_never_reuse_a_patient_within_a_run_or_across_runs(original):
    result, pool, root, order = original
    both = generate_replacement_run(result.run_dir, [(f"adult_{order[0]}", "a"), (f"adult_{order[1]}", "b")],
                                    "repl_two", pool, root)
    assert both.manifest.drop_duplicates("patient_id")["patient_id"].tolist() == [order[3], order[4]]
    later = generate_replacement_run(result.run_dir, [(f"adult_{order[2]}", "c")], "repl_three", pool, root,
                                     prior_run_dirs=[both.run_dir])
    assert set(later.manifest["patient_id"]) == {order[5]}
    chained = generate_replacement_run(result.run_dir, [(f"adult_{order[3]}", "d")], "repl_four", pool, root,
                                       prior_run_dirs=[both.run_dir, later.run_dir])
    assert set(chained.manifest["patient_id"]) == {order[6]}      # a replacement can itself be replaced


def test_running_out_of_eligible_patients_raises_and_writes_nothing(original):
    result, pool, root, order = original
    pairs = [(f"adult_{order[i % 3]}", "x") for i in range(6)]       # 5 candidates remain after the cohort of 3
    with pytest.raises(ValueError, match="no eligible patient left"):
        generate_replacement_run(result.run_dir, pairs, "repl_exhausted", pool, root)
    assert not os.path.exists(os.path.join(root, "loop_risk_v2_0", "loop_risk_v2_urai_rai", "repl_exhausted"))


def test_an_unknown_pair_and_a_changed_pool_are_errors(original, tmp_path):
    result, pool, root, order = original
    with pytest.raises(ValueError, match="Unknown pair"):
        generate_replacement_run(result.run_dir, [("adult_nobody", "x")], "repl_bad", pool, root)
    os.remove(os.path.join(pool, f"{order[1]}.csv"))
    with pytest.raises(ValueError, match="no longer reproduces the original draw"):
        generate_replacement_run(result.run_dir, [(f"adult_{order[0]}", "x")], "repl_bad2", pool, root)


def test_read_replacements(tmp_path):
    path = tmp_path / "pairs.csv"
    path.write_text("pair_id,reason\nadult_icgm_user_1,run crashed\n")
    assert read_replacements(str(path)) == [("adult_icgm_user_1", "run crashed")]
    path.write_text("pair_id,reason\nadult_icgm_user_1,\n")
    with pytest.raises(ValueError, match="needs a pair_id and a reason"):
        read_replacements(str(path))
