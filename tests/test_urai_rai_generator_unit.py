"""Unit tests for the TRSET-74 URAI-RAI scenario generator, using inline synthetic records only (no real patients)."""
import copy
import datetime
import json
import logging
import os

import pandas as pd
import pytest

from urai_rai_support import load_sim, make_scenario_root, parsed_settings, write_raw_patient
from tidepool_data_science_simulator.projects.urai_rai_analysis import cohorts as cohorts_mod
from tidepool_data_science_simulator.projects.urai_rai_analysis.config import MealConfig, parse_run_config
from tidepool_data_science_simulator.projects.urai_rai_analysis.generate import generate_run
from tidepool_data_science_simulator.projects.urai_rai_analysis.meals import generate_meals
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import parse_patient, scan_pool
from tidepool_data_science_simulator.projects.urai_rai_analysis.scenario_builder import (
    build_scenario, glucose_seed_rows,
)

RUN_START = datetime.datetime(2019, 8, 15, 12, 0, 0)
MEAL_CFG = MealConfig()
FMT = "%m/%d/%Y %H:%M:%S"


def _config(cohorts, **overrides):
    cfg = {"run_name": "unit", "master_seed": 7, "cohorts": cohorts}
    cfg.update(overrides)
    return parse_run_config(cfg)


def _table(ages, isfs):
    return pd.DataFrame({"patient_id": [f"p{i}" for i in range(len(ages))], "age": ages, "isf": isfs,
                         "path": ["x"] * len(ages)})


def _members(table, name, filt, min_age=6):
    cfg = _config([{"name": name, "filter": filt, "n": 100}], min_age=min_age, accept_smaller_group=True)
    draw = cohorts_mod.draw_cohort(table, cfg.cohorts[0], cfg)
    return set(table.set_index("patient_id").loc[draw.patient_ids, "age"].tolist()), draw


# ---- cutoffs ------------------------------------------------------------------------------------------------

AGES = [5, 6, 17, 18, 54, 55]


@pytest.mark.parametrize("name,filt,expected", [
    ("pediatric", {"age_min": 6, "age_max": 18}, {6, 17}),
    ("adult", {"age_min": 18, "age_max": 55}, {18, 54}),
    ("older_adult", {"age_min": 55}, {55}),
])
def test_age_cutoffs(name, filt, expected):
    ages, _ = _members(_table(AGES, [50] * len(AGES)), name, filt)
    assert ages == expected  # age 5 never appears: excluded by the floor before any filter


def test_age_floor_applies_before_filters():
    table = _table([5, 6], [50, 50])
    ages, _ = _members(table, "everyone", {})
    assert ages == {6}


@pytest.mark.parametrize("name,filt,expected", [
    ("isf_high", {"isf_min": 100}, {100}),
    ("isf_low", {"isf_max": 25}, {25}),
])
def test_isf_cutoffs(name, filt, expected):
    isfs = [25, 26, 99, 100]
    table = _table([30] * 4, isfs)
    cfg = _config([{"name": name, "filter": filt, "n": 100}], accept_smaller_group=True)
    draw = cohorts_mod.draw_cohort(table, cfg.cohorts[0], cfg)
    assert set(table.set_index("patient_id").loc[draw.patient_ids, "isf"]) == expected


def test_age_band_labels():
    assert [cohorts_mod.age_band(a) for a in AGES] == ["<18", "<18", "<18", "18-54", "18-54", "55+"]


# ---- shortfall ----------------------------------------------------------------------------------------------

def test_shortfall_is_hard_error_by_default():
    cfg = _config([{"name": "adult", "filter": {"age_min": 18, "age_max": 55}, "n": 3}])
    with pytest.raises(ValueError, match="requested n=3, only 2 eligible"):
        cohorts_mod.draw_cohort(_table([18, 54, 10], [50] * 3), cfg.cohorts[0], cfg)


def test_accepted_shortfall_is_logged_and_noted(caplog):
    cfg = _config([{"name": "adult", "filter": {"age_min": 18, "age_max": 55}, "n": 3}], accept_smaller_group=True)
    with caplog.at_level(logging.WARNING):
        draw = cohorts_mod.draw_cohort(_table([18, 54, 10], [50] * 3), cfg.cohorts[0], cfg)
    assert len(draw.patient_ids) == 2
    assert "requested n=3, only 2 eligible" in draw.shortfall_note
    assert "Accepted shortfall" in caplog.text


# ---- seeds / config -----------------------------------------------------------------------------------------

def test_master_seed_is_required():
    with pytest.raises(ValueError, match="master_seed"):
        parse_run_config({"run_name": "x", "cohorts": []})


def test_cohort_draw_is_independent_of_other_cohorts():
    table = _table(list(range(18, 60)), [50] * 42)
    adult = {"name": "adult", "filter": {"age_min": 18, "age_max": 55}, "n": 5}
    other = {"name": "older_adult", "filter": {"age_min": 55}, "n": 2}
    a = _config([adult])
    b = _config([other, adult])
    assert (cohorts_mod.draw_cohort(table, a.cohorts[0], a).patient_ids
            == cohorts_mod.draw_cohort(table, b.cohorts[1], b).patient_ids)


def test_derive_seed_is_stable_and_distinct():
    assert cohorts_mod.derive_seed(1, "draw", "a", "p") == cohorts_mod.derive_seed(1, "draw", "a", "p")
    assert cohorts_mod.derive_seed(1, "draw", "a", "p") != cohorts_mod.derive_seed(1, "draw", "b", "p")
    assert cohorts_mod.derive_seed(1, "draw", "a", "p") != cohorts_mod.derive_seed(2, "draw", "a", "p")


# ---- raw patient parsing (both layouts) ---------------------------------------------------------------------

@pytest.mark.parametrize("layout", ["observed", "shifted"])
def test_both_column_layouts_parse_identically(tmp_path, layout):
    path = str(tmp_path / "icgm_user_1.csv")
    write_raw_patient(path, age=30, layout=layout, basal=(("00:00:00", "08:00:00"), (0.8, 1.1)),
                      isf=(55.5, 61.0), isf_times=("00:00:00", "12:00:00"))
    record = parse_patient(path, n_glucose=30)
    assert record.age == 30
    assert record.basal.start_times == ("00:00:00", "08:00:00") and record.basal.values == (0.8, 1.1)
    assert record.isf.values == (55.5, 61.0)
    assert len(record.glucose_tokens) == 30


def test_scan_never_modifies_the_pool(tmp_path):
    path = str(tmp_path / "icgm_user_1.csv")
    write_raw_patient(path, age=30)
    before = (open(path, "rb").read(), os.stat(path).st_mtime_ns)
    scan_pool(str(tmp_path))
    parse_patient(path, 30)
    assert (open(path, "rb").read(), os.stat(path).st_mtime_ns) == before


def test_missing_required_setting_is_an_error(tmp_path):
    path = str(tmp_path / "icgm_user_1.csv")
    write_raw_patient(path, age=30)
    text = open(path).read().replace("suspend_threshold", "suspend_thresholdX")
    open(path, "w").write(text)
    with pytest.raises(ValueError, match="suspend_threshold"):
        parse_patient(path, 30)


def test_pump_and_actual_settings_must_agree(tmp_path):
    path = str(tmp_path / "icgm_user_1.csv")
    write_raw_patient(path, age=30)
    text = open(path).read().replace("actual_basal_rates,0.9", "actual_basal_rates,0.95", 1)
    text = text.replace("actual_basal_rates,,0.9", "actual_basal_rates,,0.95")
    open(path, "w").write(text)
    with pytest.raises(ValueError, match="differs from the pump setting"):
        parse_patient(path, 30)


# ---- meal rules ---------------------------------------------------------------------------------------------

def _minutes(meals):
    return [int((m.time - RUN_START).total_seconds() // 60) for m in meals]


@pytest.mark.parametrize("seed", range(300))
def test_meal_rule_invariants(seed):
    schedule = generate_meals(seed, MEAL_CFG, RUN_START, 24)
    meals, run_end = schedule.meals, RUN_START + datetime.timedelta(hours=24)
    times = _minutes(meals)
    assert times == sorted(times)
    for m in meals:
        assert m.time.minute % 5 == 0 and m.time.second == 0          # on the 5-minute grid
        assert RUN_START <= m.time < run_end                            # clipped to the run
        assert datetime.time(5, 0) <= m.time.time() <= datetime.time(22, 0)
        assert 20 <= m.carbs_g <= 80
        assert m.absorption_min in (30, 180, 300)
    by_day = {}
    for m in meals:
        by_day.setdefault(m.time.date(), []).append(m)
    assert len(by_day) == 2
    assert all(2 <= len(v) <= 4 for v in by_day.values())
    close = [(a, b) for a, b in zip(times, times[1:]) if b - a < 60]
    if schedule.has_seconds_pair:
        assert len(close) == 1 and 5 <= close[0][1] - close[0][0] <= 55
        # the second meal of the pair keeps 60 min from every other meal
        second = close[0][1]
        assert all(abs(second - t) >= 60 for t in times if t not in close[0])
    else:
        assert close == []


def test_seconds_pair_rate_and_absorption_mix():
    schedules = [generate_meals(s, MEAL_CFG, RUN_START, 24) for s in range(600)]
    assert 0.18 < sum(s.has_seconds_pair for s in schedules) / len(schedules) < 0.32
    absorptions = pd.Series([m.absorption_min for s in schedules for m in s.meals]).value_counts(normalize=True)
    assert set(absorptions.index) == {30, 180, 300} and absorptions.min() > 0.25


def test_meals_are_deterministic_per_seed():
    assert generate_meals(11, MEAL_CFG, RUN_START, 24) == generate_meals(11, MEAL_CFG, RUN_START, 24)
    assert generate_meals(11, MEAL_CFG, RUN_START, 24) != generate_meals(12, MEAL_CFG, RUN_START, 24)


# ---- pair / announcement invariants -------------------------------------------------------------------------

def _record(tmp_path, **kwargs):
    path = str(tmp_path / "icgm_user_9.csv")
    write_raw_patient(path, **{"age": 30, **kwargs})
    return parse_patient(path, 137)


def _diff(a, b, path=""):
    """Paths at which two JSON-like structures differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        assert a.keys() == b.keys()
        return [p for k in a for p in _diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list):
        assert len(a) == len(b)
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


@pytest.mark.parametrize("announced", [True, False])
def test_pair_differs_only_in_the_two_insulin_fields(tmp_path, announced):
    record, cfg = _record(tmp_path), _config([{"name": "c", "filter": {}}])
    meals = list(generate_meals(3, MEAL_CFG, RUN_START, 24).meals)
    urai = build_scenario(record, cfg, "urai", "c", meals, announced)
    rai = build_scenario(record, cfg, "rai", "c", meals, announced)
    assert sorted(_diff(urai, rai)) == sorted([
        ".metadata.simulation_id", ".override_config[0].sim_id",
        ".override_config[0].patient.patient_model.metabolism_settings.patient_insulin_type",
        ".override_config[0].controller.settings.model",
    ])
    o_u, o_r = urai["override_config"][0], rai["override_config"][0]
    assert o_u["controller"]["settings"]["model"] == o_u["patient"]["patient_model"]["metabolism_settings"]["patient_insulin_type"] == "fiasp"
    assert o_r["controller"]["settings"]["model"] == o_r["patient"]["patient_model"]["metabolism_settings"]["patient_insulin_type"] == "rapid_acting_adult"


def test_announced_scenario_carries_meals_on_both_sides(tmp_path):
    record, cfg = _record(tmp_path), _config([{"name": "c", "filter": {}}])
    meals = list(generate_meals(3, MEAL_CFG, RUN_START, 24).meals)
    o = build_scenario(record, cfg, "urai", "c", meals, True)["override_config"][0]["patient"]
    assert o["pump"]["carb_entries"] == o["patient_model"]["carb_entries"]
    assert [c["duration"] for c in o["pump"]["carb_entries"]] == [m.absorption_min for m in meals]
    assert o["patient_model"]["bolus_entries"] == [
        {"time": m.time.strftime(FMT), "value": "accept_recommendation"} for m in meals]


def test_unannounced_scenario_hides_meals_from_the_pump(tmp_path):
    record, cfg = _record(tmp_path), _config([{"name": "c", "filter": {}, "announced": False}])
    meals = list(generate_meals(3, MEAL_CFG, RUN_START, 24).meals)
    o = build_scenario(record, cfg, "rai", "c", meals, False)["override_config"][0]["patient"]
    assert o["pump"]["carb_entries"] == [] and o["patient_model"]["bolus_entries"] == []
    assert len(o["patient_model"]["carb_entries"]) == len(meals)


# ---- CSV -> JSON fidelity -----------------------------------------------------------------------------------

SOURCE_GLUCOSE = [f"{80 + i * 1.2345678:.7f}" for i in range(140)]  # unrounded decimals, 140 > 137


def test_glucose_seed_is_first_137_source_values_unchanged(tmp_path):
    record = _record(tmp_path, glucose=SOURCE_GLUCOSE)
    cfg = _config([{"name": "c", "filter": {}}])
    rows = glucose_seed_rows(record, cfg)
    assert [r["value"] for r in rows] == SOURCE_GLUCOSE[:137]       # same text, same order, none dropped or added
    start = datetime.datetime(2019, 8, 15, 0, 40)
    assert [r["datetime"] for r in rows] == [(start + datetime.timedelta(minutes=5 * i)).strftime(FMT)
                                             for i in range(137)]
    assert rows[-1]["datetime"] == "08/15/2019 12:00:00"


MULTI = dict(
    basal=(("00:00:00", "06:30:00", "18:00:00"), (0.8, 1.15, 0.95)),
    carb_ratio=(("00:00:00", "11:00:00"), (14.0, 9.5)),
    isf=(55.5, 48.25, 61.0), isf_times=("00:00:00", "07:00:00", "21:15:00"),
    target=(("00:00:00", "08:00:00"), (95.0, 105.0), (115.0, 125.0)),
)


@pytest.mark.parametrize("layout,schedules", [
    ("observed", MULTI), ("shifted", MULTI),
    ("observed", {}), ("shifted", {}),   # single-entry schedules as seen in the real pool
])
def test_generated_json_and_parser_preserve_every_schedule(tmp_path, layout, schedules):
    pool = tmp_path / "pool"
    pool.mkdir()
    write_raw_patient(str(pool / "icgm_user_1.csv"), age=30, layout=layout, glucose=SOURCE_GLUCOSE, **schedules)
    root = make_scenario_root(tmp_path)
    cfg = {"run_name": "fidelity", "master_seed": 5, "default_n": 1,
           "cohorts": [{"name": "adult", "filter": {"age_min": 18, "age_max": 55}}]}
    result = generate_run(cfg, pool_dir=str(pool), scenario_root=root)

    basal = schedules.get("basal", (("00:00:00",), (0.9,)))
    carb_ratio = schedules.get("carb_ratio", (("00:00:00",), (12.0,)))
    target = schedules.get("target", (("00:00:00",), (100.0,), (120.0,)))
    expected = {
        "basal": list(zip(*basal)),
        "carb_ratio": list(zip(*carb_ratio)),
        "isf": list(zip(schedules.get("isf_times", ("00:00:00",)), schedules.get("isf", (60.0,)))),
        "target": list(zip(target[0], zip(target[1], target[2]))),
    }

    for arm in ("urai", "rai"):
        path = os.path.join(result.scenario_dir, f"Simulation-Configuration-TLR-urai_rai_adult_icgm_user_1_{arm}.json")
        config = json.load(open(path))["override_config"][0]["patient"]
        # source -> JSON: same start times, values and entry counts
        for side in ("pump", "patient_model"):
            ms = config[side]["metabolism_settings"]
            assert list(zip(ms["basal_rate"]["start_times"], ms["basal_rate"]["values"])) == expected["basal"]
            assert list(zip(ms["carb_insulin_ratio"]["start_times"], ms["carb_insulin_ratio"]["values"])) == expected["carb_ratio"]
            assert list(zip(ms["insulin_sensitivity_factor"]["start_times"], ms["insulin_sensitivity_factor"]["values"])) == expected["isf"]
        t = config["pump"]["target_range"]
        assert list(zip(t["start_times"], zip(t["lower_values"], t["upper_values"]))) == expected["target"]

        # JSON -> parsed simulation, pump and patient model
        parsed = parsed_settings(load_sim(path, root))
        for side in ("pump", "patient_model"):
            for name in ("basal", "carb_ratio", "isf"):
                assert [(t.strftime("%H:%M:%S"), v) for t, v in parsed[side][name]] == expected[name]
        assert [(t.strftime("%H:%M:%S"), v) for t, v in parsed["pump"]["target"]] == expected["target"]
        # the 80/180 fallback must never be used
        assert (80.0, 180.0) not in [v for _, v in parsed["pump"]["target"]]


@pytest.mark.parametrize("layout", ["observed", "shifted"])
def test_glucose_flows_from_csv_through_parser_to_sensor_and_patient(tmp_path, layout):
    pool = tmp_path / "pool"
    pool.mkdir()
    write_raw_patient(str(pool / "icgm_user_1.csv"), age=30, layout=layout, glucose=SOURCE_GLUCOSE)
    root = make_scenario_root(tmp_path)
    cfg = {"run_name": "fidelity", "master_seed": 5, "default_n": 1,
           "cohorts": [{"name": "adult", "filter": {"age_min": 18, "age_max": 55}}]}
    result = generate_run(cfg, pool_dir=str(pool), scenario_root=root)

    seed = pd.read_csv(os.path.join(result.glucose_dir, "icgm_user_1.csv"), dtype={"value": str})
    assert seed["value"].tolist() == SOURCE_GLUCOSE[:137]            # raw CSV -> seed CSV, exact text

    path = os.path.join(result.scenario_dir, "Simulation-Configuration-TLR-urai_rai_adult_icgm_user_1_urai.json")
    vp = load_sim(path, root).virtual_patient
    expected = [(datetime.datetime.strptime(d, FMT), float(v)) for d, v in zip(seed["datetime"], seed["value"])]
    assert list(zip(vp.sensor.sensor_bg_history.datetimes, vp.sensor.sensor_bg_history.bg_values)) == expected
    assert list(zip(vp.patient_config.glucose_history.datetimes, vp.patient_config.glucose_history.bg_values)) == expected
