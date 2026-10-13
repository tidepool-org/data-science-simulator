"""Integration test for the TRSET-74 URAI-RAI scenario generator.

Runs the real generator against the real virtual-patient pool (path from ``URAI_RAI_PATIENT_POOL``), loads every
output through the real ``ScenarioParserV2``, and runs URAI/RAI pairs through the real Swift Loop controller. Nothing
is mocked. Output goes to a pytest tmp dir, never the repo.

The pool check FAILS (does not skip) when the pool is unset, missing or empty. Only the Swift test skips, when the
.dylib is not importable. Run under the arm64 conda env ``tidepool-data-science-simulator``.
"""
import datetime
import filecmp
import hashlib
import json
import os

import pandas as pd
import pytest

from urai_rai_support import load_sim, make_scenario_root
from tidepool_data_science_simulator.projects.urai_rai_analysis.generate import generate_run
from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import scan_pool

try:
    import loop_to_python_api.api  # noqa: F401
    _DYLIB_AVAILABLE, _DYLIB_ERR = True, ""
except Exception as e:  # pragma: no cover - env-dependent
    _DYLIB_AVAILABLE, _DYLIB_ERR = False, repr(e)

POOL_ENV_VAR = "URAI_RAI_PATIENT_POOL"
FMT = "%m/%d/%Y %H:%M:%S"
RUN_START = datetime.datetime(2019, 8, 15, 12, 0, 0)
SWIFT_HOURS = 4
N = 3
INSULIN = {"urai": "fiasp", "rai": "rapid_acting_adult"}


# ---- fixtures ----------------------------------------------------------------------------------------------

def _hash_tree(path):
    digest = {}
    for name in sorted(os.listdir(path)):
        with open(os.path.join(path, name), "rb") as f:
            digest[name] = hashlib.sha256(f.read()).hexdigest()
    return digest


@pytest.fixture(scope="module")
def pool_dir():
    path = os.environ.get(POOL_ENV_VAR)
    if not path:
        pytest.fail(f"{POOL_ENV_VAR} is not set; the integration test needs the virtual patient pool")
    if not os.path.isdir(path) or not [f for f in os.listdir(path) if f.endswith(".csv")]:
        pytest.fail(f"{POOL_ENV_VAR}={path} is missing or has no CSV files")
    return path


@pytest.fixture(scope="module")
def pool_hash_before(pool_dir):
    return _hash_tree(pool_dir)


@pytest.fixture(scope="module")
def pool_table(pool_dir):
    return scan_pool(pool_dir)


@pytest.fixture(scope="module")
def run_config(pool_table):
    """Small run: N per cohort, with a cohort sized past its eligible patients to exercise the accepted shortfall."""
    isf_sorted = sorted(pool_table["isf"], reverse=True)
    short_cutoff = isf_sorted[2]                                   # at least 3 eligible, usually exactly 3
    eligible = int((pool_table["isf"] >= short_cutoff).sum())
    return {
        "run_name": "it_run", "master_seed": 74, "default_n": N, "accept_smaller_group": True,
        "cohorts": [
            {"name": "pediatric", "filter": {"age_min": 6, "age_max": 18}},
            {"name": "adult", "filter": {"age_min": 18, "age_max": 55}},
            {"name": "older_adult", "filter": {"age_min": 55}},
            {"name": "isf_high", "filter": {"isf_min": 100}},
            {"name": "isf_low", "filter": {"isf_max": 25}},
            {"name": "unannounced_meals", "filter": {}, "announced": False},
            {"name": "isf_short", "filter": {"isf_min": short_cutoff}, "n": eligible + 2},
        ],
        "_eligible_short": eligible,
    }


def _clean(cfg):
    return {k: v for k, v in cfg.items() if not k.startswith("_")}


@pytest.fixture(scope="module")
def run_a(tmp_path_factory, pool_dir, pool_hash_before, run_config):
    root = make_scenario_root(tmp_path_factory.mktemp("run_a"))
    return root, generate_run(_clean(run_config), pool_dir=pool_dir, scenario_root=root)


@pytest.fixture(scope="module")
def run_b(tmp_path_factory, pool_dir, pool_hash_before, run_config):
    root = make_scenario_root(tmp_path_factory.mktemp("run_b"))
    return root, generate_run(_clean(run_config), pool_dir=pool_dir, scenario_root=root)


@pytest.fixture(scope="module")
def run_c(tmp_path_factory, pool_dir, pool_hash_before, run_config):
    """Same config plus one extra cohort, to show existing draws do not move."""
    cfg = _clean(run_config)
    cfg["cohorts"] = cfg["cohorts"] + [{"name": "extra_cohort", "filter": {"age_min": 30, "age_max": 40}}]
    root = make_scenario_root(tmp_path_factory.mktemp("run_c"))
    return root, generate_run(cfg, pool_dir=pool_dir, scenario_root=root)


@pytest.fixture(scope="module")
def manifest(run_a):
    return pd.read_csv(run_a[1].manifest_path, dtype={"also_in_cohorts": str, "shortfall_note": str},
                       keep_default_na=False)


def _scenario(run, file_name):
    return json.load(open(os.path.join(run[1].scenario_dir, file_name)))["override_config"][0]


def _meals(row):
    return [(datetime.datetime.strptime(m["time"], FMT), m["carbs_g"], m["absorption_min"])
            for m in json.loads(row["meal_schedule"])]


# ---- IT-1: the pool is read-only ---------------------------------------------------------------------------

def test_it1_pool_unchanged_and_no_patient_under_6(pool_dir, pool_hash_before, run_a, run_b, run_c, manifest):
    assert _hash_tree(pool_dir) == pool_hash_before
    assert (manifest["age"] >= 6).all()


# ---- IT-2: cohorts ------------------------------------------------------------------------------------------

COHORT_RULES = {
    "pediatric": lambda r: 6 <= r.age < 18,
    "adult": lambda r: 18 <= r.age < 55,
    "older_adult": lambda r: r.age >= 55,
    "isf_high": lambda r: r.isf >= 100,
    "isf_low": lambda r: r.isf <= 25,
    "unannounced_meals": lambda r: True,
}


def test_it2_cohort_membership_sizes_and_shortfall(manifest, run_config, pool_dir, tmp_path):
    for cohort, rule in COHORT_RULES.items():
        rows = manifest[(manifest["cohort"] == cohort) & (manifest["arm"] == "urai")]
        assert len(rows) == N, cohort
        assert rows.apply(rule, axis=1).all(), cohort
        assert rows["patient_id"].is_unique
    short = manifest[manifest["cohort"] == "isf_short"]
    assert len(short) == 2 * run_config["_eligible_short"]          # two arms per patient
    assert short["shortfall_note"].str.contains("requested n=").all()
    assert (manifest[manifest["cohort"] != "isf_short"]["shortfall_note"] == "").all()
    # without the opt-in the same config is a hard error and writes nothing
    strict = dict(_clean(run_config), accept_smaller_group=False)
    root = make_scenario_root(tmp_path)
    with pytest.raises(ValueError, match="requested n="):
        generate_run(strict, pool_dir=pool_dir, scenario_root=root)
    assert not os.path.exists(os.path.join(root, "loop_risk_v2_0"))


# ---- IT-3: pairing ------------------------------------------------------------------------------------------

def _diff(a, b, path=""):
    if isinstance(a, dict):
        assert a.keys() == b.keys()
        return [p for k in a for p in _diff(a[k], b[k], f"{path}.{k}")]
    if isinstance(a, list):
        assert len(a) == len(b)
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in _diff(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def test_it3_every_draw_is_a_urai_rai_pair_differing_only_in_insulin(manifest, run_a):
    assert len(manifest) % 2 == 0
    for pair_id, pair in manifest.groupby("pair_id"):
        assert sorted(pair["arm"]) == ["rai", "urai"], pair_id
        urai = _scenario(run_a, pair[pair["arm"] == "urai"].iloc[0]["file_name"])
        rai = _scenario(run_a, pair[pair["arm"] == "rai"].iloc[0]["file_name"])
        assert sorted(_diff(urai, rai)) == sorted([
            ".sim_id", ".patient.patient_model.metabolism_settings.patient_insulin_type",
            ".controller.settings.model"]), pair_id
        for arm, o in (("urai", urai), ("rai", rai)):
            assert o["controller"]["settings"]["model"] == INSULIN[arm]
            assert o["patient"]["patient_model"]["metabolism_settings"]["patient_insulin_type"] == INSULIN[arm]


# ---- IT-4 / IT-5: controller and patient settings ---------------------------------------------------------

def test_it4_controller_and_patient_settings(manifest, run_a):
    for _, row in manifest.iterrows():
        o = _scenario(run_a, row["file_name"])
        settings = o["controller"]["settings"]
        assert settings["partial_application_factor"] == 0.4 and settings["max_bolus"] == 20
        assert settings["suspend_threshold"] == json.loads(row["controller_settings"])["suspend_threshold"]
        pump, patient = o["patient"]["pump"]["metabolism_settings"], o["patient"]["patient_model"]["metabolism_settings"]
        for key in ("basal_rate", "carb_insulin_ratio", "insulin_sensitivity_factor"):
            assert pump[key] == patient[key]                           # no settings variance


# ---- IT-5: glucose seed -----------------------------------------------------------------------------------

def test_it5_glucose_seed_grid_and_pointers(manifest, run_a):
    start = datetime.datetime(2019, 8, 15, 0, 40)
    for pid in manifest["patient_id"].unique():
        seed = pd.read_csv(os.path.join(run_a[1].glucose_dir, f"{pid}.csv"))
        assert len(seed) == 137
        assert list(seed["datetime"]) == [(start + datetime.timedelta(minutes=5 * i)).strftime(FMT) for i in range(137)]
    for _, row in manifest.iterrows():
        o = _scenario(run_a, row["file_name"])
        expected = f"reusable.glucose.urai_rai.it_run.{row['patient_id']}"
        assert o["patient"]["sensor"]["glucose_history"] == expected
        assert o["patient"]["patient_model"]["glucose_history"] == expected


# ---- IT-6 / IT-7: meals and announcement -------------------------------------------------------------------

def test_it6_meal_rules_and_sharing(manifest, run_a):
    run_end = RUN_START + datetime.timedelta(hours=24)
    for pair_id, pair in manifest.groupby("pair_id"):
        urai, rai = (pair[pair["arm"] == a].iloc[0] for a in ("urai", "rai"))
        assert urai["meal_schedule"] == rai["meal_schedule"]            # shared by both arms
        meals = _meals(urai)
        times = [(t - RUN_START).total_seconds() / 60 for t, _, _ in meals]
        assert times == sorted(times)
        assert all(RUN_START <= t < run_end and t.minute % 5 == 0 and datetime.time(5) <= t.time() <= datetime.time(22)
                   for t, _, _ in meals)
        assert all(20 <= c <= 80 and a in (30, 180, 300) for _, c, a in meals)
        per_day = pd.Series([t.date() for t, _, _ in meals]).value_counts()
        assert len(per_day) == 2 and per_day.between(2, 4).all()
        close = [(a, b) for a, b in zip(times, times[1:]) if b - a < 60]
        assert len(close) == (1 if urai["has_seconds_pair"] else 0)
        assert all(5 <= b - a <= 55 for a, b in close)


def test_it7_announcement(manifest, run_a):
    for _, row in manifest.iterrows():
        o = _scenario(run_a, row["file_name"])["patient"]
        meals = _meals(row)
        if row["cohort"] == "unannounced_meals":
            assert not row["meals_announced"]
            assert o["pump"]["carb_entries"] == [] and o["patient_model"]["bolus_entries"] == []
        else:
            assert row["meals_announced"]
            assert o["pump"]["carb_entries"] == o["patient_model"]["carb_entries"]
            assert o["patient_model"]["bolus_entries"] == [
                {"time": t.strftime(FMT), "value": "accept_recommendation"} for t, _, _ in meals]
        assert [(datetime.datetime.strptime(c["start_time"], FMT), c["value"], c["duration"])
                for c in o["patient_model"]["carb_entries"]] == meals


# ---- IT-8: parser loads every file ---------------------------------------------------------------------

def test_it8_every_file_loads_through_the_parser(manifest, run_a):
    root, result = run_a
    assert os.path.isfile(os.path.join(result.scenario_dir, "run_config.json"))
    assert os.path.isfile(result.manifest_path)
    files = list(manifest["file_name"])
    for name in files:
        sim = load_sim(os.path.join(result.scenario_dir, name), root)   # raises if any override fails to apply
        row = manifest[manifest["file_name"] == name].iloc[0]
        assert sim.virtual_patient.patient_config.patient_insulin_type == INSULIN[row["arm"]]


# ---- IT-9: reproducibility ------------------------------------------------------------------------------

def _compare_dirs(a, b, names):
    for name in names:
        assert filecmp.cmp(os.path.join(a, name), os.path.join(b, name), shallow=False), name


def test_it9_same_config_and_seed_give_identical_bytes(run_a, run_b):
    for key in ("scenario_dir", "glucose_dir"):
        a, b = getattr(run_a[1], key), getattr(run_b[1], key)
        assert sorted(os.listdir(a)) == sorted(os.listdir(b))
        _compare_dirs(a, b, os.listdir(a))


def test_it9_adding_a_cohort_leaves_existing_draws_unchanged(run_a, run_c, manifest):
    names = list(manifest["file_name"])
    _compare_dirs(run_a[1].scenario_dir, run_c[1].scenario_dir, names)
    extra = run_c[1].manifest
    assert (extra["cohort"] == "extra_cohort").any()


def test_it9_missing_master_seed_is_an_error(run_config, pool_dir, tmp_path):
    cfg = _clean(run_config)
    del cfg["master_seed"]
    with pytest.raises(ValueError, match="master_seed"):
        generate_run(cfg, pool_dir=pool_dir, scenario_root=make_scenario_root(tmp_path))


# ---- IT-10: manifest ------------------------------------------------------------------------------------

def test_it10_manifest_columns_and_join(manifest, run_a):
    expected = ["file_name", "patient_id", "cohort", "also_in_cohorts", "pair_id", "arm", "insulin_model",
                "meals_announced", "age_band", "age", "isf", "derived_seed", "meal_schedule", "has_seconds_pair",
                "controller_settings", "shortfall_note"]
    assert list(manifest.columns) == expected
    on_disk = sorted(f for f in os.listdir(run_a[1].scenario_dir) if f.startswith("Simulation-Configuration"))
    assert sorted(manifest["file_name"]) == on_disk and manifest["file_name"].is_unique
    per_pair = manifest.groupby("pair_id")["derived_seed"].nunique()
    assert (per_pair == 1).all()                                       # both arms share the draw's seed
    assert manifest.drop_duplicates("pair_id")["derived_seed"].is_unique
    assert (manifest["insulin_model"] == manifest["arm"].map(INSULIN)).all()


# ---- IT-11: CSV -> JSON fidelity on real patients ----------------------------------------------------------

def _raw_tokens(path):
    from tidepool_data_science_simulator.projects.urai_rai_analysis.patient_pool import read_tokens
    return read_tokens(path)


def test_it11_real_patient_values_pass_through_exactly(manifest, run_a, pool_dir):
    root, result = run_a
    for pid in manifest["patient_id"].unique()[:5]:
        raw = _raw_tokens(os.path.join(pool_dir, f"{pid}.csv"))
        seed = pd.read_csv(os.path.join(result.glucose_dir, f"{pid}.csv"), dtype={"value": str})
        assert seed["value"].tolist() == raw["actual_blood_glucose"][:137]
        row = manifest[(manifest["patient_id"] == pid) & (manifest["arm"] == "urai")].iloc[0]
        o = _scenario(run_a, row["file_name"])["patient"]
        for side in ("pump", "patient_model"):
            ms = o[side]["metabolism_settings"]
            for key, (times, values) in {
                "basal_rate": ("basal_rate_start_times", "basal_rate_values"),
                "carb_insulin_ratio": ("carb_ratio_start_times", "carb_ratio_values"),
                "insulin_sensitivity_factor": ("sensitivity_ratio_start_times", "sensitivity_ratio_values"),
            }.items():
                assert ms[key]["start_times"] == raw[times]
                assert ms[key]["values"] == [float(v) for v in raw[values]]
        t = o["pump"]["target_range"]
        assert t["start_times"] == raw["target_range_start_times"]
        assert t["lower_values"] == [float(v) for v in raw["target_range_minimum_values"]]
        assert t["upper_values"] == [float(v) for v in raw["target_range_maximum_values"]]

        sim = load_sim(os.path.join(result.scenario_dir, row["file_name"]), root)
        vp = sim.virtual_patient
        expected = list(zip([datetime.datetime.strptime(d, FMT) for d in seed["datetime"]],
                            [float(v) for v in seed["value"]]))
        assert list(zip(vp.sensor.sensor_bg_history.datetimes, vp.sensor.sensor_bg_history.bg_values)) == expected
        pm = vp.patient_config.glucose_history
        assert list(zip(pm.datetimes, pm.bg_values)) == expected


# ---- IT-12: end to end through the Swift simulator ------------------------------------------------------

def _first_early_meal_pair(manifest, cohort_filter):
    """First pair in the cohorts whose first meal falls inside the Swift run window."""
    limit = RUN_START + datetime.timedelta(hours=SWIFT_HOURS)
    for _, row in manifest[(manifest["arm"] == "urai") & cohort_filter(manifest)].iterrows():
        meals = _meals(row)
        if meals[0][0] < limit - datetime.timedelta(minutes=30):
            return row["pair_id"], meals[0]
    pytest.fail("No draw in the cohort has a meal inside the Swift run window")


@pytest.mark.skipif(not _DYLIB_AVAILABLE, reason=f"loop_to_python_api / .dylib not available: {_DYLIB_ERR}")
@pytest.mark.parametrize("announced", [True, False], ids=["announced", "unannounced"])
def test_it12_pair_runs_end_to_end_through_swift(announced, manifest, run_a, tmp_path):
    root, result = run_a
    is_cohort = (lambda m: m["cohort"] != "unannounced_meals") if announced else (lambda m: m["cohort"] == "unannounced_meals")
    pair_id, (meal_time, carbs, absorption) = _first_early_meal_pair(manifest, is_cohort)
    pair = manifest[manifest["pair_id"] == pair_id]

    frames = {}
    for _, row in pair.iterrows():
        sim = load_sim(os.path.join(result.scenario_dir, row["file_name"]), root)
        sim.controller.loop_algo_io_dir = str(tmp_path)               # keep loop_algo_*.json out of the repo
        sim.run(early_stop_datetime=sim.start_time + datetime.timedelta(hours=SWIFT_HOURS))
        frames[row["arm"]] = sim.get_results_df()

    assert set(frames) == {"urai", "rai"}
    for arm, df in frames.items():
        active = df[df["active"] == 1]
        assert len(active) >= SWIFT_HOURS * 12 and active["bg"].notna().all()
        assert df.loc[meal_time, "true_carb_value"] == carbs and df.loc[meal_time, "true_carb_duration"] == absorption
        if announced:
            assert df.loc[meal_time, "reported_carb_value"] == carbs
            # accept_recommendation resolves to the Loop recommendation made at the meal step, delivered next step
            nxt = meal_time + datetime.timedelta(minutes=5)
            recommended = df.loc[meal_time, "loop_recommended_bolus_value"]
            assert recommended > 0 and df.loc[nxt, "true_bolus"] == pytest.approx(recommended)
        else:
            assert df["reported_carb_value"].isna().all()               # the pump never learns of the meal
    # the arms differ only through insulin model, so their results must differ
    assert not frames["urai"]["bg"].equals(frames["rai"]["bg"])
