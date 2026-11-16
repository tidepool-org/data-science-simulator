"""Tests for the v1 insulin-action curves (tidepool_data_science_simulator/models/insulin_action_v1.py).

Expected values are those reported in the "Simulated patient insulin action" white paper, release v1.0:
time of peak effect 86 / 79 min (faster 65 / 60, slower 115 / 106), effect still to come at 2 h 53% (NovoLog,
Humalog), factors 0.750 / 1.334.
"""

import json
import os

import numpy as np
import pytest

from tidepool_data_science_models.models.simple_metabolism_model import SimpleMetabolismModel
from tidepool_data_science_models.models.treatment_models import PalermInsulinModel

from tidepool_data_science_simulator.models.insulin_action_v1 import (
    INSULIN_TIMING_FACTORS, MAX_EFFECT_OUTSIDE_WINDOW, InsulinActionV1Model, SimpleMetabolismModelV1,
    V1_TIME_CONSTANTS_MIN, effect_fraction_delivered, prediction_horizon_hours,
)
from tidepool_data_science_simulator.makedata.make_patient import (
    DATETIME_DEFAULT, get_canonical_risk_patient_config, get_canonical_risk_pump_config, get_canonical_sensor_config,
)
from tidepool_data_science_simulator.models.patient import VirtualPatient
from tidepool_data_science_simulator.models.pump import ContinuousInsulinPump
from tidepool_data_science_simulator.models.sensor import IdealSensor

FINE = np.arange(0, 720, 0.01)


def peak_minutes(time_constants):
    rate = np.diff(effect_fraction_delivered(time_constants, FINE))
    return FINE[np.argmax(rate)]


def scaled(insulin_type, timing):
    return tuple(tau * INSULIN_TIMING_FACTORS[timing] for tau in V1_TIME_CONSTANTS_MIN[insulin_type])


@pytest.mark.parametrize("insulin_type, timing, expected_peak", [
    ("rapid_acting_v1", "main", 86), ("rapid_acting_v1", "faster", 65), ("rapid_acting_v1", "slower", 115),
    ("ultra_rapid_v1", "main", 79), ("ultra_rapid_v1", "faster", 60), ("ultra_rapid_v1", "slower", 106),
])
def test_time_of_peak_effect_matches_white_paper(insulin_type, timing, expected_peak):
    assert round(peak_minutes(scaled(insulin_type, timing))) == expected_peak


def test_timing_factors():
    assert INSULIN_TIMING_FACTORS["main"] == 1.0
    assert INSULIN_TIMING_FACTORS["faster"] == pytest.approx(0.750, abs=5e-4)
    assert INSULIN_TIMING_FACTORS["slower"] == pytest.approx(1.334, abs=5e-4)


def test_effect_remaining_at_two_hours():
    remaining = 1 - effect_fraction_delivered(V1_TIME_CONSTANTS_MIN["rapid_acting_v1"], np.array([120.0]))[0]
    assert round(100 * remaining) == 53


def test_closed_form_matches_numerical_integration_of_the_compartments():
    taus = V1_TIME_CONSTANTS_MIN["rapid_acting_v1"]
    dt = 0.01
    x = np.array([1.0, 0.0, 0.0])
    delivered, out = 0.0, []
    for _ in range(int(480 / dt)):
        out.append(delivered)
        k = 1 / np.array(taus)
        flow = k * x
        delivered += flow[2] * dt
        x = x + dt * np.array([-flow[0], flow[0] - flow[1], flow[1] - flow[2]])
    t = np.arange(len(out)) * dt
    assert np.max(np.abs(np.array(out) - effect_fraction_delivered(taus, t))) < 1e-3


def test_run_has_palerm_interface_and_conserves_effect():
    isf, dose = 40.0, 2.0
    v1 = InsulinActionV1Model("rapid_acting_v1", isf=isf, cir=10.0)
    palerm = PalermInsulinModel(isf=isf, cir=10.0)
    for five_min in (True, False):
        out_v1, out_p = v1.run(8, dose, five_min=five_min), palerm.run(8, dose, five_min=five_min)
        assert [a.shape for a in out_v1] == [a.shape for a in out_p]
    t, bg_delta, bg, iob = v1.run(24, dose, five_min=True)
    assert t[0] == 0 and t[1] == 5
    assert iob[0] == pytest.approx(dose) and bg[0] == 0
    assert np.all(np.diff(iob) <= 1e-12)
    assert np.allclose(np.cumsum(bg_delta), bg)
    assert bg[-1] == pytest.approx(-isf * dose, rel=1e-3)
    assert np.allclose(iob, dose + bg / isf)  # insulin on board = share of the effect still to come


def test_existing_insulin_types_are_unchanged():
    kwargs = dict(insulin_sensitivity_factor=40.0, carb_insulin_ratio=10.0)
    for insulin_type in ("rapid_acting_adult", "fiasp"):
        old = SimpleMetabolismModel(**kwargs, patient_insulin_type=insulin_type)
        new = SimpleMetabolismModelV1(**kwargs, patient_insulin_type=insulin_type)
        assert type(new.insulin_model) is PalermInsulinModel
        for a, b in zip(old.run(carb_amount=30, insulin_amount=3.0), new.run(carb_amount=30, insulin_amount=3.0)):
            assert np.allclose(a, b, equal_nan=True)


def test_v1_metabolism_model_uses_three_stage_curve():
    model = SimpleMetabolismModelV1(insulin_sensitivity_factor=40.0, carb_insulin_ratio=10.0,
                                    patient_insulin_type="ultra_rapid_v1", insulin_timing="slower")
    assert isinstance(model.insulin_model, InsulinActionV1Model)
    assert model.insulin_model.time_constants == pytest.approx(scaled("ultra_rapid_v1", "slower"))
    # Steady-state basal insulin on board uses the v1 curve: about basal rate x mean action time (hours).
    iob = model.get_iob_from_sbr(1.0)
    assert iob[0] == pytest.approx(sum(scaled("ultra_rapid_v1", "slower")) / 60, rel=0.03)


@pytest.mark.parametrize("insulin_type, timing", [("rapid_acting_adult", "faster"), ("rapid_acting_v1", "fast")])
def test_invalid_settings_raise(insulin_type, timing):
    with pytest.raises(ValueError):
        SimpleMetabolismModelV1(insulin_sensitivity_factor=40.0, carb_insulin_ratio=10.0,
                                patient_insulin_type=insulin_type, insulin_timing=timing)


def make_patient(insulin_type, timing="main", metabolism_model=SimpleMetabolismModelV1):
    t0, patient_config = get_canonical_risk_patient_config()
    t0, sensor_config = get_canonical_sensor_config()
    t0, pump_config = get_canonical_risk_pump_config()
    patient_config.patient_insulin_type = insulin_type
    patient_config.insulin_timing = timing
    return VirtualPatient(time=DATETIME_DEFAULT, pump=ContinuousInsulinPump(pump_config, t0),
                          sensor=IdealSensor(t0, sensor_config), metabolism_model=metabolism_model,
                          patient_config=patient_config)


def test_virtual_patient_builds_v1_curve_from_config():
    patient = make_patient("rapid_acting_v1", "faster")
    model = patient.instantiate_metabolism_model()
    assert model.insulin_model.time_constants == pytest.approx(scaled("rapid_acting_v1", "faster"))
    assert type(make_patient("rapid_acting_adult").instantiate_metabolism_model().insulin_model) is PalermInsulinModel


def test_virtual_patient_rejects_v1_type_with_old_metabolism_model():
    patient = make_patient("ultra_rapid_v1", metabolism_model=SimpleMetabolismModel)
    with pytest.raises(ValueError, match="SimpleMetabolismModelV1"):
        patient.instantiate_metabolism_model()


def insulin_only_glucose_change(insulin_type):
    patient = make_patient(insulin_type)
    patient.bg_current = 110.0
    with_insulin, _, _ = patient.run_metabolism_model(2.0, 0, 180, 0)
    without_insulin, _, _ = patient.run_metabolism_model(0.0, 0, 180, 0)
    return with_insulin - without_insulin


def test_v1_lag_lowers_glucose_later_than_palerm():
    bg_delta_v1 = insulin_only_glucose_change("rapid_acting_v1")
    bg_delta_old = insulin_only_glucose_change("rapid_acting_adult")
    first_hour = slice(0, 13)
    assert np.sum(bg_delta_v1[first_hour]) > np.sum(bg_delta_old[first_hour])  # less glucose lowering in hour 1


FIXTURE = os.path.join(os.path.dirname(__file__), "test_data", "diagramgen",
                       "Simulation-Configuration-TRSET51-fixture.json")


def test_scenario_json_sets_v1_curve(tmp_path):
    """Base scenario sets the v1 insulin type and insulin_timing; override blocks may then change them (the parser
    only overrides settings present in the base)."""
    from tidepool_data_science_simulator.makedata.scenario_json_parser_v2 import ScenarioParserV2

    config = json.load(open(FIXTURE))
    base = {"base": config["base_config"]}
    parser = ScenarioParserV2()
    parser.resolve_pointers(base)
    parser.resolve_pointers(base["base"]["patient"]["patient_model"])
    base["base"]["patient"]["patient_model"]["metabolism_settings"].update(
        {"patient_insulin_type": "rapid_acting_v1", "insulin_timing": "main"})
    config["base_config"] = base["base"]
    config["override_config"][0].setdefault("patient", {})["patient_model"] = {
        "metabolism_settings": {"patient_insulin_type": "ultra_rapid_v1", "insulin_timing": "slower"}}
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    sims = list(ScenarioParserV2(path_to_json_config=str(path)).get_sims().values())
    first = sims[0].virtual_patient.instantiate_metabolism_model().insulin_model.time_constants
    others = [s.virtual_patient.instantiate_metabolism_model().insulin_model.time_constants for s in sims[1:]]
    assert first == pytest.approx(scaled("ultra_rapid_v1", "slower"))
    assert others and all(tc == pytest.approx(scaled("rapid_acting_v1", "main")) for tc in others)


# Window over which each dose's effect is applied (8 h for existing types; for v1, at most 0.1% outside).

@pytest.mark.parametrize("insulin_type, timing, hours", [
    ("rapid_acting_v1", "faster", 8), ("rapid_acting_v1", "main", 11), ("rapid_acting_v1", "slower", 14),
    ("ultra_rapid_v1", "faster", 8), ("ultra_rapid_v1", "main", 10), ("ultra_rapid_v1", "slower", 13),
    ("rapid_acting_adult", "main", 8), ("fiasp", "main", 8), ("unknown_type", "main", 8),
])
def test_prediction_window_per_curve(insulin_type, timing, hours):
    assert prediction_horizon_hours(insulin_type, timing) == hours


@pytest.mark.parametrize("insulin_type", sorted(V1_TIME_CONSTANTS_MIN))
@pytest.mark.parametrize("timing", sorted(INSULIN_TIMING_FACTORS))
def test_window_is_the_shortest_that_leaves_at_most_the_limit_outside(insulin_type, timing):
    hours = prediction_horizon_hours(insulin_type, timing)
    outside = lambda h: 1 - effect_fraction_delivered(scaled(insulin_type, timing), np.array([h * 60.0]))[0]
    assert outside(hours) <= MAX_EFFECT_OUTSIDE_WINDOW
    assert hours == 8 or outside(hours - 1) > MAX_EFFECT_OUTSIDE_WINDOW


def test_metabolism_model_runs_over_the_window():
    kwargs = dict(insulin_sensitivity_factor=40.0, carb_insulin_ratio=10.0)
    v1 = SimpleMetabolismModelV1(**kwargs, patient_insulin_type="rapid_acting_v1", insulin_timing="slower")
    delta_bg, *_ = v1.run(carb_amount=0, insulin_amount=2.0)
    assert len(delta_bg) == 14 * 12
    assert -np.sum(delta_bg) >= 40.0 * 2.0 * (1 - MAX_EFFECT_OUTSIDE_WINDOW)  # nearly all of the effect is applied
    legacy, *_ = SimpleMetabolismModelV1(**kwargs, patient_insulin_type="rapid_acting_adult").run(
        carb_amount=0, insulin_amount=2.0)
    assert len(legacy) == 8 * 12


def test_basal_insulin_on_board_uses_the_window_and_matches_the_package_method():
    kwargs = dict(insulin_sensitivity_factor=40.0, carb_insulin_ratio=10.0)
    old = SimpleMetabolismModel(**kwargs)
    assert np.allclose(SimpleMetabolismModelV1(**kwargs).get_iob_from_sbr(1.2), old.get_iob_from_sbr(1.2))
    v1 = SimpleMetabolismModelV1(**kwargs, patient_insulin_type="ultra_rapid_v1", insulin_timing="slower")
    iob = v1.get_iob_from_sbr(1.2)
    assert len(iob) == 13 * 12
    # Same calculation as the package: with the package's curve and an 8-hour window the results are identical.
    v1.prediction_horizon_hours, v1.insulin_model = 8, old.insulin_model
    assert np.allclose(v1.get_iob_from_sbr(1.2), old.get_iob_from_sbr(1.2), atol=1e-12)


def test_virtual_patient_window_and_initial_state():
    patient = make_patient("rapid_acting_v1", "slower")
    assert patient.num_prediction_steps == 14 * 12
    patient.init()
    assert len(patient.bg_prediction) == len(patient.iob_prediction) == 14 * 12
    assert make_patient("rapid_acting_adult").num_prediction_steps == 8 * 12
