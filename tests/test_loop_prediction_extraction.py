"""Unit tests for TRSET-24 Loop prediction/effect/COB column population.

These are dylib-free: they exercise (a) the pure mapping from a controller's
``prediction_output`` payload to the ``get_results_df()`` columns, (b) the
no-silent-swallow logging behaviour, and (c) the controller wiring with the
Swift prediction API patched to sentinels. The end-to-end run against the real
.dylib lives in ``test_loop_prediction_columns_integration.py``.
"""
import datetime
import logging
import types

import pytest

from tidepool_data_science_simulator.models.controller import ControllerState, OpenLoopController
from tidepool_data_science_simulator.models.simulation import Simulation, SimulationState
from tidepool_data_science_simulator.models.state import VirtualPatientState, PumpState
from tidepool_data_science_simulator.models import swift_controller as swift_mod
from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController


T0 = datetime.datetime(2019, 8, 15, 12, 0, 0)


def _patient_state():
    """Minimal but complete patient state for get_results_df()."""
    return VirtualPatientState(
        bg=120.0, sensor_bg=118.0, iob=1.25, ei=0.0, pump_state=PumpState()
    )


def _results_df(controller_state, active=1):
    """Run only get_results_df() over a one-step results dict (no full Simulation)."""
    results = {
        T0: SimulationState(
            patient_state=_patient_state(),
            controller_state=controller_state,
            randint=0,
            active=active,
        )
    }
    sim_like = types.SimpleNamespace(simulation_results=results)
    return Simulation.get_results_df(sim_like)


# --- U1: payload maps to the right columns -----------------------------------

def test_prediction_output_maps_to_columns():
    payload = {
        "predicted_glucose_values": [300.0, 250.0, 180.0, 99.81],
        "predicted_glucose_dates": ["a", "b", "c", "d"],
        "glucose_effect_velocity_values": [0.0, 1.0, 2.5],
        "active_carbs": 12.0,
        "active_insulin": 0.0064,
    }
    rec = {"automatic": {"bolusUnits": 0.0, "basalAdjustment": {"unitsPerHour": 0.5, "duration": 1800}}}
    df = _results_df(ControllerState(pyloopkit_recommendations=rec, prediction_output=payload))

    row = df.loc[T0]
    assert row["loop_final_glucose_pred"] == 99.81            # final value convention
    assert row["loop_final_counteraction_effect"] == 2.5      # final ICE value
    assert row["loop_cob"] == 12.0
    # existing recommendation-backed columns unchanged
    assert row["loop_automatic_bolus_rec"] == 0.0
    assert row["loop_recommended_temp_basal_value"] == 0.5
    assert row["loop_temp_basal_duration_sec"] == 1800


def test_cob_zero_is_preserved_not_treated_as_missing():
    payload = {
        "predicted_glucose_values": [110.0, 110.0],
        "glucose_effect_velocity_values": [0.0],
        "active_carbs": 0.0,
        "active_insulin": 0.0,
    }
    df = _results_df(ControllerState(pyloopkit_recommendations=None, prediction_output=payload))
    assert df.loc[T0]["loop_cob"] == 0.0


# --- AC#3 / D1 / D5: no fabrication where data doesn't exist ------------------

def test_no_prediction_output_leaves_prediction_columns_none():
    """DoNothing-style step: recommendation present, no prediction payload."""
    df = _results_df(ControllerState(pyloopkit_recommendations=None, prediction_output=None))
    row = df.loc[T0]
    for col in ["loop_final_glucose_pred", "loop_final_counteraction_effect", "loop_cob"]:
        assert row[col] is None


def test_unbacked_effect_columns_and_bolus_value_stay_none():
    """D1: the four effect columns have no API source. D5: loop_recommended_bolus_value deprecated."""
    payload = {
        "predicted_glucose_values": [120.0],
        "glucose_effect_velocity_values": [0.0],
        "active_carbs": 0.0,
        "active_insulin": 0.0,
    }
    df = _results_df(ControllerState(pyloopkit_recommendations={"automatic": {"bolusUnits": 1.0}},
                                     prediction_output=payload))
    row = df.loc[T0]
    for col in ["loop_final_insulin_effect", "loop_final_carb_effect",
                "loop_final_momentum_effect", "loop_final_rc_effect",
                "loop_recommended_bolus_value"]:
        assert row[col] is None


# --- U2a: no silent swallow in get_results_df --------------------------------

def test_malformed_prediction_output_is_logged_not_swallowed(caplog):
    # prediction_output is a list -> `.get` raises AttributeError inside the try.
    bad_state = ControllerState(pyloopkit_recommendations=None, prediction_output=["not", "a", "dict"])
    with caplog.at_level(logging.WARNING, logger="tidepool_data_science_simulator.models.simulation"):
        df = _results_df(bad_state)
    assert any("Error extracting loop outputs" in r.message for r in caplog.records)
    # degrades gracefully: columns are empty, no exception raised
    assert df.loc[T0]["loop_final_glucose_pred"] is None


# --- U3: controller wiring reuses the same input dict ------------------------

def test_compute_prediction_output_assembles_payload_from_same_dict(monkeypatch):
    seen = {}
    sentinel_dict = {"predictionStart": "2019-08-15T12:00:00Z"}

    monkeypatch.setattr(swift_mod, "get_prediction_values_and_dates",
                        lambda d: (seen.setdefault("pred", d), ([300.0, 99.0], ["a", "b"]))[1])
    monkeypatch.setattr(swift_mod, "get_glucose_velocity_values_and_dates",
                        lambda d: (seen.setdefault("ice", d), ([0.0, 2.0], ["a", "b"]))[1])
    monkeypatch.setattr(swift_mod, "get_active_carbs",
                        lambda d: (seen.setdefault("cob", d), 7.5)[1])
    monkeypatch.setattr(swift_mod, "get_active_insulin",
                        lambda d: (seen.setdefault("iob", d), 0.5)[1])

    controller = SwiftLoopController.__new__(SwiftLoopController)  # bypass heavy __init__
    controller.time = T0
    payload = controller._compute_prediction_output(sentinel_dict)

    assert payload["predicted_glucose_values"] == [300.0, 99.0]
    assert payload["glucose_effect_velocity_values"] == [0.0, 2.0]
    assert payload["active_carbs"] == 7.5
    assert payload["active_insulin"] == 0.5
    # DRY: every API function received the *same* dict object, no rebuild
    for key in ["pred", "ice", "cob", "iob"]:
        assert seen[key] is sentinel_dict


# --- U2b: controller prediction failure logs and returns None ----------------

def test_compute_prediction_output_logs_and_returns_none_on_failure(monkeypatch, caplog):
    def boom(_d):
        raise RuntimeError("dylib blew up")

    monkeypatch.setattr(swift_mod, "get_prediction_values_and_dates", boom)
    controller = SwiftLoopController.__new__(SwiftLoopController)
    controller.time = T0

    with caplog.at_level(logging.WARNING, logger="tidepool_data_science_simulator.models.swift_controller"):
        result = controller._compute_prediction_output({"x": 1})

    assert result is None
    assert any("Loop prediction extraction failed" in r.message for r in caplog.records)


# --- U4: gating -- non-Swift controllers never expose prediction data --------

def test_controller_state_defaults_prediction_output_none():
    assert ControllerState(pyloopkit_recommendations={}).prediction_output is None


def test_openloop_controller_state_has_no_prediction_output():
    controller = OpenLoopController.__new__(OpenLoopController)  # bypass heavy __init__
    controller.recommendations = {"automatic": {"bolusUnits": 0}}
    controller.prediction_output = None  # set by LoopController.__init__ in real use
    state = controller.get_state()
    assert state.prediction_output is None
