"""Unit tests for TRSET-54: get_results_df() must not crash when no pump is attached.

Dylib-free: runs only get_results_df() over a hand-built results dict, the same
way test_loop_prediction_extraction.py does.
"""
import datetime
import types

import pandas as pd
import pytest

from tidepool_data_science_simulator.models.controller import ControllerState
from tidepool_data_science_simulator.models.simulation import Simulation, SimulationState
from tidepool_data_science_simulator.models.state import VirtualPatientState, PumpState


T0 = datetime.datetime(2019, 8, 15, 12, 0, 0)
STEP = datetime.timedelta(minutes=5)

PUMP_DERIVED_COLUMNS = [
    "reported_bolus",
    "reported_carb_value",
    "reported_carb_duration",
    "pump_sbr",
    "pump_isf",
    "pump_cir",
    "temp_basal",
    "temp_basal_time_remaining",
    "delivered_basal_insulin",
    "undelivered_basal_insulin",
]


def _state(pump_attached):
    patient_state = VirtualPatientState(
        bg=120.0,
        sensor_bg=118.0,
        iob=1.25,
        ei=0.0,
        pump_state=PumpState() if pump_attached else None,
    )
    return SimulationState(
        patient_state=patient_state,
        controller_state=ControllerState(pyloopkit_recommendations=None),
        randint=0,
        active=1,
    )


def _results_df(pump_attached_by_step):
    results = {
        T0 + i * STEP: _state(attached)
        for i, attached in enumerate(pump_attached_by_step)
    }
    sim_like = types.SimpleNamespace(simulation_results=results)
    return Simulation.get_results_df(sim_like)


def test_no_pump_steps_do_not_crash():
    df = _results_df([False, False])
    assert len(df) == 2


def test_no_pump_rows_have_null_pump_derived_columns():
    df = _results_df([True, False, False, True])
    no_pump_rows = df.iloc[1:3]
    for col in PUMP_DERIVED_COLUMNS:
        assert col in df.columns
        assert no_pump_rows[col].isna().all(), col


def test_attached_rows_unchanged_by_removal_and_reattach():
    attached_only = _results_df([True, True])
    attach_remove_attach = _results_df([True, False, True])

    expected = attached_only.iloc[[0]]
    for row in (attach_remove_attach.iloc[[0]], attach_remove_attach.iloc[[2]]):
        pd.testing.assert_frame_equal(
            row.reset_index(drop=True),
            expected.reset_index(drop=True),
            check_dtype=False,
        )


def test_column_order_unchanged_with_no_pump():
    with_pump = _results_df([True])
    without_pump = _results_df([False])
    assert list(with_pump.columns) == list(without_pump.columns)
