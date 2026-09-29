"""
Unit tests for TRSET-25: the four Loop effect columns
(loop_final_insulin_effect / loop_final_carb_effect / loop_final_momentum_effect
/ loop_final_rc_effect) and loop_recommended_bolus_value in
Simulation.get_results_df().

These construct a Simulation instance without running the engine (bypassing
__init__/init(), which need a full virtual patient/controller stack) and feed
get_results_df() a hand-built simulation_results dict, so the column
extraction logic is exercised directly and quickly, with no dependency on the
Swift dylib.
"""

__author__ = "Shawn Foster"

import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from tidepool_data_science_simulator.models.simulation import Simulation, SimulationState


T0 = datetime.datetime(2019, 8, 15, 12, 0, 0)


def _make_simulation(prediction_output=None, pyloopkit_recommendations=None):
    """Build a Simulation with a single fake result row, bypassing __init__."""
    sim = Simulation.__new__(Simulation)

    controller_state = SimpleNamespace(
        prediction_output=prediction_output,
        pyloopkit_recommendations=pyloopkit_recommendations,
    )
    simulation_state = SimulationState(
        patient_state=MagicMock(),
        controller_state=controller_state,
        randint=0,
        active=True,
    )
    sim.simulation_results = {T0: simulation_state}
    return sim


def _effect_series(*values):
    return [{"date": f"2019-08-15T12:0{i}:00Z", "value": v} for i, v in enumerate(values)]


class TestLoopEffectColumns:
    """loop_final_{insulin,carb,momentum,rc}_effect take the last value of their series."""

    @pytest.mark.parametrize(
        "key, column",
        [
            ("insulin_effect_values", "loop_final_insulin_effect"),
            ("carb_effect_values", "loop_final_carb_effect"),
            ("momentum_effect_values", "loop_final_momentum_effect"),
            ("retrospective_correction_effect_values", "loop_final_rc_effect"),
        ],
    )
    def test_populated_series_takes_last_value(self, key, column):
        prediction_output = {key: _effect_series(1.1, 2.2, 3.3)}
        sim = _make_simulation(prediction_output=prediction_output)
        row = sim.get_results_df().iloc[0]
        assert row[column] == 3.3

    @pytest.mark.parametrize(
        "key, column",
        [
            ("insulin_effect_values", "loop_final_insulin_effect"),
            ("carb_effect_values", "loop_final_carb_effect"),
            ("momentum_effect_values", "loop_final_momentum_effect"),
            ("retrospective_correction_effect_values", "loop_final_rc_effect"),
        ],
    )
    def test_empty_series_is_none(self, key, column):
        prediction_output = {key: []}
        sim = _make_simulation(prediction_output=prediction_output)
        row = sim.get_results_df().iloc[0]
        assert row[column] is None

    @pytest.mark.parametrize(
        "column",
        [
            "loop_final_insulin_effect",
            "loop_final_carb_effect",
            "loop_final_momentum_effect",
            "loop_final_rc_effect",
        ],
    )
    def test_missing_key_is_none(self, column):
        sim = _make_simulation(prediction_output={})
        row = sim.get_results_df().iloc[0]
        assert row[column] is None

    @pytest.mark.parametrize(
        "column",
        [
            "loop_final_insulin_effect",
            "loop_final_carb_effect",
            "loop_final_momentum_effect",
            "loop_final_rc_effect",
        ],
    )
    def test_no_prediction_output_is_none(self, column):
        """Steps where Loop isn't producing recommendations stay None."""
        sim = _make_simulation(prediction_output=None)
        row = sim.get_results_df().iloc[0]
        assert row[column] is None


class TestLoopRecommendedBolusValue:
    """loop_recommended_bolus_value: automatic if present (0.0 counts), else manual, else None."""

    def test_automatic_only(self):
        sim = _make_simulation(
            pyloopkit_recommendations={"automatic": {"bolusUnits": 1.5}}
        )
        row = sim.get_results_df().iloc[0]
        assert row["loop_recommended_bolus_value"] == 1.5

    def test_manual_only(self):
        sim = _make_simulation(
            pyloopkit_recommendations={"manual": {"amount": 2.5}}
        )
        row = sim.get_results_df().iloc[0]
        assert row["loop_recommended_bolus_value"] == 2.5

    def test_both_present_automatic_wins(self):
        sim = _make_simulation(
            pyloopkit_recommendations={
                "automatic": {"bolusUnits": 1.5},
                "manual": {"amount": 2.5},
            }
        )
        row = sim.get_results_df().iloc[0]
        assert row["loop_recommended_bolus_value"] == 1.5

    def test_neither_present_is_none(self):
        sim = _make_simulation(pyloopkit_recommendations={})
        row = sim.get_results_df().iloc[0]
        assert row["loop_recommended_bolus_value"] is None

    def test_automatic_zero_counts_as_present(self):
        """0.0 is a legitimate bolus value, not a missing one."""
        sim = _make_simulation(
            pyloopkit_recommendations={"automatic": {"bolusUnits": 0.0}}
        )
        row = sim.get_results_df().iloc[0]
        assert row["loop_recommended_bolus_value"] == 0.0
        assert row["loop_recommended_bolus_value"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
