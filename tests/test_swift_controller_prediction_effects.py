"""
Unit tests for TRSET-25: SwiftLoopController._compute_prediction_output() wiring
the new get_prediction_effects() bridge call into the prediction_output payload.

Mocks the loop_to_python_api.api functions (following the pattern already used
in tests/test_swift_controller_insulin_type.py) so no Swift dylib call happens.
"""

__author__ = "Shawn Foster"

import datetime
from unittest.mock import patch

import pytest

from tidepool_data_science_simulator.models.controller import AutomationControlTimeline
from tidepool_data_science_simulator.makedata.scenario_parser import ControllerConfig
from tidepool_data_science_simulator.models.events import BolusTimeline, CarbTimeline
from tidepool_data_science_simulator.models.swift_controller import SwiftLoopController


T0 = datetime.datetime(2019, 8, 15, 12, 0, 0)


def _make_controller():
    settings = {
        "model": "novolog",
        "max_basal_rate": 3.5,
        "max_bolus": 10.0,
        "suspend_threshold": 70.0,
        "partial_application_factor": 0.0,
        "use_mid_absorption_isf": False,
    }
    cfg = ControllerConfig(
        bolus_event_timeline=BolusTimeline(),
        carb_event_timeline=CarbTimeline(),
        controller_settings=settings,
    )
    ctrl = SwiftLoopController(T0, cfg, AutomationControlTimeline([], []))
    ctrl.time = T0
    return ctrl


_EFFECTS = {
    "insulin": [{"date": "2019-08-15T12:00:00Z", "value": -1.0}],
    "carbs": [{"date": "2019-08-15T12:00:00Z", "value": 2.0}],
    "momentum": [{"date": "2019-08-15T12:00:00Z", "value": 0.5}],
    "retrospectiveCorrection": [{"date": "2019-08-15T12:00:00Z", "value": -0.25}],
}


class TestComputePredictionOutputEffects:

    def test_effect_series_added_to_payload(self):
        ctrl = _make_controller()
        with patch(
            "tidepool_data_science_simulator.models.swift_controller.get_prediction_values_and_dates",
            return_value=([100.0], ["2019-08-15T12:00:00Z"]),
        ), patch(
            "tidepool_data_science_simulator.models.swift_controller.get_glucose_velocity_values_and_dates",
            return_value=([0.1], ["2019-08-15T12:00:00Z"]),
        ), patch(
            "tidepool_data_science_simulator.models.swift_controller.get_prediction_effects",
            return_value=_EFFECTS,
        ), patch(
            "tidepool_data_science_simulator.models.swift_controller.get_active_carbs",
            return_value=5.0,
        ), patch(
            "tidepool_data_science_simulator.models.swift_controller.get_active_insulin",
            return_value=1.0,
        ):
            payload = ctrl._compute_prediction_output({})

        assert payload["insulin_effect_values"] == _EFFECTS["insulin"]
        assert payload["carb_effect_values"] == _EFFECTS["carbs"]
        assert payload["momentum_effect_values"] == _EFFECTS["momentum"]
        assert payload["retrospective_correction_effect_values"] == _EFFECTS["retrospectiveCorrection"]
        # Existing TRSET-24 keys are unaffected.
        assert payload["predicted_glucose_values"] == [100.0]
        assert payload["active_carbs"] == 5.0
        assert payload["active_insulin"] == 1.0

    def test_exception_from_prediction_effects_returns_none(self):
        """A failure anywhere in the call chain must return None, not raise or
        return a partial payload (workflow section 4 / AC #10: never silently
        swallowed -- caller sees None and logs, it doesn't propagate)."""
        ctrl = _make_controller()
        with patch(
            "tidepool_data_science_simulator.models.swift_controller.get_prediction_values_and_dates",
            return_value=([100.0], ["2019-08-15T12:00:00Z"]),
        ), patch(
            "tidepool_data_science_simulator.models.swift_controller.get_glucose_velocity_values_and_dates",
            return_value=([0.1], ["2019-08-15T12:00:00Z"]),
        ), patch(
            "tidepool_data_science_simulator.models.swift_controller.get_prediction_effects",
            side_effect=RuntimeError("boom"),
        ):
            payload = ctrl._compute_prediction_output({})

        assert payload is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
